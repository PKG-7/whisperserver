from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import numpy as np
from scipy.io import wavfile
import os
import uvicorn
import signal
import sys
from contextlib import contextmanager
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('whisper_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные для хранения ресурсов
global_processor = None
global_model = None
global_device = None

def cleanup():
    """Очистка ресурсов при выключении"""
    logger.info("Выполняется очистка ресурсов...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Ресурсы очищены")

def signal_handler(signum, frame):
    """Обработчик сигналов для корректного завершения"""
    logger.info(f"Получен сигнал {signum}. Выполняется корректное завершение...")
    cleanup()
    sys.exit(0)

# Регистрация обработчиков сигналов
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def initialize_model():
    """Инициализация модели с обработкой ошибок"""
    global global_processor, global_model, global_device
    try:
        logger.info("Загрузка модели Whisper...")
        global_device = "cuda" if torch.cuda.is_available() else "cpu"
        global_processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
        global_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo").to(global_device)
        logger.info(f"Модель загружена! Используется: {global_device.upper()}")
        return True
    except Exception as e:
        logger.error(f"Ошибка при инициализации модели: {str(e)}")
        return False

# Инициализация FastAPI
app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@contextmanager
def handle_errors():
    """Контекстный менеджер для обработки ошибок"""
    try:
        yield
    except Exception as e:
        logger.error(f"Произошла ошибка: {str(e)}")
        cleanup()
        initialize_model()
        raise

def process_audio(audio_file_path):
    with handle_errors():
        sample_rate, audio_data = wavfile.read(audio_file_path)
        audio_data = audio_data.astype(np.float32) / 32768.0
        
        inputs = global_processor(
            audio_data, 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        input_features = inputs.input_features.to(global_device)
        attention_mask = torch.ones_like(input_features[:, :, 0], dtype=torch.long).to(global_device)
        
        forced_decoder_ids = global_processor.get_decoder_prompt_ids(language="ru", task="transcribe")
        
        predicted_ids = global_model.generate(
            input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_decoder_ids,
            max_length=448,
        )
        
        transcription = global_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        file_path = "temp_audio.wav"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        transcription = process_audio(file_path)
        
        os.remove(file_path)
        logger.info("Транскрипция успешно выполнена")
        return {"transcription": transcription}
    except Exception as e:
        logger.error(f"Ошибка при обработке запроса: {str(e)}")
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    """Эндпоинт для проверки здоровья сервера"""
    return {"status": "healthy", "device": global_device}

if __name__ == "__main__":
    if not initialize_model():
        logger.error("Не удалось инициализировать модель. Завершение работы.")
        sys.exit(1)
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8007,
        workers=1,
        log_level="info"
    ) 