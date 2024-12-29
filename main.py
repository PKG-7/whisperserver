from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import numpy as np
from scipy.io import wavfile
import os
import uvicorn

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

# Инициализация модели
print("Загрузка модели Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo").to(device)
print(f"Модель загружена! Используется: {device.upper()}")

def process_audio(audio_file_path):
    # Загружаем аудио файл
    sample_rate, audio_data = wavfile.read(audio_file_path)
    # Преобразуем в float32 и нормализуем
    audio_data = audio_data.astype(np.float32) / 32768.0
    
    # Обрабатываем через модель с явным указанием русского языка
    inputs = processor(
        audio_data, 
        sampling_rate=sample_rate, 
        return_tensors="pt",
        padding=True
    )
    
    input_features = inputs.input_features.to(device)
    
    # Создаем маску внимания вручную
    attention_mask = torch.ones_like(input_features[:, :, 0], dtype=torch.long).to(device)
    
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="ru", task="transcribe")
    
    predicted_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
        forced_decoder_ids=forced_decoder_ids,
        max_length=448,
    )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Сохраняем загруженный файл
        file_path = "temp_audio.wav"
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Обрабатываем аудио
        transcription = process_audio(file_path)
        
        # Удаляем временный файл
        os.remove(file_path)
        
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8007) 