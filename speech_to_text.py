from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import sounddevice as sd
from scipy.io import wavfile
import keyboard
import winsound
import time
import torch
import numpy as np
import os

# вариант где все происходит на одном устройстве


# Инициализация модели
print("Загрузка модели Whisper...")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3-turbo").to(device)

print(f"Модель загружена! Используется: {device.upper()}")

is_recording = False
audio_data = []
SAMPLE_RATE = 16000

def play_start_beep():
    # winsound.Beep(1500, 200)
    # time.sleep(0.1)
    winsound.Beep(1500, 200)

def play_stop_beep():
    winsound.Beep(400, 400)

def audio_callback(indata, frames, time, status):
    if is_recording:
        audio_data.append(indata.copy())

def process_audio(audio_file):
    # Загружаем аудио файл
    sample_rate, audio_data = wavfile.read(audio_file)
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

print("\nНажмите F7 для начала/остановки записи")
print("Для выхода нажмите Ctrl+C")

# Открываем поток для записи
with sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE, dtype=np.int16):
    while True:
        try:
            if keyboard.is_pressed('f7'):
                is_recording = not is_recording
                if is_recording:
                    print("\n🎤 ЗАПИСЬ НАЧАТА")
                    play_start_beep()
                    audio_data.clear()  # Очищаем буфер перед новой записью
                else:
                    print("\n⏹️ ЗАПИСЬ ОСТАНОВЛЕНА")
                    play_stop_beep()
                    
                    if len(audio_data) > 0:
                        # Сохраняем записанное аудио
                        recorded_audio = np.concatenate(audio_data, axis=0)
                        wavfile.write('recorded_audio.wav', SAMPLE_RATE, recorded_audio)
                        print("\nОбработка записи...")
                        
                        # Распознаем речь
                        transcription = process_audio('recorded_audio.wav')
                        print(f"\nРаспознанный текст: {transcription}")
                
                # Ждем, пока клавиша будет отпущена
                while keyboard.is_pressed('f7'):
                    time.sleep(0.1)
                time.sleep(0.3)
                
        except KeyboardInterrupt:
            print("\nПрограмма завершена")
            break
        
        except Exception as e:
            print(f"\nОшибка: {e}") 