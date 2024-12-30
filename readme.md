# 🎙️ Whisper Server

Сервер для распознавания речи на базе Whisper Large V3 Turbo. Работает с GPU через CUDA.

## 🚀 Быстрый запуск в VS Code (для разработки)

```bash
# 1️⃣ Клонируем репозиторий
git clone https://github.com/PKG-7/whisperserver
cd whisperserver

# 2. Создаем и активируем окружение
python -m venv venv
source venv/bin/activate  # Linux
# или
venv\Scripts\activate     # Windows

# 3. Устанавливаем зависимости
pip install -r requirements.txt

# 4. Запускаем для разработки
python main.py
```

## 🖥️ Установка на сервер (для продакшена)

```bash
# 1. Клонируем репозиторий
git clone https://github.com/PKG-7/whisperserver
cd whisperserver

# 2. Создаем и активируем окружение
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Запускаем сервер напрямую
python main.py

# или Устанавливаем systemd сервис не работает сейчас
sudo cp whisper-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable whisper-server
sudo systemctl start whisper-server
```

## 🔄 Обновление на сервере

```bash
# 1. Переходим в директорию
cd whisperserver

# 2. Получаем обновления
git pull origin main

# 3. Активируем окружение и обновляем зависимости
source venv/bin/activate
pip install -r requirements.txt

# 4. Перезапускаем сервис
sudo systemctl restart whisper-server
```

## 🔍 Проверка работы сервера

```bash
# Статус сервиса
sudo systemctl status whisper-server

# Просмотр логов
sudo journalctl -u whisper-server -f
# или
tail -f /var/log/whisper-server/whisper_server.log
```

## 📁 Структура проекта

```
whisperserver/
├── main.py                # Основной файл сервера
├── requirements.txt       # Зависимости
├── whisper-server.service # Конфигурация systemd
├── models/               # Кэш моделей (создается автоматически)
├── temp/                 # Временные файлы (создается автоматически)
└── venv/                 # Виртуальное окружение
```

## 💻 Требования

-   Python 3.8+
-   CUDA-совместимая видеокарта (для GPU)
-   16GB RAM минимум
-   20GB свободного места (для моделей)

## ⚠️ Возможные проблемы

1. **Ошибка доступа к логам или директориям**

    ```bash
    # Создаем директории и устанавливаем права
    sudo mkdir -p /var/log/whisper-server
    sudo mkdir -p /home/pkgpu/whisperserver/models
    sudo mkdir -p /home/pkgpu/whisperserver/temp

    # Устанавливаем владельца
    sudo chown -R $USER:$USER /var/log/whisper-server
    sudo chown -R $USER:$USER /home/pkgpu/whisperserver/models
    sudo chown -R $USER:$USER /home/pkgpu/whisperserver/temp

    # Устанавливаем права
    sudo chmod 755 /home/pkgpu/whisperserver/models
    sudo chmod 755 /home/pkgpu/whisperserver/temp

    # Перезапускаем сервис
    sudo systemctl restart whisper-server
    ```

2. **Ошибка CUDA**

    ```bash
    # Проверить версию CUDA
    nvidia-smi

    # Установить CUDA если нет
    sudo apt install nvidia-cuda-toolkit
    ```

3. **Сервис не запускается**
    ```bash
    # Проверить логи
    sudo journalctl -u whisper-server -n 50
    ```

## 🌐 API Endpoints

-   `POST /transcribe` - Распознавание речи из WAV файла
-   `GET /health` - Проверка состояния сервера

## 📊 Мониторинг

```bash
# Использование GPU
nvidia-smi -l 1

# Использование CPU и RAM
htop

# Логи сервера
sudo journalctl -u whisper-server -f
```
