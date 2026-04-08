# Используем легковесный образ с предустановленным Python и PyTorch
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Устанавливаем системные зависимости для OpenCV и Kivy
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Указываем переменную окружения для корректной работы python
ENV PYTHONUNBUFFERED=1

# По умолчанию запускаем скрипт обучения (или main.py)
# Для мобильной сборки Docker используется как сборочная среда
CMD ["python", "training.py"]
