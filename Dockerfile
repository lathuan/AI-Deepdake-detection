# Dùng base image Python 3.11 (Debian 13)
FROM python:3.11-slim

# Đặt thư mục làm việc
WORKDIR /app

# Sao chép toàn bộ code vào container
COPY . /app

# Cài các thư viện hệ thống cần thiết cho OpenCV, TensorFlow, Gunicorn
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Nâng cấp pip và cài các thư viện Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Cấu hình biến môi trường
ENV PORT=5000

# Mở port 5000 cho Render
EXPOSE 5000

# Dùng Gunicorn để chạy Flask app trong môi trường production
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000", "--workers", "2"]
