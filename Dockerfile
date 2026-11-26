FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HEADLESS=1 \
    PORT=8000

# Install system dependencies (OpenCV requirements, FFmpeg, audio development libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure working directory and models directories exist
RUN mkdir -p logs recordings reports uploads models

# Expose default port
EXPOSE 8000

# Start command: runs simulation pipeline and web dashboard headless on Render's assigned $PORT
CMD ["sh", "-c", "python main.py --sim --no-window --port ${PORT:-8000}"]
