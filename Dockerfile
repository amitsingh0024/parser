# Dockerfile for Baidu Unlimited-OCR-2 (Linux / CUDA with Native Triton Acceleration)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    git \
    curl \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 & native Linux Triton
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    triton

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    fastapi \
    uvicorn \
    python-multipart \
    pymupdf

# Copy application source code
COPY . /app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
