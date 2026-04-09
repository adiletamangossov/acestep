FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    git ffmpeg \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch 2.6.0 + cu124
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ACE-Step 1.5
RUN git clone --depth 1 https://github.com/ace-step/ACE-Step-1.5.git /app/ACE-Step-1.5
WORKDIR /app/ACE-Step-1.5

# Install ACE-Step deps (skip their torch pin, we already have ours)
RUN pip install --no-cache-dir -e . || true
RUN pip install --no-cache-dir \
    "transformers>=4.51.0" \
    "diffusers>=0.37.0" \
    "vector-quantize-pytorch>=1.27.15,<1.28.0" \
    "accelerate>=1.12.0" \
    "soundfile>=0.13.1" \
    "einops>=0.8.1" \
    "scipy>=1.10.1" \
    "numba>=0.63.1" \
    safetensors \
    "peft>=0.18.0" \
    huggingface_hub

# RunPod SDK
RUN pip install --no-cache-dir runpod

# Flash attention (optional, speeds up inference)
RUN pip install --no-cache-dir flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn build skipped"

# Download model weights
ARG HF_TOKEN
COPY download_weights.py /app/download_weights.py
RUN HF_TOKEN=${HF_TOKEN} python /app/download_weights.py

# Copy handler
COPY handler.py /app/handler.py

CMD ["python", "/app/handler.py"]