# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ACE-Step 1.5 XL music generation model — inference and LoRA training. Deployed on a supercomputer via Docker. Also supports RunPod Serverless mode.

## Architecture

- **api.py** — FastAPI server with endpoints for inference (`/generate`), LoRA training (`/train`), LoRA management (`/lora/*`), and health check (`/health`). Default entry point.
- **handler.py** — RunPod Serverless handler (alternative entry point).
- **download_weights.py** — Downloads model weights from HuggingFace during Docker build.
- **Dockerfile** — CUDA 12.8 + Python 3.11 + PyTorch 2.10.0 image. Default CMD runs FastAPI on port 8000. Override with `CMD ["python", "/app/handler.py"]` for RunPod.

## Build & Deploy

```bash
# Build (Linux x86_64, needs ~50GB disk)
docker build --build-arg HF_TOKEN=$HF_TOKEN -t innlabkz/acestep-runpod:latest .

# Run FastAPI (default)
docker run --gpus all -p 8000:8000 innlabkz/acestep-runpod:latest

# Run RunPod handler
docker run --gpus all innlabkz/acestep-runpod:latest python /app/handler.py
```

## API Endpoints

- `POST /generate` — music generation, returns `{status, audio_base64, mime_type, seed, actual_bpm, actual_key}`
- `POST /train` — start LoRA training in background (dataset_dir, output_dir, epochs, lr, lora_rank...)
- `GET /train/status` — check training progress
- `POST /lora/load` — load LoRA adapter (lora_path, adapter_name, scale)
- `POST /lora/unload` — unload all LoRA adapters
- `GET /lora/status` — current LoRA adapter state
- `GET /health` — health check

## Key Implementation Details

- Models: `acestep-v15-xl-turbo` (4B DiT) + `acestep-5Hz-lm-4B` (LLM). Weights at `/weights/checkpoints/`.
- BPM/key metadata comes from `result.extra_outputs["lm_metadata"]`, NOT from `result.audios[i]`.
- WAV generated first, then converted to MP3 via ffmpeg (`libmp3lame -qscale:a 2`).
- Two patches required for `vector_quantize_pytorch`: assertion removal in `residual_fsq.py` and meta-device fallback in `finite_scalar_quantization.py`. Patch auto-detects `dist-packages` vs `site-packages`.
- LoRA training: `convert2hf_dataset.py` converts raw audio to HF dataset, then `trainer.py` trains with LoRA config JSON. Uses PyTorch Lightning.
- LoRA loading via `dit_handler.add_lora()` / `dit_handler.unload_lora()`. Not compatible with quantized models.
- ACE-Step cloned at `/app/ACE-Step-1.5` inside the container.
