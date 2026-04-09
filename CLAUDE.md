# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RunPod Serverless deployment for ACE-Step 1.5 XL music generation model. A Go backend sends requests; this service generates music and returns MP3 audio as base64.

## Architecture

- **handler.py** — RunPod Serverless entry point. Loads DiT (XL turbo 4B) + LLM (4B) at worker startup, handles generation requests. Patches `vector_quantize_pytorch` for meta-tensor compatibility before model loading.
- **download_weights.py** — Runs during Docker build to download model weights from HuggingFace (main model, 4B LLM, XL turbo DiT).
- **Dockerfile** — CUDA 12.4 + Python 3.11 + PyTorch 2.6.0 image with ACE-Step 1.5 cloned from GitHub and weights baked in.

## Build & Deploy

```bash
# Build (on Linux x86_64, needs ~50GB disk)
export HF_TOKEN=your_token
docker build --build-arg HF_TOKEN=$HF_TOKEN -t innlabkz/acestep-runpod:latest .

# Push
docker push innlabkz/acestep-runpod:latest
```

RunPod Serverless config: GPU 24GB+ (RTX 4090 / A40), Container disk 50GB, Execution timeout 300s.

## API Contract

Go backend expects this exact response shape:

```json
{"status": "ok", "audio_base64": "...", "mime_type": "audio/mp3", "seed": 123, "actual_bpm": 120, "actual_key": "C major"}
```

Error response: `{"status": "error", "message": "...", "trace": "..."}`.

## Key Implementation Details

- Models: `acestep-v15-xl-turbo` (DiT) + `acestep-5Hz-lm-4B` (LLM). Weights stored at `/weights/checkpoints/`.
- BPM/key metadata comes from `result.extra_outputs["lm_metadata"]`, NOT from `result.audios[i]`.
- WAV generated first, then converted to MP3 via ffmpeg (`libmp3lame -qscale:a 2`).
- Two patches required for `vector_quantize_pytorch`: assertion removal in `residual_fsq.py` and meta-device fallback in `finite_scalar_quantization.py`.
- ACE-Step is installed from git clone at `/app/ACE-Step-1.5` inside the container.
