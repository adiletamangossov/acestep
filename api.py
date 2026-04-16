"""FastAPI server for ACE-Step 1.5 XL inference and LoRA training."""

import base64
import os
import random
import subprocess
import tempfile
import pathlib
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Patch vector_quantize_pytorch before any model import
# ---------------------------------------------------------------------------

def patch_vector_quantize():
    site = pathlib.Path("/usr/local/lib/python3.11/site-packages/vector_quantize_pytorch")

    f1 = site / "residual_fsq.py"
    if f1.exists():
        t = f1.read_text()
        if "assert (levels_tensor > 1).all()" in t:
            f1.write_text(t.replace("assert (levels_tensor > 1).all()", "pass  # patched"))

    f2 = site / "finite_scalar_quantization.py"
    if f2.exists():
        t = f2.read_text()
        old = "self.codebook_size = self._levels.prod().item()"
        new = (
            'self.codebook_size = int(self._levels.prod().item()) '
            'if self._levels.device.type != "meta" '
            'else int(__import__("functools").reduce(lambda a,b: a*b, levels))'
        )
        if old in t:
            f2.write_text(t.replace(old, new))

    for pyc in site.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Global model handles
# ---------------------------------------------------------------------------

dit_handler = None
llm_handler = None
_training_status = {"running": False, "message": "idle"}


def load_models():
    global dit_handler, llm_handler
    patch_vector_quantize()

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    dit_handler = AceStepHandler()
    msg, ok = dit_handler.initialize_service(
        project_root="/weights",
        config_path="acestep-v15-xl-turbo",
        device="cuda",
    )
    print(f"DiT init: {msg} (ok={ok})", flush=True)

    llm_handler = LLMHandler()
    msg, ok = llm_handler.initialize(
        checkpoint_dir="/weights/checkpoints",
        lm_model_path="acestep-5Hz-lm-4B",
        backend="pt",
        device="cuda",
    )
    print(f"LLM init: {msg} (ok={ok})", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(title="ACE-Step 1.5 XL API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str = ""
    lyrics: str = ""
    duration: float = Field(default=30.0, ge=10, le=600)
    seed: Optional[int] = None
    bpm: Optional[int] = Field(default=None, ge=30, le=300)
    keyscale: str = ""
    vocal_language: str = "en"
    instrumental: bool = False


class GenerateResponse(BaseModel):
    status: str
    audio_base64: str
    mime_type: str = "audio/mp3"
    seed: int
    actual_bpm: Optional[int] = None
    actual_key: Optional[str] = None


class TrainRequest(BaseModel):
    dataset_dir: str = Field(description="Path to dataset directory with audio + lyrics/metadata files")
    output_dir: str = Field(default="/weights/lora_output", description="Where to save trained LoRA adapter")
    epochs: int = Field(default=100, ge=1, le=1000)
    learning_rate: float = Field(default=1e-4, gt=0)
    batch_size: int = Field(default=1, ge=1, le=8)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    lora_rank: int = Field(default=8, ge=1, le=128)
    lora_alpha: int = Field(default=16, ge=1, le=256)
    save_every_n_epochs: int = Field(default=10, ge=1)
    gradient_checkpointing: bool = True


class LoraLoadRequest(BaseModel):
    lora_path: str = Field(description="Path to LoRA adapter directory")
    adapter_name: str = Field(default="default", description="Name for the adapter")
    scale: float = Field(default=1.0, ge=0.0, le=2.0)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if dit_handler is None:
        raise HTTPException(503, "Models not loaded yet")
    if _training_status["running"]:
        raise HTTPException(409, "Training in progress, inference unavailable")

    seed = req.seed if req.seed is not None else random.randint(0, 2**32 - 1)

    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    params = GenerationParams(
        task_type="text2music",
        caption=req.prompt,
        lyrics=req.lyrics,
        duration=req.duration,
        bpm=req.bpm,
        keyscale=req.keyscale,
        vocal_language=req.vocal_language,
        instrumental=req.instrumental,
        thinking=True,
    )

    config = GenerationConfig(
        batch_size=1,
        seeds=[seed],
        audio_format="wav",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        result = generate_music(dit_handler, llm_handler, params, config, save_dir=tmpdir)

        if not result.success:
            raise HTTPException(500, result.error or result.status_message)

        wav_path = result.audios[0]["path"]
        mp3_path = wav_path.replace(".wav", ".mp3")
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path,
             "-codec:a", "libmp3lame", "-qscale:a", "2",
             mp3_path],
            check=True, capture_output=True,
        )

        with open(mp3_path, "rb") as f:
            mp3_bytes = f.read()

    lm_meta = result.extra_outputs.get("lm_metadata", {})

    return GenerateResponse(
        status="ok",
        audio_base64=base64.b64encode(mp3_bytes).decode(),
        seed=seed,
        actual_bpm=lm_meta.get("bpm"),
        actual_key=lm_meta.get("keyscale"),
    )


# ---------------------------------------------------------------------------
# LoRA Training
# ---------------------------------------------------------------------------

def _run_training(req: TrainRequest):
    global _training_status
    _training_status = {"running": True, "message": "preprocessing dataset..."}

    try:
        checkpoint_dir = "/weights/checkpoints"
        os.makedirs(req.output_dir, exist_ok=True)

        # Step 1: Preprocess audio to tensors
        preprocessed_dir = os.path.join(req.output_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)

        _training_status["message"] = "preprocessing audio to tensors..."
        subprocess.run([
            "python", "train.py", "fixed",
            "--preprocess",
            "--audio-dir", req.dataset_dir,
            "--tensor-output", preprocessed_dir,
            "--checkpoint-dir", checkpoint_dir,
            "--model-variant", "xl_turbo",
        ], check=True, cwd="/app/ACE-Step-1.5")

        # Step 2: Train LoRA
        _training_status["message"] = f"training LoRA (0/{req.epochs} epochs)..."
        subprocess.run([
            "python", "train.py", "fixed",
            "--tensor-dir", preprocessed_dir,
            "--checkpoint-dir", checkpoint_dir,
            "--model-variant", "xl_turbo",
            "--output-dir", req.output_dir,
            "--epochs", str(req.epochs),
            "--lr", str(req.learning_rate),
            "--batch-size", str(req.batch_size),
            "--grad-accum", str(req.gradient_accumulation_steps),
            "--lora-rank", str(req.lora_rank),
            "--lora-alpha", str(req.lora_alpha),
            "--save-every", str(req.save_every_n_epochs),
        ] + (["--gradient-checkpointing"] if req.gradient_checkpointing else []),
            check=True, cwd="/app/ACE-Step-1.5")

        _training_status = {"running": False, "message": f"done. adapter saved to {req.output_dir}"}

    except subprocess.CalledProcessError as e:
        _training_status = {"running": False, "message": f"training failed: {e}"}
    except Exception as e:
        _training_status = {"running": False, "message": f"error: {e}"}


@app.post("/train")
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    if _training_status["running"]:
        raise HTTPException(409, "Training already in progress")

    if not os.path.isdir(req.dataset_dir):
        raise HTTPException(400, f"Dataset directory not found: {req.dataset_dir}")

    background_tasks.add_task(_run_training, req)
    return {"status": "started", "message": "Training started in background. Check /train/status for progress."}


@app.get("/train/status")
def training_status():
    return _training_status


# ---------------------------------------------------------------------------
# LoRA Management
# ---------------------------------------------------------------------------

@app.post("/lora/load")
def load_lora(req: LoraLoadRequest):
    if dit_handler is None:
        raise HTTPException(503, "Models not loaded yet")

    if not os.path.isdir(req.lora_path):
        raise HTTPException(400, f"LoRA path not found: {req.lora_path}")

    dit_handler.add_lora(req.lora_path, adapter_name=req.adapter_name)
    dit_handler.set_lora_scale(req.adapter_name, req.scale)
    return {"status": "ok", "message": f"LoRA '{req.adapter_name}' loaded from {req.lora_path}"}


@app.post("/lora/unload")
def unload_lora():
    if dit_handler is None:
        raise HTTPException(503, "Models not loaded yet")

    dit_handler.unload_lora()
    return {"status": "ok", "message": "All LoRA adapters unloaded"}


@app.get("/lora/status")
def lora_status():
    if dit_handler is None:
        raise HTTPException(503, "Models not loaded yet")

    return dit_handler.get_lora_status()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": dit_handler is not None,
        "training": _training_status,
    }
