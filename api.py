"""FastAPI server for ACE-Step 1.5 XL inference and LoRA training."""

import base64
import json
import os
import random
import subprocess
import tempfile
import pathlib
from contextlib import asynccontextmanager
from typing import Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Patch vector_quantize_pytorch before any model import
# ---------------------------------------------------------------------------

def _find_vq_dir():
    """Find vector_quantize_pytorch in either site-packages or dist-packages."""
    for variant in ["dist-packages", "site-packages"]:
        p = pathlib.Path(f"/usr/local/lib/python3.11/{variant}/vector_quantize_pytorch")
        if p.exists():
            return p
    return None


def patch_vector_quantize():
    site = _find_vq_dir()
    if site is None:
        print("WARNING: vector_quantize_pytorch not found, skipping patch", flush=True)
        return

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

    print(f"Patched vector_quantize_pytorch at {site}", flush=True)


# ---------------------------------------------------------------------------
# Global model handles
# ---------------------------------------------------------------------------

dit_handler = None
llm_handler = None
_training_status = {"running": False, "message": "idle"}

CHECKPOINTS_DIR = "/weights/checkpoints"
ACE_STEP_DIR = "/app/ACE-Step-1.5"


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
        checkpoint_dir=CHECKPOINTS_DIR,
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
    dataset_dir: str = Field(description="Path to dir with audio files + .lyrics.txt / .json metadata")
    output_dir: str = Field(default="/weights/lora_output", description="Where to save trained LoRA adapter")
    exp_name: str = Field(default="lora_train", description="Experiment name for logging")
    max_steps: int = Field(default=2000, ge=100, le=10000000)
    learning_rate: float = Field(default=1e-4, gt=0)
    accumulate_grad_batches: int = Field(default=1, ge=1)
    save_every_n_steps: int = Field(default=2000, ge=100)
    lora_rank: int = Field(default=256, ge=1, le=512)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    target_modules: List[str] = Field(
        default=["speaker_embedder", "linear_q", "linear_k", "linear_v",
                 "to_q", "to_k", "to_v", "to_out.0"],
        description="LoRA target modules"
    )
    repeat_count: int = Field(default=2000, ge=1, description="Dataset repeat count for convert2hf_dataset")
    gradient_clip_val: float = Field(default=0.5, ge=0)


class LoraLoadRequest(BaseModel):
    lora_path: str = Field(description="Path to LoRA adapter directory")
    adapter_name: Optional[str] = Field(default=None, description="Name for the adapter (auto-generated if None)")
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
    _training_status = {"running": True, "message": "converting dataset to HF format..."}

    try:
        os.makedirs(req.output_dir, exist_ok=True)
        dataset_name = os.path.join(req.output_dir, f"{req.exp_name}_dataset")

        # Step 1: Convert raw audio + metadata to HuggingFace dataset
        _training_status["message"] = "converting dataset to HF format..."
        subprocess.run([
            "python", "convert2hf_dataset.py",
            "--data_dir", req.dataset_dir,
            "--repeat_count", str(req.repeat_count),
            "--output_name", dataset_name,
        ], check=True, cwd=ACE_STEP_DIR)

        # Step 2: Generate LoRA config
        lora_config = {
            "r": req.lora_rank,
            "lora_alpha": req.lora_alpha,
            "target_modules": req.target_modules,
            "use_rslora": True,
        }
        lora_config_path = os.path.join(req.output_dir, "lora_config.json")
        with open(lora_config_path, "w") as f:
            json.dump(lora_config, f, indent=2)

        # Step 3: Train LoRA via trainer.py
        _training_status["message"] = f"training LoRA (max_steps={req.max_steps})..."
        subprocess.run([
            "python", "trainer.py",
            "--dataset_path", dataset_name,
            "--exp_name", req.exp_name,
            "--checkpoint_dir", CHECKPOINTS_DIR,
            "--learning_rate", str(req.learning_rate),
            "--max_steps", str(req.max_steps),
            "--every_n_train_steps", str(req.save_every_n_steps),
            "--accumulate_grad_batches", str(req.accumulate_grad_batches),
            "--gradient_clip_val", str(req.gradient_clip_val),
            "--gradient_clip_algorithm", "norm",
            "--lora_config_path", lora_config_path,
            "--logger_dir", os.path.join(req.output_dir, "logs"),
            "--shift", "3.0",
            "--precision", "bf16-mixed",
            "--devices", "1",
            "--epochs", "-1",
        ], check=True, cwd=ACE_STEP_DIR)

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

    msg = dit_handler.add_lora(req.lora_path, adapter_name=req.adapter_name)
    if req.scale != 1.0:
        adapter = req.adapter_name or "default"
        dit_handler.set_lora_scale(adapter, req.scale)
    return {"status": "ok", "message": msg}


@app.post("/lora/unload")
def unload_lora():
    if dit_handler is None:
        raise HTTPException(503, "Models not loaded yet")

    msg = dit_handler.unload_lora()
    return {"status": "ok", "message": msg}


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
