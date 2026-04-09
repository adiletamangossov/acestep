import runpod
import base64
import random
import subprocess
import tempfile
import os
import pathlib


def patch_vector_quantize():
    """Patch vector_quantize_pytorch for meta-tensor compatibility."""
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

    # Clear bytecode cache
    for pyc in site.rglob("*.pyc"):
        pyc.unlink(missing_ok=True)


def load_model():
    patch_vector_quantize()

    from acestep.handler import AceStepHandler
    from acestep.llm_inference import LLMHandler

    dit = AceStepHandler()
    msg, ok = dit.initialize_service(
        project_root="/weights",
        config_path="acestep-v15-xl-turbo",
        device="cuda",
    )
    print(f"DiT init: {msg} (ok={ok})", flush=True)

    llm = LLMHandler()
    msg, ok = llm.initialize(
        checkpoint_dir="/weights/checkpoints",
        lm_model_path="acestep-5Hz-lm-4B",
        backend="pt",
        device="cuda",
    )
    print(f"LLM init: {msg} (ok={ok})", flush=True)

    return dit, llm


# Load once at worker start
dit_handler, llm_handler = load_model()


def handler(job):
    try:
        inp = job["input"]

        prompt = inp.get("prompt", "")
        lyrics = inp.get("lyrics", "")
        duration = float(inp.get("duration", 30.0))
        seed = inp.get("seed") or random.randint(0, 2**32 - 1)
        bpm = inp.get("bpm")
        keyscale = inp.get("keyscale", "")
        vocal_language = inp.get("vocal_language", "en")
        instrumental = inp.get("instrumental", False)

        from acestep.inference import GenerationParams, GenerationConfig, generate_music

        params = GenerationParams(
            task_type="text2music",
            caption=prompt,
            lyrics=lyrics,
            duration=duration,
            bpm=bpm,
            keyscale=keyscale,
            vocal_language=vocal_language,
            instrumental=instrumental,
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
                return {"status": "error", "message": result.error or result.status_message}

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

        # Metadata from LLM chain-of-thought
        lm_meta = result.extra_outputs.get("lm_metadata", {})

        return {
            "status": "ok",
            "audio_base64": base64.b64encode(mp3_bytes).decode(),
            "mime_type": "audio/mp3",
            "seed": seed,
            "actual_bpm": lm_meta.get("bpm"),
            "actual_key": lm_meta.get("keyscale"),
        }

    except Exception as e:
        import traceback
        return {"status": "error", "message": str(e), "trace": traceback.format_exc()}


runpod.serverless.start({"handler": handler})