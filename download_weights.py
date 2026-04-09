#!/usr/bin/env python3
"""Download ACE-Step 1.5 model weights during Docker build."""

import os

CHECKPOINTS_DIR = "/weights/checkpoints"

def main():
    token = os.environ.get("HF_TOKEN")

    from huggingface_hub import snapshot_download

    # Main model: DiT (turbo) + VAE + text encoder + default LM
    print("Downloading ACE-Step/Ace-Step1.5 ...", flush=True)
    snapshot_download(
        "ACE-Step/Ace-Step1.5",
        local_dir=CHECKPOINTS_DIR,
        token=token,
    )

    # Larger 4B language model
    print("Downloading ACE-Step/acestep-5Hz-lm-4B ...", flush=True)
    snapshot_download(
        "ACE-Step/acestep-5Hz-lm-4B",
        local_dir=f"{CHECKPOINTS_DIR}/acestep-5Hz-lm-4B",
        token=token,
    )

    # XL turbo DiT (4B)
    print("Downloading ACE-Step/acestep-v15-xl-turbo ...", flush=True)
    snapshot_download(
        "ACE-Step/acestep-v15-xl-turbo",
        local_dir=f"{CHECKPOINTS_DIR}/acestep-v15-xl-turbo",
        token=token,
    )

    print("All weights downloaded.", flush=True)

if __name__ == "__main__":
    main()
