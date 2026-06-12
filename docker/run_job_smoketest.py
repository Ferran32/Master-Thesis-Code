#!/usr/bin/env python3
"""Minimal load + single-generation smoke test for a model.

Loads the model in 4-bit exactly like the real pipeline and generates a few
tokens. Used to validate a new image/model (e.g. qwen3.6-27b's hybrid
linear-attention layers) before spending GPU hours on a full evaluation job.

Exits 0 on success, non-zero on any failure.
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, "/app")

import torch
import transformers

from src.config import get_model_id, model_short_name
from src.pipeline.inference import generate, load_model

MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.6-27b")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16"))
PROMPT = os.environ.get(
    "SMOKE_PROMPT",
    "Reply with the single word OK if you can read this.",
)


def main() -> int:
    print(f"[smoketest] transformers={transformers.__version__} torch={torch.__version__}")
    print(f"[smoketest] cuda_available={torch.cuda.is_available()} "
          f"device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"[smoketest] gpu={torch.cuda.get_device_name(0)}")

    full_model_id = get_model_id(MODEL_NAME)
    short = model_short_name(full_model_id)
    print(f"[smoketest] model={MODEL_NAME} -> {full_model_id} (short={short}) "
          f"max_seq={MAX_SEQ_LENGTH}")

    try:
        model, tokenizer = load_model(full_model_id, max_seq_len=MAX_SEQ_LENGTH)
        arch = type(model).__name__
        model_type = getattr(getattr(model, "config", None), "model_type", "?")
        print(f"[smoketest] loaded OK: arch={arch} model_type={model_type}")

        output = generate(
            model, tokenizer, PROMPT, full_model_id,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        print(f"[smoketest] generation OK. output={output!r}")
    except Exception:
        print("[smoketest] FAILED:")
        traceback.print_exc()
        return 1

    print("[smoketest] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
