"""Original SraVaani TorchScript/Transformers runtime."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .base import SttBackend


class TransformersBackend(SttBackend):
    engine = "torchscript-transformers"

    def __init__(self):
        import torch
        from transformers import AutoModel

        model_dir = Path(os.environ.get("STT_MODEL_DIR") or os.environ.get("SRAVAANI_MODEL", "")).expanduser()
        if not (model_dir / "model-asr.fp16.ts").is_file():
            raise SystemExit(
                "set SRAVAANI_MODEL (or STT_MODEL_DIR) to the folder containing model-asr.fp16.ts"
            )
        requested_device = os.environ.get("STT_DEVICE", "auto").lower()
        if requested_device not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("STT_DEVICE must be one of: auto, cpu, cuda")
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("STT_DEVICE=cuda was requested, but PyTorch cannot access CUDA")
        self.device = "cuda" if torch.cuda.is_available() and requested_device != "cpu" else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        if self.device == "cpu":
            threads = os.environ.get("STT_CPU_THREADS")
            if threads:
                torch.set_num_threads(int(threads))
        self.model = AutoModel.from_pretrained(
            str(model_dir), trust_remote_code=True, torch_dtype=dtype
        ).to(self.device).eval()
        self.model._ensure_loaded()
        self.model_name = model_dir.name or "sravaani"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        self.profile_attributes = {"runtime": "transformers", "model_path": str(model_dir)}

    def transcribe_waveform(self, waveform: np.ndarray) -> str:
        hypotheses = self.model.transcribe([waveform], return_hypotheses=True)
        return str(hypotheses[0].text)
