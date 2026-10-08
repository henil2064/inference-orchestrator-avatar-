"""Bare-PyTorch TTS backend: the bundled DhVaani / ZipVoice flow-matching model.

This is the path the service has always run. It loads one HF remote-code model
and conditions one reference voice at startup, then renders in-memory.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from service.dhvaani_runtime import prepare_prompt, synthesize
from .base import TtsBackend


class PyTorchBackend(TtsBackend):
    engine = "zipvoice-flow-matching"

    def __init__(self, model_dir: Path):
        import torch  # deferred so the llama.cpp backend needn't import torch
        from transformers import AutoModel

        self.model_dir = model_dir
        prompt_wav = Path(os.environ.get("TTS_PROMPT_WAV", ""))
        prompt_text = os.environ.get("TTS_PROMPT_TEXT", "")
        if not (model_dir / "config.json").is_file():
            raise SystemExit(f"TTS_MODEL_DIR is not a DhVaani model: {model_dir}")
        if not prompt_wav.is_file():
            raise SystemExit("set TTS_PROMPT_WAV to a readable reference WAV")
        if not prompt_text:
            raise SystemExit("set TTS_PROMPT_TEXT to the reference WAV transcript")

        device = os.environ.get("TTS_DEVICE", "auto").lower()
        if device not in {"auto", "cpu", "cuda"}:
            raise RuntimeError("TTS_DEVICE must be auto, cpu, or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("TTS_DEVICE=cuda requested but CUDA unavailable")
        self.device = "cuda" if device != "cpu" and torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "float32"
        if self.device == "cpu":
            # CPU inference is intra-op-thread bound; leave torch's default unless
            # pinned so it doesn't oversubscribe when co-located with other services.
            threads = int(os.environ.get("TTS_NUM_THREADS", "0"))
            if threads > 0:
                torch.set_num_threads(threads)

        self.num_step = int(os.environ.get("TTS_NUM_STEP", "16"))
        self.guidance_scale = float(os.environ.get("TTS_GUIDANCE_SCALE", "1.0"))
        self.speed = float(os.environ.get("TTS_SPEED", "1.0"))

        # The remote-code model imports its bundled ZipVoice backend from model_dir.
        self.model = AutoModel.from_pretrained(str(model_dir), trust_remote_code=True).to(self.device).eval()
        self.prompt_condition = prepare_prompt(self.model, str(prompt_wav), prompt_text)
        self.sampling_rate = int(self.model.sampling_rate)
        self.model_name = "DhVaani-0.5"
        self.profile_attributes = {
            "num_step": str(self.num_step),
            "guidance_scale": str(self.guidance_scale),
            "speed": str(self.speed),
            "sample_rate_hz": str(self.sampling_rate),
        }

    def synthesize(self, text: str) -> np.ndarray:
        return synthesize(
            self.model, self.prompt_condition, text,
            num_step=self.num_step, guidance_scale=self.guidance_scale, speed=self.speed,
        )
