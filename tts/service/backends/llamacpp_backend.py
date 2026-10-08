"""llama.cpp TTS backend for MioTTS-family GGUF models (e.g. SPRINGLab/Indic-Mio).

MioTTS is a plain causal LM: prompt it with text, it autoregressively emits
audio tokens, and MioCodec decodes those tokens to a waveform. So llama.cpp runs
the backbone (the reason to want the .gguf) and MioCodec does the decode.

Pipeline (mirrors Aratako/MioTTS-Inference, the reference server):
  1. Build input ids with the model's chat template (HF tokenizer -- shares the
     gguf's vocab, so no retokenization mismatch).
  2. Generate with llama.cpp; keep raw token ids.
  3. Content tokens are the generated ids in [SPEECH_OFFSET, SPEECH_OFFSET+CODEBOOK)
     minus SPEECH_OFFSET (equivalent to the server's <|s_N|> regex: N is the code).
  4. Speaker identity comes from the reference voice: encode TTS_PROMPT_WAV once
     for its global_embedding, then
     MioCodecModel.decode(content_token_indices=<1-D codes>, global_embedding=...).

ponytail: offset/codebook/codec/tokenizer default to Indic-Mio and are env-
overridable -- no per-model subclass until a second Mio variant needs one.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import numpy as np

from .base import TtsBackend


class LlamaCppBackend(TtsBackend):
    engine = "miotts-llamacpp"
    compute_type = "gguf"

    def __init__(self, model_path: Path):
        # Load PyTorch before llama.cpp: both load CUDA/NCCL libraries, and this avoids a clean-restart symbol conflict on this host.
        import torch

        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
        except ImportError:
            raise SystemExit("GGUF TTS needs llama-cpp-python: pip install llama-cpp-python")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise SystemExit("GGUF TTS needs transformers for the chat template")
        try:
            from miocodec import MioCodecModel, load_audio
        except ImportError:
            raise SystemExit(
                "GGUF TTS needs the MioCodec decoder: "
                "pip install git+https://github.com/Aratako/MioCodec"
            )

        if not model_path.is_file():
            raise SystemExit(f"GGUF model path must be a file: {model_path}")
        prompt_wav = Path(os.environ.get("TTS_PROMPT_WAV", ""))
        if not prompt_wav.is_file():
            raise SystemExit("set TTS_PROMPT_WAV to the reference voice the codec should clone")

        self.speech_offset = int(os.environ.get("TTS_SPEECH_OFFSET", "151669"))
        self.codebook_size = int(os.environ.get("TTS_CODEBOOK_SIZE", "12800"))
        self.temperature = float(os.environ.get("TTS_LLAMACPP_TEMPERATURE", "0.9"))
        self.top_p = float(os.environ.get("TTS_TOP_P", "0.9"))
        # 25 Hz codec -> ~25 audio tokens/sec; 1500 caps a turn near 60 s.
        self.max_audio_tokens = int(os.environ.get("TTS_MAX_AUDIO_TOKENS", "1500"))
        # Opt-in, EXPERIMENTAL (see synthesize_stream's docstring): decode this
        # many audio tokens as soon as they're generated, so first audio can
        # reach the caller before the backbone LM finishes a longer reply. 0
        # (default) keeps the single whole-utterance decode this backend has
        # always used.
        self.stream_first_chunk_tokens = int(os.environ.get("TTS_STREAM_DECODE_FIRST_CHUNK_TOKENS", "0"))
        self.supports_stream = self.stream_first_chunk_tokens > 0
        n_gpu_layers = int(os.environ.get("TTS_N_GPU_LAYERS", "0"))
        n_ctx = int(os.environ.get("TTS_N_CTX", "8192"))
        n_threads = int(os.environ.get("TTS_NUM_THREADS", "0")) or max(1, (os.cpu_count() or 4) // 2)
        # TTS_N_GPU_LAYERS>0 is a request, not a fact. A CPU-only llama-cpp-python
        # wheel ignores it in silence, and the backbone decode -- which is nearly
        # all of TTS compute, and is memory-bandwidth bound, so it loses roughly an
        # order of magnitude on CPU -- keeps running there while every profile this
        # service emits claims cuda. That wrong device lands in ClickHouse and is
        # read back as capacity data, so refuse to start instead. Mirrors the STT
        # ONNX provider check (stt/service/backends/onnx_backend.py).
        if n_gpu_layers > 0:
            if not llama_supports_gpu_offload():
                raise RuntimeError(
                    "TTS_N_GPU_LAYERS>0 was requested, but this llama-cpp-python build has no "
                    'GPU support. Reinstall it with CMAKE_ARGS="-DGGML_CUDA=on" pip install '
                    "llama-cpp-python --force-reinstall --no-cache-dir, or set TTS_N_GPU_LAYERS=0."
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "TTS_N_GPU_LAYERS>0 was requested, but torch cannot reach CUDA, so the "
                    "MioCodec decoder would stay on CPU. Install a CUDA torch build, or set "
                    "TTS_N_GPU_LAYERS=0."
                )
        self.device = "cuda" if n_gpu_layers > 0 else "cpu"

        # Tokenizer builds the prompt only; generation runs on the gguf. Defaults
        # to the .gguf's own directory, then the Indic-Mio repo.
        tokenizer_id = (os.environ.get("TTS_GGUF_TOKENIZER")
                        or (str(model_path.parent) if (model_path.parent / "tokenizer_config.json").is_file()
                            else "SPRINGLab/Indic-Mio"))
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)

        self.llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_gpu_layers=n_gpu_layers,
                         n_threads=n_threads, verbose=False)

        # 24 kHz Mio codec: integrated iSTFT decoder, no separate vocoder.
        codec_id = os.environ.get("TTS_MIOCODEC", "Aratako/MioCodec-25Hz-24kHz")
        # Both checks above already passed, so the two stages cannot disagree.
        self.codec_device = self.device
        self.codec = MioCodecModel.from_pretrained(codec_id).eval().to(self.codec_device)
        self.sampling_rate = int(getattr(self.codec.config, "sample_rate", 24000))

        # Clone the reference voice once: encode -> global_embedding drives decode.
        waveform = load_audio(str(prompt_wav), sample_rate=self.sampling_rate)
        if hasattr(waveform, "to"):
            waveform = waveform.to(self.codec_device)
        with torch.inference_mode():
            ref = self.codec.encode(waveform, return_content=False, return_global=True)
        self.global_embedding = ref.global_embedding

        self.model_name = model_path.stem
        self.profile_attributes = {
            "n_gpu_layers": str(n_gpu_layers), "temperature": str(self.temperature),
            "top_p": str(self.top_p), "codec": codec_id, "sample_rate_hz": str(self.sampling_rate),
            # The two stages cost wildly different amounts -- backbone decode is
            # almost all of it -- so record where each actually ran, not just the
            # request. Both are always reported, including on the CPU path.
            "llm_device": self.device, "codec_device": self.codec_device,
        }

    def _generate_codes(self, text: str) -> list[int]:
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True)
        eos = self.llm.token_eos()
        codes: list[int] = []
        lo, hi = self.speech_offset, self.speech_offset + self.codebook_size
        for token in self.llm.generate(input_ids, temp=self.temperature, top_p=self.top_p):
            if token == eos or len(codes) >= self.max_audio_tokens:
                break
            if lo <= token < hi:
                codes.append(token - self.speech_offset)
        return codes

    def _decode(self, codes: list[int]) -> np.ndarray:
        import torch

        # Reference server passes content tokens as a 1-D [T] long tensor.
        tokens = torch.tensor(codes, dtype=torch.long, device=self.codec_device)
        with torch.inference_mode():
            audio = self.codec.decode(content_token_indices=tokens, global_embedding=self.global_embedding)
        return np.asarray(audio.squeeze().detach().cpu().numpy(), dtype=np.float32)

    def synthesize(self, text: str) -> np.ndarray:
        codes = self._generate_codes(text)
        if not codes:
            raise RuntimeError("model generated no audio tokens for the given text")
        return self._decode(codes)

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Decode the first `stream_first_chunk_tokens` audio tokens as soon
        as they're generated, then whatever remains in one call once
        generation finishes -- at most two codec calls, so first audio can
        reach the caller well before the backbone LM is done with a longer
        reply, instead of waiting for the entire utterance to generate.

        EXPERIMENTAL, opt-in (TTS_STREAM_DECODE_FIRST_CHUNK_TOKENS>0):
        MioCodec's decoder is not documented as causal or chunk-streamable --
        the reference MioTTS-Inference server, and this backend's own
        synthesize(), always decode the whole token sequence in one call.
        Decoding the first chunk in isolation, without the tokens that would
        follow it in a single-shot decode, may make that one chunk sound
        subtly different at its trailing edge -- this has not been verified
        by ear. Do a listening comparison against the default (0, disabled)
        before relying on this in production. The remainder chunk, if any,
        is unaffected: it is decoded as its own complete, self-contained
        call, identical in kind to a normal short synthesize().
        """
        if not self.stream_first_chunk_tokens:
            yield self.synthesize(text)
            return
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], add_generation_prompt=True)
        eos = self.llm.token_eos()
        codes: list[int] = []
        lo, hi = self.speech_offset, self.speech_offset + self.codebook_size
        first_chunk_emitted = False
        for token in self.llm.generate(input_ids, temp=self.temperature, top_p=self.top_p):
            if token == eos or len(codes) >= self.max_audio_tokens:
                break
            if lo <= token < hi:
                codes.append(token - self.speech_offset)
            if not first_chunk_emitted and len(codes) >= self.stream_first_chunk_tokens:
                first_chunk_emitted = True
                yield self._decode(codes)
                codes = []
        if not codes and not first_chunk_emitted:
            raise RuntimeError("model generated no audio tokens for the given text")
        if codes:
            yield self._decode(codes)
