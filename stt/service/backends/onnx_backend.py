"""SraVaani TDT ONNX runtime, derived from the supplied reference script."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .base import SttBackend


class OnnxBackend(SttBackend):
    engine = "onnxruntime-tdt"
    preserve_input_length = True

    def __init__(self):
        # Load PyTorch's CUDA/cuDNN libraries before ONNX Runtime creates a
        # CUDA provider.  The two packages share the same CUDA-12/cuDNN-9 ABI.
        import torch
        import onnxruntime as ort
        import sentencepiece as spm

        variant = os.environ.get("STT_ONNX_VARIANT", "int8").strip().lower()
        model_specs = {
            "int8": ("sravaani-onnx", "encoder_real_int8.onnx", "decoder_joint_real_int8.onnx"),
            "fp8": ("sravaani-onnx", "encoder_real_fp8_storage.onnx", "decoder_joint_real_fp8_storage.onnx"),
            "fp16": ("srvaani-16-onnx", "encoder.fp16.onnx", "decoder_joint.fp16.onnx"),
        }
        if variant not in model_specs:
            raise RuntimeError("STT_ONNX_VARIANT must be int8, fp8, or fp16")
        default_folder, encoder_name, decoder_name = model_specs[variant]
        default_dir = Path(__file__).resolve().parents[2] / "models" / default_folder
        self.model_dir = Path(os.environ.get("STT_ONNX_MODEL_DIR", default_dir)).expanduser()
        encoder_path = self.model_dir / encoder_name
        decoder_path = self.model_dir / decoder_name
        required = [self.model_dir / "config.json", self.model_dir / "preproc.pt",
                    self.model_dir / "tokenizer.model", encoder_path, decoder_path]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise SystemExit("ONNX SraVaani model is incomplete: " + ", ".join(missing))

        provider_setting = os.environ.get("STT_ONNX_PROVIDER", "auto").strip().lower()
        available = ort.get_available_providers()
        if provider_setting == "auto":
            providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                         if "CUDAExecutionProvider" in available else ["CPUExecutionProvider"])
        elif provider_setting == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("STT_ONNX_PROVIDER=cuda requested but CUDAExecutionProvider is unavailable")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif provider_setting == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            raise RuntimeError("STT_ONNX_PROVIDER must be auto, cpu, or cuda")

        with (self.model_dir / "config.json").open() as f:
            cfg = json.load(f)
        self.blank_id = int(cfg["blank_id"])
        self.num_durations = int(cfg["num_durations"])
        self.durations = [int(value) for value in cfg["durations"]]
        self.pred_rnn_layers = int(cfg["pred_rnn_layers"])
        self.pred_hidden = int(cfg["pred_hidden"])
        self.max_symbols = int(cfg["max_symbols"])
        # The FP16 export's config predates the sample_rate field.  SraVaani's
        # supplied frontend is 16 kHz, which is also the gRPC service contract.
        if int(cfg.get("sample_rate", 16000)) != 16000:
            raise RuntimeError("the gRPC service requires a 16 kHz ONNX SraVaani model")
        preproc = torch.load(self.model_dir / "preproc.pt", map_location="cpu")
        self.params = preproc["params"]
        self.window = preproc["window"].float()
        self.fb = preproc["fb"].float()
        self.torch = torch
        self.tokenizer = spm.SentencePieceProcessor(model_file=str(self.model_dir / "tokenizer.model"))
        options = ort.SessionOptions()
        threads = os.environ.get("STT_CPU_THREADS")
        if threads and providers == ["CPUExecutionProvider"]:
            options.intra_op_num_threads = int(threads)
        self.encoder = ort.InferenceSession(str(encoder_path), sess_options=options, providers=providers)
        self.decoder_joint = ort.InferenceSession(str(decoder_path), sess_options=options, providers=providers)
        encoder_inputs = {item.name: item for item in self.encoder.get_inputs()}
        decoder_inputs = {item.name: item for item in self.decoder_joint.get_inputs()}
        # The INT8/FP8 exports take float32 features as ``feats``.  The newer
        # FP16 export uses equivalent features but names them ``audio_signal``
        # and stores both encoder and decoder state in float16.
        if {"feats", "feature_lengths"} <= encoder_inputs.keys():
            self.encoder_input_names = ("feats", "feature_lengths")
            self.encoder_dtype = np.float32
        elif {"audio_signal", "length"} <= encoder_inputs.keys():
            self.encoder_input_names = ("audio_signal", "length")
            self.encoder_dtype = np.float16
        else:
            raise RuntimeError(
                "unsupported ONNX SraVaani encoder inputs: "
                + ", ".join(encoder_inputs)
            )
        if {"enc_frame", "targets", "target_length", "h_in", "c_in"} <= decoder_inputs.keys():
            self.decoder_input_names = ("enc_frame", "h_in", "c_in")
            self.decoder_dtype = np.float32
            self.target_dtype = np.int64
        elif {"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"} <= decoder_inputs.keys():
            self.decoder_input_names = ("encoder_outputs", "input_states_1", "input_states_2")
            self.decoder_dtype = np.float16
            self.target_dtype = np.int32
        else:
            raise RuntimeError(
                "unsupported ONNX SraVaani decoder inputs: "
                + ", ".join(decoder_inputs)
            )
        actual = self.encoder.get_providers()
        if provider_setting == "cuda" and (not actual or actual[0] != "CUDAExecutionProvider"):
            raise RuntimeError(
                "STT_ONNX_PROVIDER=cuda was requested, but the ONNX sessions fell back to CPU; "
                "install a CUDA/cuDNN-compatible onnxruntime-gpu wheel"
            )
        self.device = "cuda" if actual and actual[0] == "CUDAExecutionProvider" else "cpu"
        self.compute_type = variant
        self.model_name = f"sravaani-onnx-{variant}"
        self.profile_attributes = {
            "runtime": "onnx", "variant": variant, "provider": actual[0] if actual else "unknown",
            "encoder": encoder_path.name, "decoder_joint": decoder_path.name,
        }

    def _extract_features(self, waveform: np.ndarray):
        torch = self.torch
        wav = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32)).unsqueeze(0)
        wav_len = torch.tensor([wav.shape[1]])
        p = self.params
        n_fft, hop, win = p["n_fft"], p["hop_length"], p["win_length"]
        seq_len = torch.div(wav_len, hop, rounding_mode="floor")
        tmask = torch.arange(wav.shape[1]).unsqueeze(0) < wav_len.unsqueeze(1)
        x = torch.cat((wav[:, :1], wav[:, 1:] - p["preemph"] * wav[:, :-1]), dim=1).masked_fill(~tmask, 0.0)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win, window=self.window,
                          center=True, pad_mode="constant", return_complex=True)
        x = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1)).pow(p["mag_power"])
        x = torch.matmul(self.fb, x)
        x = torch.log(x + p["log_zero_guard_value"])
        _, _, max_time = x.shape
        steps = torch.arange(max_time).unsqueeze(0)
        valid = steps < seq_len.unsqueeze(1)
        denom = valid.sum(dim=1)
        mean = torch.where(valid.unsqueeze(1), x, torch.zeros_like(x)).sum(dim=2) / denom.unsqueeze(1)
        var = torch.sum(torch.where(valid.unsqueeze(1), x - mean.unsqueeze(2), torch.zeros_like(x)) ** 2,
                        dim=2) / (denom.unsqueeze(1) - 1.0)
        std = torch.sqrt(var).masked_fill(torch.sqrt(var).isnan(), 0.0) + p["CONSTANT"]
        x = (x - mean.unsqueeze(2)) / std.unsqueeze(2)
        padding = torch.arange(max_time).repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        return x.masked_fill(padding.unsqueeze(1), p["pad_value"]), seq_len

    def transcribe_waveform(self, waveform: np.ndarray) -> str:
        feats, feature_lengths = self._extract_features(waveform)
        encoder_features, encoder_lengths = self.encoder_input_names
        enc_out, enc_len = self.encoder.run(None, {
            encoder_features: feats.numpy().astype(self.encoder_dtype),
            encoder_lengths: feature_lengths.numpy().astype(np.int64),
        })
        h = c = np.zeros((self.pred_rnn_layers, 1, self.pred_hidden), dtype=self.decoder_dtype)
        last, tokens, frame = self.blank_id, [], 0
        while frame < int(enc_len[0]):
            enc_frame = enc_out[0:1, :, frame:frame + 1].astype(self.decoder_dtype, copy=False)
            for _ in range(self.max_symbols):
                encoder_input, h_input, c_input = self.decoder_input_names
                logits, _, h, c = self.decoder_joint.run(None, {
                    encoder_input: enc_frame, "targets": np.array([[last]], dtype=self.target_dtype),
                    "target_length": np.ones((1,), dtype=np.int32), h_input: h, c_input: c,
                })
                logits = logits[0, 0, 0]
                token = int(np.argmax(logits[:-self.num_durations]))
                skip = self.durations[int(np.argmax(logits[-self.num_durations:]))]
                if token != self.blank_id:
                    tokens.append(token)
                    last = token
                if skip > 0:
                    frame += skip
                    break
            else:
                frame += 1
        return str(self.tokenizer.decode(tokens))
