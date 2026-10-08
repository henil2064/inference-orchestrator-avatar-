"""Production in-memory inference helpers for the bundled DhVaani model.

This code deliberately lives outside the downloaded ``models/`` directory: that
directory is ignored by git and can be replaced when a model is upgraded.
"""
import re
import torch


def prepare_prompt(model, prompt_wav: str, prompt_text: str) -> dict:
    """Decode and condition a fixed reference voice once for this model."""
    from zipvoice.utils.infer import add_punctuation, load_prompt_wav, remove_silence, rms_norm

    rt = model._runtime()
    wav = load_prompt_wav(prompt_wav, sampling_rate=model.sampling_rate)
    wav = remove_silence(wav, model.sampling_rate, only_edge=False, trail_sil=200)
    wav, rms = rms_norm(wav, target_rms=0.1)
    prompt_tokens = rt["tokenizer"].texts_to_tokens([add_punctuation(prompt_text)])[0]
    if not prompt_tokens:
        raise ValueError("reference transcript produced no tokens")
    return {
        "features": rt["feature"].extract(wav, sampling_rate=model.sampling_rate).to(model.device).unsqueeze(0) * 0.1,
        "tokens": rt["tokenizer"].tokens_to_token_ids([prompt_tokens]),
        "duration": wav.shape[-1] / model.sampling_rate,
        "rms": rms,
        "token_count": len(prompt_tokens),
    }


@torch.inference_mode()
def synthesize(model, condition: dict, text: str, *, num_step: int, guidance_scale: float,
               speed: float, seed: int = 666):
    """Render speech with cached voice conditioning and no temporary WAV file."""
    from zipvoice.utils.infer import add_punctuation, batchify_tokens, chunk_tokens_punctuation, cross_fade_concat, remove_silence

    torch.manual_seed(seed)
    rt = model._runtime()
    tokenizer, vocoder = rt["tokenizer"], rt["vocoder"]
    tokens_str = tokenizer.texts_to_tokens([add_punctuation(text)])[0]
    token_duration = condition["duration"] / (condition["token_count"] * speed)
    max_tokens = max(1, int((25 - condition["duration"]) / token_duration))
    chunked_tokens = tokenizer.tokens_to_token_ids(
        chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
    )
    batches, chunked_index = batchify_tokens(
        chunked_tokens, max_duration=100, prompt_duration=condition["duration"],
        token_duration=token_duration,
    )
    # fp16 matmuls on CUDA (params stay fp32, so the sampler's fp32 noise and the
    # fp32 vocoder keep working); plain fp32 on CPU where fp16 is slower/unsupported.
    dev_type = model.device.type
    autocast = torch.autocast(device_type=dev_type, dtype=torch.float16, enabled=dev_type == "cuda")
    chunked_features = []
    wavs = []
    with autocast:
        for batch_tokens in batches:
            batch_size = len(batch_tokens)
            pred_features, pred_lens, _, _ = model.model.sample(
                tokens=batch_tokens,
                prompt_tokens=condition["tokens"] * batch_size,
                prompt_features=condition["features"].repeat(batch_size, 1, 1),
                prompt_features_lens=torch.full((batch_size,), condition["features"].size(1), device=model.device),
                speed=speed, t_shift=0.5, duration="predict", num_step=num_step,
                guidance_scale=guidance_scale,
            )
            chunked_features.append((pred_features.permute(0, 2, 1) / 0.1, pred_lens))

        for features, lengths in chunked_features:
            for i in range(features.size(0)):
                # .float() pins downstream cross-fade/silence/numpy to fp32 regardless of autocast.
                wav = vocoder.decode(features[i][None, :, : lengths[i]]).squeeze(1).clamp(-1, 1).float()
                if condition["rms"] < 0.1:
                    wav = wav * condition["rms"] / 0.1
                wavs.append(wav)
    ordered = [wav for _, wav in sorted(zip(chunked_index, wavs), key=lambda item: item[0])]
    final_wav = cross_fade_concat(ordered, fade_duration=0.1, sample_rate=model.sampling_rate)
    final_wav = remove_silence(final_wav, model.sampling_rate, only_edge=True, trail_sil=0)
    return final_wav.squeeze().detach().cpu().numpy().astype("float32", copy=False)


def split_sentences(text: str, max_chars: int = 360) -> list:
    """Break a reply into the smallest safe synthesis units so audio can stream out
    as each is rendered: sentences (incl. Indic danda), then hard-wrapped at max_chars.

    DhVaani's sampler is not token-streamable, so one sentence is the smallest safe
    unit; the caller synthesizes and emits each piece before starting the next.
    """
    sentences = [p.strip() for p in re.split(r"(?<=[.!?।！？])\s+", text) if p.strip()]
    pieces = []
    for sentence in sentences or [text]:
        for start in range(0, len(sentence), max_chars):
            pieces.append(sentence[start:start + max_chars])
    return pieces
