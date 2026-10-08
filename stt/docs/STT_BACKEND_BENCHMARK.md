# STT backend benchmark

Audio: `tts/sample/harvard.wav`. Each result is the median of 5 post-warm-up direct backend inferences.
Audio decoding/downmixing/resampling, gRPC transport, and service segmentation are excluded from latency; model load and first inference are shown separately.

| Configuration | Status | Device/provider | Load (s) | Warm-up (s) | Median (s) | P95 (s) | RTF | WER |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Transformers FP16 (CUDA) | ok | cuda | 2.407 | 1.287 | 0.792 | 2.520 | 0.043 | 9.3% |
| ONNX INT8 (CPU) | ok | CPUExecutionProvider | 1.737 | 0.799 | 0.561 | 0.733 | 0.031 | 14.0% |
| ONNX FP8 storage (CPU) | ok | CPUExecutionProvider | 4.297 | 0.642 | 0.539 | 0.663 | 0.029 | 11.6% |
| ONNX FP16 (CPU) | ok | CPUExecutionProvider | 2.958 | 0.414 | 0.421 | 0.445 | 0.023 | 11.6% |
| ONNX INT8 (CUDA) | ok | CUDAExecutionProvider | 2.227 | 0.755 | 0.678 | 0.845 | 0.037 | 11.6% |
| ONNX FP8 storage (CUDA) | ok | CUDAExecutionProvider | 4.152 | 0.272 | 0.189 | 0.190 | 0.010 | 11.6% |
| ONNX FP16 (CUDA) | ok | CUDAExecutionProvider | 2.146 | 0.898 | 0.757 | 0.759 | 0.041 | 11.6% |

## Transcripts

### Transformers FP16 (CUDA)

the stale smell of old beard lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham takos al pastora are my favorite a zestful food is the hot cross bun

### ONNX INT8 (CPU)

the stale smell of old beard lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham takos al pastora are my favorite azful food is the hot cross bun

### ONNX FP8 storage (CPU)

stale smell of old beard lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham takos al pastora are my favorite a zestful food is the hot cross bun

### ONNX FP16 (CPU)

stale smell of old beard lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham takos al pastora are my favorite a zestful food is the hot cross bun

### ONNX INT8 (CUDA)

stale smell of old beard lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham takos al pastora are my favorite a zestful food is the hot cross bun

### ONNX FP8 storage (CUDA)

stale smell of old beard lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham takos al pastora are my favorite a zestful food is the hot cross bun

### ONNX FP16 (CUDA)

stale smell of old beard lingers it takes heat to bring out the odor a cold dip restores health and zest a salt pickle taste fine with ham takos al pastora are my favorite a zestful food is the hot cross bun

## Notes

- RTF is median inference time divided by audio duration; lower is faster than real time.
- WER is case/punctuation-insensitive word error rate against the supplied reference transcript; lower is better.
- The repository’s ONNX choices are `int8`, `fp8` (the files are named `*_real_fp8_storage.onnx`), and `fp16` (the files are in `srvaani-16-onnx`).
