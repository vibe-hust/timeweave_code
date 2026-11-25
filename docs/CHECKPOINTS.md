# Checkpoint requirements

The current public inference entrypoint requires the exact research pair below. No pretrained weight download is published here yet.

| Asset | SHA-256 |
|---|---|
| GlobalUntimed Euler48 ODE-3000 full export | `67f49bd8d051ea0dd07f2f6e9542fc1bd313151e2b9de6228adc651124d47ac3` |
| Raw SF-600 generator LoRA | `f05041a5bb8e3d5cfd700d97961bf37546f1575abcd405ce075d4ab3b40b1047` |

The adapter is rank/alpha 128. It is merged into FP32 ODE weights before conversion to BF16. The generator uses GCA with joint softmax, untimed global context, global action intervals, first-block size 5, subsequent size 6, local attention 41, sink 5, 16 FPS, and shift-5 warped four-step sampling. Preserve the checkpoint schema and provenance; renaming a different checkpoint does not make it compatible.

`--model-root` contains `Wan2.2-TI2V-5B/config.json`, its UMT5 tokenizer/encoder, and Wan2.2 VAE files. Obtain base assets under their upstream terms. Only load checkpoint files from a trusted source.
