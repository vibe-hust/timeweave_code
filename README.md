# ⏱️ TimeWeave (formerly OnTime)

## Global Time, Local Blocks: Generating Minute-Long Videos from Action Schedules

[Yang Chen](https://github.com/cheyan9) · [Chengsong Qu](https://github.com/Qthrive) · [Runting Li](https://scholar.google.com/citations?user=CwT5bq4AAAAJ&hl=en) · [Xuemei Zhou](https://www.cwi.nl/en/people/xuemei-zhou/) · [Teng Long](https://tenglon.github.io/) · [Vincent Tao Hu](https://taohu.me/)

**HUST · CWI · University of Trento**

[📄 Paper + supplement](https://vibe-hust.github.io/timeweave_page/TimeWeave-paper.pdf) · [🎬 Project and videos](https://vibe-hust.github.io/timeweave_page/)

## 💡 Method

TimeWeave turns action schedules into long videos with a causal generator trained on short sequences.

- **Rolling Timeline Conditioning (RTC):** each block sees nearby actions, with their original global intervals preserved.
- **Global Context Anchoring (GCA):** scene and participant descriptions remain untimed and share attention with the scheduled actions.

![Method overview](https://vibe-hust.github.io/timeweave_page/method.png)

## 📦 Release scope

This initial release contains the inference runtime, a portable generation entrypoint, an example schedule, and CPU schedule tests. The runtime is extracted from the research implementation; the wrapper exposes local checkpoint paths instead of cluster-specific configuration. Training runners, model weights, and training datasets are **not included**. The public wrapper has been checked with CPU tests and static validation; it has not been run on a GPU in this release-preparation task. Paper results are not a new validation run of this package.

The required ODE-3000 model and matching raw SF-600 adapter currently have no public download in this release. A generic Wan or Self-Forcing checkpoint is not a substitute. See [checkpoint requirements](docs/CHECKPOINTS.md). You can inspect and test schedule construction without those weights.

## 🚀 Quick start

Python 3.10+ is recommended. Inspect the schedule without installing Torch:

```sh
python scripts/inference.py --prompt examples/bicycle.json --plan-only
python -m unittest discover -s tests -v
```

For generation, install a CUDA-compatible PyTorch and the requirements, then supply the exact checkpoint pair:

```sh
pip install -r requirements.txt
python scripts/inference.py \
  --prompt examples/bicycle.json \
  --model-root /path/to/wan_models \
  --ode-checkpoint /path/to/final_student_full.pt \
  --sf-checkpoint /path/to/sf600_generator.pt \
  --output outputs/bicycle
```

Output: `outputs/bicycle/artifacts/videos/sf600_base/sample.mp4`. Generation requires an NVIDIA CUDA GPU; this release does not claim a tested minimum VRAM capacity. Dependencies above reflect the source runtime and are not a newly measured environment lock.

## 🗓️ Schedule format

A JSON object contains `duration` and `prompt: [starts, ends, texts, roles]`. The four arrays have equal length. Role `0` identifies untimed scene/identity context; positive roles identify participants' timed actions. Action intervals remain global and are never clipped or rescaled by RTC. Every generation block must have at least one action in its ±4-second context window. The default layout uses 5 initial latents, then blocks of up to 6, at 16 FPS with temporal VAE stride 4.

The example is 30 seconds. Durations up to 60 seconds are accepted on the same latent grid; the model generates one initial frame plus 4 frames per subsequent latent, so the delivered file includes the endpoint frame. This changes playback duration by 1/16 second relative to the specified first-to-last-frame span.

## 🗂️ Code map

| Path | Purpose |
|---|---|
| `scripts/inference.py` | Checkpoint loading, block rollout, cache refresh, and streaming decode |
| `scripts/protocol.py` | RTC event selection, original timestamps, block layout |
| `runtime/utils/global_conditioning.py` | Untimed global-context attention |
| `runtime/utils/tie.py` | Token intervals and temporal encoding |
| `runtime/wan_5b/modules/causal_model.py` | Causal video transformer |
| `runtime/stream_decode.py` | Streaming VAE video export |

## 🙏 Acknowledgments

We build on [Wan](https://github.com/Wan-Video/Wan2.2), [LongLive](https://github.com/NVlabs/LongLive), [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [TIE](https://matrixteam-ai.github.io/pages/TIE/), and the rollout infrastructure in [Astrolabe](https://github.com/franklinz233/Astrolabe). Their contributions remain credited in the source. See [third-party notices](THIRD_PARTY_NOTICES.md).

## 📚 Citation

```bibtex
@misc{chen2026timeweave,
  title={Global Time, Local Blocks: Generating Minute-Long Videos from Action Schedules},
  author={Chen, Yang and Qu, Chengsong and Li, Runting and Zhou, Xuemei and Long, Teng and Hu, Vincent Tao},
  year={2026},
  howpublished={Research preprint},
  url={https://vibe-hust.github.io/timeweave_page/}
}
```

## 📜 License

The runtime includes Apache-2.0 components; upstream notices are retained. Model weights and datasets have separate terms and are not included. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
