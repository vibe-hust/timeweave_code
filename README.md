# TimeWeave

<a id="authors"></a>

**👥 Authors:** [Yang Chen](https://github.com/cheyan9)<sup>1</sup> · [Chengsong Qu](https://github.com/Qthrive)<sup>1</sup> · [Runting Li](https://scholar.google.com/citations?user=CwT5bq4AAAAJ&hl=en)<sup>1</sup> · [Xuemei Zhou](https://www.cwi.nl/en/people/xuemei-zhou/)<sup>2</sup> · [Teng Long](https://tenglon.github.io/)<sup>3</sup> · [Vincent Tao Hu](https://taohu.me/)<sup>1</sup>

<sup>1</sup> Huazhong University of Science and Technology<br>
<sup>2</sup> Centrum voor Wiskunde en Informatica (CWI)<br>
<sup>3</sup> University of Trento

[🌐 Project website](https://vibe-hust.github.io/timeweave_page/) · <a href="https://github.com/vibe-hust/timeweave_code"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/github-mark-white.svg"><img src="docs/assets/github-mark.svg" width="16" height="16" alt="GitHub"></picture> Code repository</a> · [📄 Paper + supplement](https://vibe-hust.github.io/timeweave_page/TimeWeave-paper.pdf)

**Global Time, Local Blocks: Generating Minute-Long Videos from Action Schedules**

<a id="method"></a>

## 💡 Method

TimeWeave (formerly OnTime) turns action schedules into long videos with a causal generator trained on short sequences.

- **Rolling Timeline Conditioning (RTC):** each block sees nearby actions, with their original global intervals preserved.
- **Global Context Anchoring (GCA):** scene and participant descriptions remain untimed and share attention with the scheduled actions.

![Method overview](https://vibe-hust.github.io/timeweave_page/method.png)

<a id="release-scope"></a>

## 📦 Release scope

This initial release contains the inference runtime, a portable generation entrypoint, an example schedule, and CPU schedule tests. The runtime is extracted from the research implementation; the wrapper exposes local checkpoint paths instead of cluster-specific configuration. Training runners, model weights, and training datasets are **not included**. The public wrapper has been checked with CPU tests and static validation; it has not been run on a GPU in this release-preparation task. Paper results are not a new validation run of this package.

The required ODE-3000 model and matching raw SF-600 adapter currently have no public download in this release. A generic Wan or Self-Forcing checkpoint is not a substitute. See [checkpoint requirements](docs/CHECKPOINTS.md). You can inspect and test schedule construction without those weights.

<a id="quick-start"></a>

## 🚀 Quick start

Python 3.10+ is recommended. Inspect the schedule without installing Torch:

```bash
git clone https://github.com/vibe-hust/timeweave_code.git
cd timeweave_code
python scripts/inference.py --prompt examples/bicycle.json --plan-only
python -m unittest discover -s tests -v
```

For generation, install a CUDA-compatible PyTorch and the requirements, then supply the exact checkpoint pair:

```bash
pip install -r requirements.txt
python scripts/inference.py \
  --prompt examples/bicycle.json \
  --model-root /path/to/wan_models \
  --ode-checkpoint /path/to/final_student_full.pt \
  --sf-checkpoint /path/to/sf600_generator.pt \
  --output outputs/bicycle
```

Output: `outputs/bicycle/artifacts/videos/sf600_base/sample.mp4`. Generation requires an NVIDIA CUDA GPU; this release does not claim a tested minimum VRAM capacity. Dependencies above reflect the source runtime and are not a newly measured environment lock.

<a id="schedule-format"></a>

## 🗓️ Schedule format

A JSON object contains `duration` and `prompt: [starts, ends, texts, roles]`. The four arrays have equal length. Role `0` identifies untimed scene/identity context; positive roles identify participants' timed actions. Action intervals remain global and are never clipped or rescaled by RTC. Every generation block must have at least one action in its ±4-second context window. The default layout uses 5 initial latents, then blocks of up to 6, at 16 FPS with temporal VAE stride 4.

The example is 30 seconds. Durations up to 60 seconds are accepted on the same latent grid; the model generates one initial frame plus 4 frames per subsequent latent, so the delivered file includes the endpoint frame. This changes playback duration by 1/16 second relative to the specified first-to-last-frame span.

<a id="code-map"></a>

## 🗂️ Code map

| Path | Purpose |
|---|---|
| `scripts/inference.py` | Checkpoint loading, block rollout, cache refresh, and streaming decode |
| `scripts/protocol.py` | RTC event selection, original timestamps, block layout |
| `runtime/utils/global_conditioning.py` | Untimed global-context attention |
| `runtime/utils/tie.py` | Token intervals and temporal encoding |
| `runtime/wan_5b/modules/causal_model.py` | Causal video transformer |
| `runtime/stream_decode.py` | Streaming VAE video export |

<a id="acknowledgments"></a>

## 🙏 Acknowledgments

We build on [Wan](https://github.com/Wan-Video/Wan2.2), [LongLive](https://github.com/NVlabs/LongLive), [Self-Forcing](https://github.com/guandeh17/Self-Forcing), [TIE](https://matrixteam-ai.github.io/pages/TIE/), and the rollout infrastructure in [Astrolabe](https://github.com/franklinz233/Astrolabe). Their contributions remain credited in the source. See [third-party notices](THIRD_PARTY_NOTICES.md).

<a id="citation"></a>

## 📚 Citation

If you use this work or code, please cite:

```bibtex
@misc{chen2026timeweave,
  title={Global Time, Local Blocks: Generating Minute-Long Videos from Action Schedules},
  author={Chen, Yang and Qu, Chengsong and Li, Runting and Zhou, Xuemei and Long, Teng and Hu, Vincent Tao},
  year={2026},
  howpublished={Research preprint},
  url={https://vibe-hust.github.io/timeweave_page/}
}
```

A machine-readable citation is available in [CITATION.cff](CITATION.cff).

<a id="license"></a>

## 📄 License

The runtime includes Apache-2.0 components; upstream notices are retained. Model weights and datasets have separate terms and are not included. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
