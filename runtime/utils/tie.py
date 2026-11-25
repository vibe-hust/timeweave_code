"""Fixed Time Interval Encoding for LongLive 2.0.

The physical clock is in seconds, independent of diffusion timesteps and of
the rolling self-attention cache's storage addresses. Defaults reproduce
MatrixTeam-AI/TIE's sinc_form (alpha=1, scaling_factor=4, theta=10000,
enable_mapping=False, sinc_l2_norm=False). This module has no parameters.
"""
from dataclasses import dataclass
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class IntervalPrompt:
    starts: tuple
    ends: tuple
    descriptions: tuple
    timelines: tuple

    @classmethod
    def parse(cls, value):
        if isinstance(value, dict):
            if "prompt" not in value:
                raise ValueError("TIE JSONL rows must contain 'prompt'")
            value = value["prompt"]
        if not isinstance(value, (list, tuple)) or len(value) not in (3, 4):
            raise ValueError("Expected TIE [starts_s, ends_s, descriptions, optional_timeline_ids]")
        if not all(isinstance(v, (list, tuple)) for v in value):
            raise ValueError("Each TIE field must be an array")
        n = len(value[0])
        if not n or any(len(v) != n for v in value):
            raise ValueError("TIE arrays must have the same nonzero length")
        starts, ends = [], []
        for a, b in zip(value[0], value[1]):
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (a, b)):
                raise ValueError("TIE timestamps must be finite numbers in seconds")
            if a < 0 or b < a:
                raise ValueError("TIE intervals require 0 <= start <= end")
            starts.append(float(a))
            ends.append(float(b))
        if any(not isinstance(s, str) or not s.strip() for s in value[2]):
            raise ValueError("TIE descriptions must be nonempty strings")
        timelines = tuple(value[3]) if len(value) == 4 else (0,) * n
        if any(isinstance(t, bool) or not isinstance(t, int) or t < 0 for t in timelines):
            raise ValueError("TIE timeline IDs must be nonnegative integers")
        return cls(tuple(starts), tuple(ends), tuple(value[2]), timelines)

    def as_list(self):
        return [list(self.starts), list(self.ends), list(self.descriptions), list(self.timelines)]

    @property
    def duration_s(self):
        return max(self.ends)


def validate_clock(fps, temporal_stride=4):
    if isinstance(fps, bool) or not math.isfinite(float(fps)) or fps <= 0:
        raise ValueError("video_fps must be a finite positive number")
    if temporal_stride != 4:
        raise ValueError("This baseline supports the Wan 2.2 VAE temporal stride of 4")


def latent_intervals(indices, fps):
    """Native TIE support cells, generalized from 16 fps to the actual video fps.

    Global latent 0: [0, 1/fps]. Global latent j>0:
    [(4*j-3)/fps, (4*j+1)/fps]. These are positional support cells, not a claim
    about the VAE's full receptive field. A cropped chunk never resets j to 0.
    """
    validate_clock(fps)
    if indices.dtype not in (torch.int32, torch.int64) or torch.any(indices < 0):
        raise ValueError("Global latent indices must be nonnegative integers")
    j = indices.to(torch.float64)
    start = torch.where(indices == 0, 0.0, (4 * j - 3) / fps)
    return start, (4 * j + 1) / fps


def decoded_frames(latent_frames):
    if isinstance(latent_frames, bool) or not isinstance(latent_frames, int) or latent_frames < 1:
        raise ValueError("latent_frames must be a positive integer")
    return 4 * (latent_frames - 1) + 1


def physical_rope(start_s, end_s, head_dim=128, *, interval=False):
    """TIE mean-normalized sinc, with float64 phase math before Q/K casting."""
    if head_dim <= 0 or head_dim % 2:
        raise ValueError("TIE requires an even head dimension")
    if start_s.shape != end_s.shape or start_s.ndim != 2:
        raise ValueError("Time tensors must have matching [batch, length] shapes")
    if not torch.isfinite(start_s).all() or not torch.isfinite(end_s).all() or torch.any(end_s < start_s):
        raise ValueError("Invalid physical interval")
    frequency = 10000.0 ** (-torch.arange(0, head_dim, 2, dtype=torch.float64, device=start_s.device) / head_dim)
    center = (start_s.double() + end_s.double()) * 2.0  # gamma * center, gamma=4
    phase = center[..., None] * frequency
    cosine, sine = phase.cos(), phase.sin()
    if interval:
        radius = (end_s.double() - start_s.double()) * 2.0
        sinc = torch.sinc(radius[..., None] * frequency / torch.pi)
        normalizer = sinc.mean(dim=-1, keepdim=True)
        if torch.any(normalizer.abs() < 1e-8):
            raise ValueError("TIE sinc mean is degenerate at this interval duration")
        cosine, sine = cosine * (sinc / normalizer), sine * (sinc / normalizer)
    return cosine.float().unsqueeze(2), sine.float().unsqueeze(2)


def prepare_rope(condition, *, frames, spatial_tokens, current_start=0, head_dim=128):
    """Prepare immutable per-forward metadata, also passed into recomputation.

    current_start is LongLive's *global token index*, never cache_start or a
    rolled local_end_index. latent_offset locates a sampled source window;
    time_shift_s translates the complete timeline without changing its speed.
    """
    if condition is None:
        return None
    if spatial_tokens <= 0 or current_start < 0 or current_start % spatial_tokens:
        raise ValueError("current_start must be a nonnegative whole latent-frame token offset")
    fps = condition["video_fps"]
    base = condition["latent_offset"]
    indices = base[:, None] + current_start // spatial_tokens + torch.arange(frames, device=base.device)
    a, b = latent_intervals(indices, fps)
    shift = condition["time_shift_s"][:, None]
    origin = condition["origin_s"][:, None]
    qcos, qsin = physical_rope(a + shift - origin, b + shift - origin, head_dim)
    key_start, key_end = condition["token_start_s"] - origin, condition["token_end_s"] - origin
    global_mask = condition.get("token_is_global")
    if global_mask is not None:
        if global_mask.dtype != torch.bool or global_mask.shape != key_start.shape:
            raise ValueError("Global roles must be bool tensors matching text-token clocks")
        # Global timestamps are irrelevant, including very long nominal spans.
        key_start = torch.where(global_mask, 0, key_start)
        key_end = torch.where(global_mask, 0, key_end)
    kcos, ksin = physical_rope(key_start, key_end, head_dim, interval=True)
    valid = condition["token_event_ids"][:, :, None, None] >= 0
    kcos = torch.where(valid, kcos, 1.0)
    ksin = torch.where(valid, ksin, 0.0)
    result = {"qcos": qcos, "qsin": qsin, "kcos": kcos, "ksin": ksin,
              "spatial_tokens": spatial_tokens}
    if global_mask is not None:
        result["token_is_global"] = global_mask
    return result


def _rotate_half(x, cosine, sine):
    """Pair the first half with the second half of EACH head (TIE layout)."""
    first, second = x.float().chunk(2, dim=-1)
    return torch.cat((first * cosine - second * sine,
                      first * sine + second * cosine), dim=-1).to(x.dtype)


def apply_rope(q, k, rope, *, teacher_forcing=False):
    if rope is None:
        return q, k
    if q.shape[0] != rope["qcos"].shape[0] or k.shape[:2] != rope["kcos"].shape[:2]:
        raise ValueError("TIE batch/token metadata does not match Q/K; do not flatten events into shots")
    qc = rope["qcos"].repeat_interleave(rope["spatial_tokens"], dim=1)
    qs = rope["qsin"].repeat_interleave(rope["spatial_tokens"], dim=1)
    if teacher_forcing:
        qc, qs = torch.cat((qc, qc), dim=1), torch.cat((qs, qs), dim=1)
    if q.shape[1] != qc.shape[1]:
        raise ValueError("TIE query times do not match the video patch grid")
    return _rotate_half(q, qc, qs), _rotate_half(k, rope["kcos"], rope["ksin"])


def pack_tokens(prompt, tokenizer, text_len=512):
    """Native TIE concatenate-before-T5 packing; fail instead of truncating."""
    prompt = prompt if isinstance(prompt, IntervalPrompt) else IntervalPrompt.parse(prompt)
    pieces, event_ids = [], []
    for i, text in enumerate(prompt.descriptions):
        # Override the upstream wrapper's truncation default so overflow is visible.
        ids, mask = tokenizer(text, return_mask=True, add_special_tokens=True,
                              truncation=False, padding=False)
        ids = ids[0, mask[0].bool()].tolist()
        if not ids or ids[-1] != tokenizer.tokenizer.eos_token_id:
            raise ValueError("TIE packing expects a tokenizer with a trailing EOS")
        if i + 1 < len(prompt.descriptions):
            ids = ids[:-1]
        pieces.extend(ids)
        event_ids.extend([i] * len(ids))
    if len(pieces) > text_len:
        raise ValueError(f"TIE prompt has {len(pieces)} tokens, exceeds {text_len}; select a whole-event window first")
    length = len(pieces)
    return (pieces + [tokenizer.tokenizer.pad_token_id] * (text_len - length),
            [1] * length + [0] * (text_len - length),
            event_ids + [-1] * (text_len - length))


@torch.no_grad()
def encode_batch(encoder, prompts, *, video_fps, latent_offsets=None, time_shifts_s=None, origins_s=None):
    validate_clock(video_fps)
    parsed = [p if isinstance(p, IntervalPrompt) else IntervalPrompt.parse(p) for p in prompts]
    packed = [pack_tokens(p, encoder.tokenizer) for p in parsed]
    device = next(encoder.text_encoder.parameters()).device
    ids, masks, event_ids = [torch.tensor([p[i] for p in packed], device=device, dtype=torch.long) for i in range(3)]
    context = encoder.text_encoder(ids, masks)
    context = context.masked_fill(~masks.bool().unsqueeze(-1), 0)
    n = len(parsed)
    offsets = torch.as_tensor(latent_offsets if latent_offsets is not None else [0] * n, device=device)
    if offsets.shape != (n,) or offsets.dtype not in (torch.int32, torch.int64) or torch.any(offsets < 0):
        raise ValueError("latent_offsets must contain one nonnegative integer per prompt")
    shifts = torch.as_tensor(time_shifts_s if time_shifts_s is not None else [0.0] * n, device=device, dtype=torch.float64)
    origins = torch.as_tensor(origins_s if origins_s is not None else [0.0] * n, device=device, dtype=torch.float64)
    if shifts.shape != (n,) or origins.shape != (n,) or not torch.isfinite(shifts).all() or not torch.isfinite(origins).all():
        raise ValueError("shifts/origins must contain one finite value per prompt")
    starts, ends = torch.zeros_like(event_ids, dtype=torch.float64), torch.zeros_like(event_ids, dtype=torch.float64)
    for i, prompt in enumerate(parsed):
        valid = event_ids[i] >= 0
        lookup = event_ids[i, valid]
        starts[i, valid] = torch.tensor(prompt.starts, device=device, dtype=torch.float64)[lookup] + shifts[i]
        ends[i, valid] = torch.tensor(prompt.ends, device=device, dtype=torch.float64)[lookup] + shifts[i]
    from utils.global_conditioning import token_global_mask
    global_masks = torch.stack([token_global_mask(event_ids[i], prompt) for i, prompt in enumerate(parsed)])
    return {"prompt_embeds": context, "tie": {
        "token_start_s": starts, "token_end_s": ends, "token_event_ids": event_ids,
        "token_is_global": global_masks,
        "latent_offset": offsets.long(), "time_shift_s": shifts, "origin_s": origins,
        "video_fps": float(video_fps),
    }}


def to_device(condition, device, dtype):
    result = {"prompt_embeds": condition["prompt_embeds"].to(device=device, dtype=dtype)}
    if "tie" in condition:
        # Never cast clocks or integer IDs to bf16 alongside model activations.
        result["tie"] = {k: v.to(device=device) if torch.is_tensor(v) else v
                         for k, v in condition["tie"].items()}
    return result


def tail_condition(condition, start_latent):
    """Keep the same event timeline when the score loss selects a rollout tail."""
    if "tie" not in condition:
        return condition
    return {**condition, "tie": {**condition["tie"],
            "latent_offset": condition["tie"]["latent_offset"] + start_latent}}


class TIEDataset(Dataset):
    """Index JSONL byte offsets without retaining all prompt text in memory.

    ``expected_duration_s`` is the annotated prompt duration.
    ``supervision_tail_s`` permits an explicitly configured continuation after
    that annotation without stretching any event timestamp.  The 64->43 run
    uses a 10-second prompt plus a 0.5-second continuation, so both latent grids
    cover the same 10.5-second endpoint span.
    """
    def __init__(self, path, *, video_fps, window_latents,
                 supervision_latents=None, expected_duration_s=None,
                 supervision_tail_s=0.0):
        validate_clock(video_fps)
        self.path = Path(path)
        self.offsets = []
        self.fps = video_fps
        self.window = window_latents
        self.supervision = supervision_latents or window_latents
        if (isinstance(self.supervision, bool) or
                not isinstance(self.supervision, int) or
                not 1 <= self.supervision <= window_latents):
            raise ValueError("supervision_latents must be in [1, window_latents]")
        if (isinstance(supervision_tail_s, bool) or
                not isinstance(supervision_tail_s, (int, float)) or
                not math.isfinite(supervision_tail_s) or
                supervision_tail_s < 0):
            raise ValueError("supervision_tail_s must be finite and nonnegative")
        self.supervision_tail_s = float(supervision_tail_s)
        # Video duration is the first-to-last frame span.  For 64 latents at
        # VAE stride 4 and 24 fps this is exactly (253 - 1) / 24 = 10.5 s.
        required = (decoded_frames(self.supervision) - 1) / video_fps
        if expected_duration_s is not None:
            if (isinstance(expected_duration_s, bool) or
                    not isinstance(expected_duration_s, (int, float)) or
                    not math.isfinite(expected_duration_s) or expected_duration_s <= 0):
                raise ValueError("expected_duration_s must be finite and positive")
            expected_duration_s = float(expected_duration_s)
            if abs(required - (
                    expected_duration_s + self.supervision_tail_s)) > 1e-9:
                raise ValueError(
                    f"Scored latent span {required:.9f}s does not match "
                    f"prompt duration {expected_duration_s:.9f}s plus "
                    f"supervision tail {self.supervision_tail_s:.9f}s")
        self.filtered_short = 0
        self.duration_mismatch = 0
        with self.path.open("rb") as stream:
            line_number = 0
            while True:
                position = stream.tell()
                line = stream.readline()
                if not line:
                    break
                line_number += 1
                try:
                    prompt = IntervalPrompt.parse(json.loads(line))
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid TIE row {line_number} in {self.path.name}: {exc}") from exc
                if (expected_duration_s is not None and
                        abs(prompt.duration_s - expected_duration_s) > 1e-9):
                    self.duration_mismatch += 1
                    continue
                if (prompt.duration_s + self.supervision_tail_s + 1e-9
                        >= required):
                    self.offsets.append(position)
                else:
                    self.filtered_short += 1
        if not self.offsets:
            raise ValueError(
                f"No TIE prompts plus configured continuation cover the "
                f"{required:.6f}s training window")

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open("rb") as stream:
            stream.seek(self.offsets[index])
            return IntervalPrompt.parse(json.loads(stream.readline()))

    def latest_start(self, prompt):
        # The last scored latent center is tied to the final decoded-frame
        # timestamp. Query support intervals may extend half a cell beyond the
        # endpoint, as in native 10-second TIE conditioning.
        return max(0, math.floor(prompt.duration_s * self.fps / 4 + 1e-8)
                   - (self.supervision - 1))
