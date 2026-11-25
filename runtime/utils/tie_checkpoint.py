"""Strict loading for the reviewed TIE-SF generator LoRA checkpoint."""
from __future__ import annotations

import re

import torch


TIE_SF_SCHEMA = "adact.ode2000_direct_tie_lora.first1_block8_41.v1"
TIE_SF_BASE_SHA256 = "bd0ae5bb22cb88c55bf574036e4fbc29b1d241767d06535581c8833b69bad360"
TIE_SF_DATA_SHA256 = "6791da25d09733ab40f5868aaca209b79612666d99172af4464947c135d6901a"
TIE_SF_6X7_SCHEMA = "adact.ode2000_direct_tie_lora.block6x7_42.v1"
TIE_SF_6X7_BASE_SHA256 = "48c41b3ae23dff3bcb36d0b8e23795577e0abe2f53e15104fe60847545be7396"
TIE_SF_6X7_DATA_SHA256 = "bb2207d0eb4d26ee962090febcc28f9a76675fc05ec92b8262af3f39a5697886"
_LORA_KEY = re.compile(
    r"^model\.base_model\.model\.blocks\.(\d+)\.(.+)\.lora_([AB])\.default\.weight$"
)


def _validate_state(state, *, expected_rank: int, exact_contract: bool):
    if not isinstance(state, dict) or not state:
        raise ValueError("generator LoRA payload is not a non-empty state dict")
    if not all(isinstance(key, str) and torch.is_tensor(value)
               for key, value in state.items()):
        raise ValueError("generator LoRA payload contains non-tensor values")

    sides = {"A": set(), "B": set()}
    blocks = set()
    for key, value in state.items():
        match = _LORA_KEY.fullmatch(key)
        if match is None:
            raise ValueError(f"unexpected generator LoRA key: {key}")
        block, module, side = int(match.group(1)), match.group(2), match.group(3)
        blocks.add(block)
        sides[side].add((block, module))
        rank_axis = 0 if side == "A" else 1
        if value.ndim != 2 or value.shape[rank_axis] != expected_rank:
            raise ValueError(
                f"invalid LoRA-{side} rank for {key}: shape={tuple(value.shape)}"
            )
    if sides["A"] != sides["B"]:
        raise ValueError("generator LoRA A/B module pairs are incomplete")
    if exact_contract:
        if len(state) != 600 or blocks != set(range(30)):
            raise ValueError(
                f"reviewed TIE-SF adapter must have 600 keys over blocks 0..29; "
                f"got {len(state)} keys and blocks {sorted(blocks)}"
            )
    return state


def load_generator_lora_checkpoint(
    path,
    *,
    expected_rank: int = 128,
    expected_step: int = 600,
    expected_final: bool = True,
    expected_source_run_id: str | None = None,
    expected_ode_step: int | None = None,
):
    """Return only the generator adapter from a supported checkpoint.

    The current TIE-SF full training checkpoint stores the two adapters under
    ``generator`` and ``critic``.  Legacy LongLive adapter-only checkpoints use
    ``generator_lora``.  A full training checkpoint is accepted only when its
    schema and all reviewed temporal/base/data invariants match.
    """
    checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if expected_source_run_id is not None:
        import os
        schema = 'adact.gu_euler48_sf600.generator_export.v1'
        if not isinstance(checkpoint, dict) or checkpoint.get('schema') != schema:
            raise ValueError('The selected base requires a verified generator-only DCP export')
        metadata = checkpoint['metadata']
        valid = (
            checkpoint.get('source_run_id') == expected_source_run_id
            and checkpoint.get('step') == expected_step == 600
            and checkpoint.get('final') is expected_final is True
            and checkpoint.get('world_size') == 64
            and metadata['initialization']['source_step'] == expected_ode_step == 3000
            and metadata['base_sha256']['generator'] == os.environ['TIW_NFT_ODE_SHA256']
            and metadata['data']['data_sha256'] == os.environ['TIW_NFT_TRAIN_SHA256']
            and metadata['cross_attention']['student'] == 'global_untimed_joint_softmax'
            and metadata['block_layout'] == [5,6,6,6,6,6,6]
            and metadata['sink_latents'] == 5
            and metadata['rollout_latents'] == metadata['supervision_latents'] == 41
            and metadata['student_fps'] == 16
            and metadata['precision']['tie_metadata'] == 'float64'
            and metadata['adapter']['rank'] == metadata['adapter']['alpha'] == expected_rank == 128
            and metadata['adapter']['dropout'] == 0.
            and metadata['projection']['loss_normalization'] == 'per_sample_full_clip'
            and metadata['optimization_contract']['rollout_model_timesteps'] == [1000.,937.5,833.3333129882812,625.]
            and metadata['ema']['enabled'] is False
        )
        if not valid:
            raise ValueError('Selected GU Euler48 ODE3k + SF600 base contract mismatch')
        return _validate_state(checkpoint['generator'],expected_rank=expected_rank,exact_contract=True)
    exact_contract = False
    global_schema = "adact.global_untimed_direct_tie_lora.first5_block6x6_full41.v1"
    if isinstance(checkpoint, dict) and checkpoint.get("schema") == global_schema:
        import os
        metadata = checkpoint["metadata"]
        assert checkpoint["step"] == expected_step == 600 and checkpoint["final"] is True
        if "world_size" in checkpoint:
            assert checkpoint["world_size"] == 64
        assert metadata["cross_attention"]["student"] == "global_untimed_joint_softmax"
        assert metadata["base_sha256"]["generator"] == os.environ["TIW_NFT_ODE_SHA256"]
        assert metadata["data"]["data_sha256"] == os.environ["TIW_NFT_TRAIN_SHA256"]
        assert metadata["initialization"]["source_step"] == 2000
        assert metadata["rollout_latents"] == metadata["supervision_latents"] == 41
        assert metadata["adapter"]["rank"] == metadata["adapter"]["alpha"] == expected_rank == 128
        assert metadata["adapter"]["dropout"] == 0.0
        assert metadata["block_layout"] == [5,6,6,6,6,6,6] and metadata["sink_latents"] == 5
        assert metadata["student_fps"] == 16 and metadata["precision"]["tie_metadata"] == "float64"
        return _validate_state(checkpoint["generator"], expected_rank=expected_rank, exact_contract=True)
    if isinstance(checkpoint, dict) and checkpoint.get("schema") in (TIE_SF_SCHEMA, TIE_SF_6X7_SCHEMA):
        exact_contract = True
        is_6x7 = checkpoint["schema"] == TIE_SF_6X7_SCHEMA
        metadata = checkpoint.get("metadata")
        adapter = metadata.get("adapter") if isinstance(metadata, dict) else None
        time_encoding = (metadata.get("three_role_time_encoding")
                         if isinstance(metadata, dict) else None)
        data = metadata.get("data") if isinstance(metadata, dict) else None
        base_hashes = metadata.get("base_sha256") if isinstance(metadata, dict) else None
        valid = (
            checkpoint.get("step") == expected_step
            and checkpoint.get("final") is expected_final
            and checkpoint.get("world_size") == 64
            and isinstance(adapter, dict)
            and adapter.get("type") == "lora"
            and adapter.get("rank") == expected_rank
            and adapter.get("alpha") == expected_rank
            and adapter.get("dropout") == 0.0
            and adapter.get("apply_to_critic") is True
            and metadata.get("encoding") == "tie_sinc_form_alpha1_gamma4_theta10000_mean_v1"
            and (metadata.get("student_fps"), metadata.get("teacher_fps"),
                 metadata.get("critic_fps"), metadata.get("vae_temporal_stride"))
                == (16, 16, 16, 4)
            and (metadata.get("rollout_latents"), metadata.get("supervision_latents"))
                == ((42, 42) if is_6x7 else (41, 41))
            and (not is_6x7 or (metadata.get("block_latents"),
                                  metadata.get("sink_latents")) == (6, 6))
            and isinstance(time_encoding, dict)
            and all(time_encoding.get(key) is True for key in
                    ("student_enabled", "teacher_enabled", "critic_enabled"))
            and time_encoding.get("video_fps") == 16
            and isinstance(base_hashes, dict)
            and base_hashes.get("generator") == (TIE_SF_6X7_BASE_SHA256 if is_6x7 else TIE_SF_BASE_SHA256)
            and isinstance(data, dict)
            and data.get("data_sha256") == (TIE_SF_6X7_DATA_SHA256 if is_6x7 else TIE_SF_DATA_SHA256)
        )
        if not valid:
            raise ValueError(f"reviewed TIE-SF checkpoint contract mismatch: {path}")
        state = checkpoint.get("generator")
    elif isinstance(checkpoint, dict) and "generator_lora" in checkpoint:
        state = checkpoint["generator_lora"]
    else:
        state = checkpoint
    return _validate_state(
        state, expected_rank=expected_rank, exact_contract=exact_contract)
