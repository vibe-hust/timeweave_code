"""Untimed global text and timed events with one joint attention softmax."""
from __future__ import annotations

import torch


def token_global_mask(event_ids, prompt):
    """Use explicit source timeline IDs; duration never determines the role.

    Timeline zero is the source's global scene/identity/task channel. Padding
    retains the native attention visibility and uses neutral, untimed Q/K.
    """
    from utils.tie import IntervalPrompt
    parsed = prompt if isinstance(prompt, IntervalPrompt) else IntervalPrompt.parse(prompt)
    if event_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("Token event IDs must be integers")
    if torch.any(event_ids >= len(parsed.descriptions)) or torch.any(event_ids < -1):
        raise ValueError("Cached token event IDs do not match the source prompt")
    roles = torch.tensor([t == 0 for t in parsed.timelines], device=event_ids.device)
    return (event_ids < 0) | roles[event_ids.clamp_min(0)]


def role_tie_condition(tie, mode):
    """Keep Teacher/Critic numerically on native TIE despite shared metadata."""
    if mode not in ("native_tie", "global_untimed_joint_softmax"):
        raise ValueError("Unknown text cross-attention mode")
    if tie is None:
        return None
    if mode == "native_tie":
        return {k: v for k, v in tie.items() if k != "token_is_global"}
    mask = tie.get("token_is_global")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool or mask.shape != tie["token_event_ids"].shape:
        raise ValueError("Untimed-global Student requires an explicit bool token-role mask")
    return tie


def joint_attention(q, k, v, rope, attention, *, teacher_forcing=False, context_lens=None):
    """One ordinary attention call exactly represents both sets of logits.

    Q'=[Q,RQ], K_global'=[K,0], K_event'=[0,TIE(K)]. The scale
    remains 1/sqrt(original_head_dim). Zero-padding V makes the fused SDPA
    Q/K/V head sizes match; strip that zero tail after the shared softmax.
    This introduces no parameters and does not materialize dense logits.
    """
    from utils.tie import apply_rope
    rotated_q, rotated_k = apply_rope(q, k, rope, teacher_forcing=teacher_forcing)
    mask = rope.get("token_is_global")
    if mask is None:
        return attention(rotated_q, rotated_k, v, k_lens=context_lens)
    if mask.dtype != torch.bool or mask.shape != k.shape[:2]:
        raise ValueError("Global token roles do not match projected text keys")
    mask = mask[:, :, None, None]
    original_dim = q.shape[-1]
    query = torch.cat((q, rotated_q), dim=-1)
    key = torch.cat((torch.where(mask, k, 0), torch.where(mask, 0, rotated_k)), dim=-1)
    value = torch.cat((v, torch.zeros_like(v)), dim=-1)
    output = attention(query, key, value, k_lens=context_lens,
                       softmax_scale=original_dim ** -0.5)
    return output[..., :original_dim].contiguous()
