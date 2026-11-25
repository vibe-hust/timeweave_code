"""Thin model adapters for the fixed-TIE baseline; upstream checkpoints stay intact."""
import inspect
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.scheduler import FlowMatchScheduler
from utils.tie import prepare_rope
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder
from wan_5b.modules.causal_model import CausalWanModel
from wan_5b.modules.model import WanModel, rope_params


def sdpa_attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0.,
                   softmax_scale=None, q_scale=None, causal=False,
                   window_size=(-1, -1), **unused):
    """Dense SDPA backend with FlashAttention's bottom-right causal convention.

    Padding/window masks are enforced rather than silently ignored. Native TIE
    intentionally supplies no text length mask, matching its released teacher.
    """
    b, nq, heads, _ = q.shape
    nk = k.shape[1]
    mask = None
    if q_lens is not None or k_lens is not None or causal or window_size != (-1, -1):
        qi = torch.arange(nq, device=q.device)[None, :, None]
        ki = torch.arange(nk, device=q.device)[None, None, :]
        qlen = torch.full((b, 1, 1), nq, device=q.device) if q_lens is None else q_lens.to(q.device)[:, None, None]
        klen = torch.full((b, 1, 1), nk, device=q.device) if k_lens is None else k_lens.to(q.device)[:, None, None]
        mask = (qi < qlen) & (ki < klen)
        diagonal = qi + klen - qlen
        if causal:
            mask = mask & (ki <= diagonal)
        left, right = window_size
        if left >= 0:
            mask = mask & (ki >= diagonal - left)
        if right >= 0:
            mask = mask & (ki <= diagonal + right)
        mask = mask[:, None]
    if q_scale is not None:
        q = q * q_scale
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
        v.transpose(1, 2), attn_mask=mask, dropout_p=dropout_p,
        scale=softmax_scale, enable_gqa=(q.shape[2] != k.shape[2])).transpose(1, 2).contiguous()


def configure_attention(backend):
    """Set an explicit backend only for this dedicated baseline entry point."""
    from wan_5b.modules import causal_model, model, attention
    if backend == "sdpa":
        model.flash_attention = sdpa_attention
        causal_model.flash_attention = sdpa_attention
        causal_model.attention = sdpa_attention
    elif backend == "flash":
        if not (attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE or attention.FLASH_ATTN_4_AVAILABLE):
            raise RuntimeError("attention_backend=flash but no supported FlashAttention is installed")
        model.flash_attention = attention.flash_attention
        causal_model.flash_attention = attention.flash_attention
        causal_model.attention = attention.flash_attention
    else:
        raise ValueError("attention_backend must be sdpa or flash")


def load_state(path):
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        for key in ("generator", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    # Released LongLive wrapper states have exactly one model. prefix. Native
    # TIE states do not. Reject mixed prefixes rather than forgiving missing keys.
    if state and all(k.startswith("model.") for k in state):
        state = {k[len("model."):]: v for k, v in state.items()}
    if not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError("Expected a full base model state, not a LoRA/resume payload")
    if any("lora_" in k for k in state):
        raise ValueError("Use a merged base OR AR+adapter; a LoRA checkpoint is not a base model")
    return state


def build_model(config_path, checkpoint_path, *, causal, block_frames, local_frames,
                sink_frames, dtype=torch.bfloat16):
    cls = CausalWanModel if causal else WanModel
    raw = json.loads(Path(config_path).read_text())
    accepted = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in raw.items() if k in accepted}
    if causal:
        kwargs.update(num_frame_per_block=block_frames, local_attn_size=local_frames, sink_size=sink_frames)
    with torch.device("meta"):
        model = cls(**kwargs)
    state = {k: v.to(dtype=dtype) for k, v in load_state(checkpoint_path).items()}
    model.load_state_dict(state, strict=True, assign=True)
    # freqs is intentionally not a registered buffer in Wan; reconstruct after
    # meta initialization instead of attempting to copy a meta tensor to CUDA.
    d = model.dim // model.num_heads
    model.freqs = torch.cat([rope_params(1024, d - 4 * (d // 6)),
                            rope_params(1024, 2 * (d // 6)),
                            rope_params(1024, 2 * (d // 6))], dim=1)
    model.eval().requires_grad_(False)
    return model


def attach_lora(model, rank, alpha):
    """Same target policy as LongLive: every Linear inside transformer blocks."""
    from peft import LoraConfig, get_peft_model
    if rank <= 0 or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    names = [name for name, module in model.named_modules()
             if name.startswith("blocks.") and isinstance(module, torch.nn.Linear)]
    if not names:
        raise ValueError("No transformer Linear modules found for LoRA")
    adapted = get_peft_model(model, LoraConfig(r=rank, lora_alpha=alpha,
        lora_dropout=0., target_modules=names, bias="none"))
    unexpected = [n for n, p in adapted.named_parameters() if p.requires_grad and "lora_" not in n]
    if unexpected:
        raise RuntimeError(f"Non-LoRA trainable parameters: {unexpected}")
    return adapted


class TIETextEncoder(WanTextEncoder):
    def __init__(self, model_dir, device="cpu", dtype=torch.bfloat16):
        torch.nn.Module.__init__(self)
        from wan_5b.modules.t5 import umt5_xxl
        from wan_5b.modules.tokenizers import HuggingfaceTokenizer
        with torch.device("meta"):
            self.text_encoder = umt5_xxl(encoder_only=True, return_tokenizer=False,
                                         dtype=dtype, device=torch.device("meta"))
        state = torch.load(Path(model_dir) / "models_t5_umt5-xxl-enc-bf16.pth",
                           map_location="cpu", weights_only=True, mmap=True)
        self.text_encoder.load_state_dict(state, strict=True, assign=True)
        self.text_encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.tokenizer = HuggingfaceTokenizer(name=str(Path(model_dir) / "google/umt5-xxl"),
            seq_len=512, clean="whitespace", local_files_only=True)

    @property
    def device(self):
        return next(self.text_encoder.parameters()).device

    @torch.no_grad()
    def encode_tie(self, prompts, *, video_fps=16.0, latent_offsets=None,
                   time_shifts_s=None, origins_s=None):
        """Encode native TIE interval prompts.

        ``prompts`` is a list of ``IntervalPrompt`` objects or the JSON
        representation accepted by :func:`utils.tie.encode_batch`.  Keeping
        this path separate from ``forward`` prevents a plain text reward
        prompt from silently dropping the physical time metadata.
        """
        from utils.tie import encode_batch
        return encode_batch(
            self,
            prompts,
            video_fps=video_fps,
            latent_offsets=latent_offsets,
            time_shifts_s=time_shifts_s,
            origins_s=origins_s,
        )


class TIEDiffusionWrapper(WanDiffusionWrapper):
    def __init__(self, model, *, causal, timestep_shift=5.0, native_teacher=False, cross_attention_mode="native_tie"):
        torch.nn.Module.__init__(self)
        self.model = model
        self.model_name = "Wan2.2-TI2V-5B"
        self.model_dir = None
        self.is_ti2v_5b = True
        self.temporal_compression_ratio = 4
        self.spatial_compression_ratio = 16
        self.latent_channels = 48
        self.frame_seq_length = 880
        self.seq_len = 36080
        self.uniform_timestep = not causal
        self.native_teacher = native_teacher
        self.cross_attention_mode = cross_attention_mode
        self._compiled_model_call = None
        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0., extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.post_init()

    @staticmethod
    def _unwrap_backbone(model):
        if hasattr(model, "get_base_model"):
            try:
                return model.get_base_model()
            except Exception:
                pass
        base_model = getattr(model, "base_model", None)
        nested = getattr(base_model, "model", None) if base_model is not None else None
        return nested if nested is not None else model

    def forward(self, noisy_image_or_video, conditional_dict, timestep,
                kv_cache=None, crossattn_cache=None, current_start=0, cache_start=None,
                clean_x=None, aug_t=None,
                immutable_cache_until_backward=False,
                force_differentiable_kernels=False):
        b, frames, _, height, width = noisy_image_or_video.shape
        backbone = self._unwrap_backbone(self.model)
        pt, ph, pw = backbone.patch_size
        if pt != 1 or height % ph or width % pw:
            raise ValueError("TIE requires whole Wan spatial patches and temporal patch size 1")
        if timestep.shape != (b, frames):
            raise ValueError("Diffusion timestep must be [batch, latent_frames]")
        spatial = (height // ph) * (width // pw)
        from utils.global_conditioning import role_tie_condition
        tie = conditional_dict.get("tie")
        if tie is not None:
            for key in ("token_start_s", "token_end_s", "time_shift_s", "origin_s"):
                if tie[key].dtype != torch.float64:
                    raise TypeError(f"TIE clock precision lost: {key}")
        rope = prepare_rope(role_tie_condition(tie, self.cross_attention_mode), frames=frames,
            spatial_tokens=spatial, current_start=current_start or 0,
            head_dim=backbone.dim // backbone.num_heads)
        if self.uniform_timestep:
            if clean_x is not None or kv_cache is not None:
                raise ValueError("TIE real/fake scores are bidirectional, without causal cache or clean-prefix forcing")
            # Wan's bidirectional model takes per-token diffusion times. Native
            # TIE first casts per-frame t to model dtype, then embeds/broadcasts.
            input_t = timestep.to(noisy_image_or_video.dtype) if self.native_teacher else timestep
            input_t = input_t.repeat_interleave(spatial, dim=1)
        else:
            input_t = timestep
        kwargs = dict(t=input_t, context=conditional_dict["prompt_embeds"],
                      seq_len=frames * spatial, tie_rope=rope)
        snapshot = kv_cache is not None and torch.is_grad_enabled()
        if immutable_cache_until_backward and not snapshot:
            raise ValueError(
                "immutable_cache_until_backward requires a gradient-bearing cached forward")
        if kv_cache is not None:
            # Checkpoint recomputation must see exactly the prefix used by this
            # forward. Blockwise replay calls backward before context refresh,
            # so its live cache is already immutable for the graph lifetime and
            # does not need the otherwise very large 32-latent clone.
            clone_cache = snapshot and not immutable_cache_until_backward
            active_cache = [{k: v.detach().clone() if torch.is_tensor(v) else v
                             for k, v in layer.items()} for layer in kv_cache] if clone_cache else kv_cache
            kwargs.update(kv_cache=active_cache, crossattn_cache=crossattn_cache,
                          current_start=current_start, cache_start=cache_start,
                          defer_cache_updates=snapshot)
        elif clean_x is not None:
            kwargs.update(clean_x=clean_x.permute(0, 2, 1, 3, 4), aug_t=aug_t)
        # Do not let the upstream optional eager cache-update optimization
        # apply writes to the immutable recomputation snapshot.
        # Compile the gradient-bearing causal tail as well as no-grad forwards.
        # A snapshot call returns (output, deferred_updates); only output is
        # part of the immutable recomputation graph.
        call = (self._compiled_model_call
                if snapshot and self._compiled_model_call is not None
                else self.model) if snapshot else self._call_model
        from wan_5b.modules.causal_model import differentiable_kernel_parity
        with differentiable_kernel_parity(force_differentiable_kernels):
            result = call(
                noisy_image_or_video.permute(0, 2, 1, 3, 4), **kwargs)
        if snapshot:
            result = result[0]  # Deliberately discard writes to the immutable snapshot.
        flow = result.permute(0, 2, 1, 3, 4)
        x0 = self._convert_flow_pred_to_x0(flow.flatten(0, 1),
             noisy_image_or_video.flatten(0, 1), timestep.flatten()).unflatten(0, (b, frames))
        return flow, x0
