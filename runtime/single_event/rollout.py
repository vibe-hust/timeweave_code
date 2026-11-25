"""Native four-step Wan sampling with an immutable, absolute-time prefix."""
from __future__ import annotations

import torch
from single_event.contract import chunk_layout


class EventRollout:
    """Wrap the established pipeline; use its model, scheduler, VAE and KV layout."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.layout = chunk_layout()

    def reset_cache(self, example):
        p = self.pipeline
        p.release_inference_caches()
        capacity = (p.local_attn_size if p.local_attn_size > 0 else 41) * p.frame_seq_length
        p._initialize_kv_cache(example.shape[0], example.dtype, example.device,
                               kv_cache_size_override=capacity)
        p._initialize_crossattn_cache(example.shape[0], example.dtype, example.device)

    def predict(self, chunk, condition, timestep, start, *, grad=False):
        p = self.pipeline
        times = torch.full(chunk.shape[:2], float(timestep), dtype=torch.float32,
                           device=chunk.device)
        return p.generator(noisy_image_or_video=chunk, conditional_dict=condition,
                           timestep=times, kv_cache=p.kv_cache1,
                           crossattn_cache=p.crossattn_cache,
                           current_start=start * p.frame_seq_length,
                           immutable_cache_until_backward=grad,
                           force_differentiable_kernels=True)[1]

    @torch.no_grad()
    def prefill(self, history, condition):
        self.reset_cache(history)
        stop = history.shape[1]
        if stop not in {0, *(c.latent_stop for c in self.layout)}:
            raise ValueError("History must end at a native chunk boundary")
        for c in self.layout:
            if c.latent_stop > stop:
                break
            self.predict(history[:, c.latent_start:c.latent_stop].detach(),
                         condition, 0, c.latent_start)

    @torch.no_grad()
    def generate(self, noise, condition, *, prefix=None):
        """Noise covers prefix+window; prefix values are never denoised again."""
        p = self.pipeline
        stop = noise.shape[1]
        if stop not in {c.latent_stop for c in self.layout}:
            raise ValueError("Generation must end at a native chunk boundary")
        if prefix is None:
            prefix = noise[:, :0]
        if prefix.shape[1] > stop:
            raise ValueError("Prefix exceeds requested output")
        self.prefill(prefix, condition)
        output = torch.empty_like(noise)
        output[:, :prefix.shape[1]] = prefix.detach()
        for c in self.layout:
            if c.latent_stop <= prefix.shape[1] or c.latent_stop > stop:
                continue
            noisy = noise[:, c.latent_start:c.latent_stop]
            for i, t in enumerate(p.denoising_step_list):
                clean = self.predict(noisy, condition, t, c.latent_start)
                if i + 1 < len(p.denoising_step_list):
                    next_t = p.denoising_step_list[i + 1]
                    noisy = p.scheduler.add_noise(
                        clean.flatten(0, 1), torch.randn_like(clean.flatten(0, 1)),
                        torch.full((clean.shape[0] * clean.shape[1],), float(next_t),
                                   device=noise.device, dtype=torch.float32)
                    ).unflatten(0, clean.shape[:2])
            output[:, c.latent_start:c.latent_stop] = clean
            self.predict(clean, condition, 0, c.latent_start)
        return output.detach()

    @torch.no_grad()
    def decode(self, latents):
        self.pipeline.release_inference_caches()
        video = self.pipeline.vae.decode_to_pixel(latents, use_cache=False)
        return (video * .5 + .5).clamp(0, 1)


def install_full_rollout(pipeline, engine_type=EventRollout):
    """Use the same 5+6x6 generator for full-video held-out evaluation."""
    engine = engine_type(pipeline)

    def inference(noise, text_prompts, *, prompt_metadata, return_latents=False,
                  release_cache_before_decode=True, **kwargs):
        if kwargs:
            raise ValueError(f"Unsupported single-event inference options: {sorted(kwargs)}")
        pipeline.text_encoder.to(noise.device)
        cond = pipeline.text_encoder.encode_tie(
            [{"prompt": m["prompt"]} for m in prompt_metadata], video_fps=pipeline.tie_video_fps)
        latents = engine.generate(noise, cond)
        video = engine.decode(latents)
        return (video, latents) if return_latents else video

    pipeline.inference = inference
    return engine
