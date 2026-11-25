from typing import List, Optional
import torch

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
import torch.distributed as dist
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation


class CausalInferencePipeline(torch.nn.Module):
    @staticmethod
    def _unwrap_backbone(model):
        """Return the Wan module underneath an optional PEFT wrapper."""
        if hasattr(model, "get_base_model"):
            try:
                return model.get_base_model()
            except Exception:
                pass
        base_model = getattr(model, "base_model", None)
        if base_model is not None:
            nested = getattr(base_model, "model", None)
            if nested is not None:
                return nested
        return model

    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        # if DEBUG:
        #     print(f"args.model_kwargs: {args.model_kwargs}")
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.denoising_step_list = torch.tensor(
            args.denoising_step_list, dtype=torch.long)
        if args.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        # Get model architecture parameters dynamically
        # Detect from the actual model instead of hardcoding
        model = self._unwrap_backbone(self.generator.model)
        if hasattr(model, 'num_layers'):
            self.num_transformer_blocks = model.num_layers
        elif hasattr(model, 'blocks'):
            self.num_transformer_blocks = len(model.blocks)
        else:
            self.num_transformer_blocks = 30  # fallback for Wan 1.3B

        # Get num_heads and head_dim from model
        if hasattr(model, 'num_heads'):
            self.num_heads = model.num_heads
        elif hasattr(model, 'blocks') and len(model.blocks) > 0:
            # Try to get from first attention block
            first_block = model.blocks[0]
            if hasattr(first_block, 'self_attn') and hasattr(first_block.self_attn, 'num_heads'):
                self.num_heads = first_block.self_attn.num_heads
            else:
                self.num_heads = 12  # fallback for Wan 1.3B
        else:
            self.num_heads = 12  # fallback for Wan 1.3B

        # Head dimension is typically 128 for Wan models
        self.head_dim = 128

        # Use the actual patch grid.  Wan2.2-TI2V-5B is 44x80 / (2x2) = 880
        # tokens per latent frame; the legacy Wan2.1 path remains 1560.
        self.frame_seq_length = getattr(self.generator, "frame_seq_length", 1560)

        self.kv_cache1 = None
        self.crossattn_cache = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        model_kwargs = getattr(args, "model_kwargs", {})
        self.local_attn_size = (getattr(model_kwargs, "local_attn_size", None)
                                if not isinstance(model_kwargs, dict)
                                else model_kwargs.get("local_attn_size", -1))
        if self.local_attn_size is None:
            self.local_attn_size = -1
        self.independent_first_frame = bool(getattr(args, "independent_first_frame", False))
        self.tie_enabled = bool(getattr(args, "tie_enabled", False))
        self.tie_video_fps = float(getattr(args, "tie_video_fps", 16.0))

        # Normalize to list if sequence-like (e.g., OmegaConf ListConfig)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"KV inference with {self.num_frame_per_block} frames per block")

        backbone = self._unwrap_backbone(self.generator.model)
        if (getattr(backbone, "num_frame_per_block", None) != self.num_frame_per_block
                or getattr(backbone, "independent_first_frame", None) != self.independent_first_frame):
            backbone.block_mask = None
            backbone._block_mask_batch_size = 0
            backbone._block_mask_key = None
        backbone.num_frame_per_block = self.num_frame_per_block
        backbone.independent_first_frame = self.independent_first_frame

        # Ensure all modules have the correct max_attention_size based on local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        prompt_metadata=None,
        release_cache_before_decode: bool = False,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        # Support initial_latent for video extension if needed
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames

        clamp_i2v_first_chunk = self.independent_first_frame and initial_latent is not None
        if clamp_i2v_first_chunk and num_input_frames != 1:
            raise ValueError("independent_first_frame with initial_latent requires one latent frame")
        if not self.independent_first_frame or clamp_i2v_first_chunk:
            assert num_frames % self.num_frame_per_block == 0
            all_num_frames = [self.num_frame_per_block] * (num_frames // self.num_frame_per_block)
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            all_num_frames = [1] + [self.num_frame_per_block] * ((num_frames - 1) // self.num_frame_per_block)

        if self.tie_enabled and hasattr(self.text_encoder, "encode_tie"):
            if prompt_metadata is None:
                raise ValueError("TIE inference requires prompt_metadata with interval timestamps")
            conditional_dict = self.text_encoder.encode_tie(
                prompt_metadata, video_fps=self.tie_video_fps)
        else:
            conditional_dict = self.text_encoder(text_prompts=text_prompts)

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

        # Decide the device for output based on low_memory (CPU for low-memory mode; otherwise GPU)
        output_device = torch.device('cpu') if low_memory else noise.device
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=output_device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        model_kwargs = getattr(self.args, "model_kwargs", {})
        local_attn_cfg = (getattr(model_kwargs, "local_attn_size", -1)
                          if not isinstance(model_kwargs, dict)
                          else model_kwargs.get("local_attn_size", -1))
        if local_attn_cfg != -1:
            kv_cache_size = local_attn_cfg * self.frame_seq_length
        else:
            kv_cache_size = num_output_frames * self.frame_seq_length

        self._initialize_kv_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device,
            kv_cache_size_override=kv_cache_size
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size,
            dtype=noise.dtype,
            device=noise.device
        )

        current_start_frame = 0
        self._unwrap_backbone(self.generator.model).local_attn_size = self.local_attn_size
        self._set_all_modules_max_attention_size(self.local_attn_size)

        # Step 2: Cache context feature (Initial Latent)
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents.to(output.device)
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        for current_block_index, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if index < len(self.denoising_step_list) - 1:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    _, denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        current_start=current_start_frame * self.frame_seq_length,
                    )

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred.to(output.device)

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones_like(timestep) * self.args.context_noise
            self.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=conditional_dict,
                timestep=context_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            vae_start.record()

        # Step 4: Decode the output
        # The complete latent video no longer needs attention history.  In
        # training, keeping these large KV buffers during VAE decode can push
        # the next rollout over the GPU memory limit after optimizer state is
        # initialized by the first update.
        if release_cache_before_decode:
            self.release_inference_caches()
        model_kwargs = getattr(self.args, "model_kwargs", {})
        use_infinite_attention = (getattr(model_kwargs, "use_infinite_attention", False)
                                  if not isinstance(model_kwargs, dict)
                                  else model_kwargs.get("use_infinite_attention", False))
        if use_infinite_attention:
            video = self.vae.decode_to_pixel_chunk(output.to(noise.device), use_cache=False)
        else:
            video = self.vae.decode_to_pixel(output.to(noise.device), use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        if profile:
            # End VAE timing and synchronize CUDA
            vae_end.record()
            torch.cuda.synchronize()
            vae_time = vae_start.elapsed_time(vae_end)
            total_time = init_time + diffusion_time + vae_time

            print("Profiling results:")
            print(f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)")
            print(f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)")
            for i, block_time in enumerate(block_times):
                print(f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)")
            print(f"  - VAE decoding time: {vae_time:.2f} ms ({100 * vae_time / total_time:.2f}%)")
            print(f"  - Total time: {total_time:.2f} ms")

        if return_latents:
            return video, output.to(noise.device)
        else:
            return video

    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size_override: int | None = None):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        # Determine cache size
        if kv_cache_size_override is not None:
            kv_cache_size = kv_cache_size_override
        else:
            if self.local_attn_size != -1:
                # Local attention: cache only needs to store the window
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                # Global attention: default cache for 21 frames (backward compatibility)
                kv_cache_size = 32760

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "scene_cut": False
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, self.num_heads, self.head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

    def release_inference_caches(self):
        """Release rollout-only KV caches before NFT's backward pass."""
        self.kv_cache1 = None
        self.crossattn_cache = None

    def _set_all_modules_max_attention_size(self, local_attn_size_value: int):
        """
        Set max_attention_size on all submodules that define it.
        If local_attn_size_value == -1, use the model's global default (32760 for Wan, 28160 for 5B).
        Otherwise, set to local_attn_size_value * frame_seq_length.
        """
        if local_attn_size_value == -1:
            target_size = getattr(self.generator, "seq_len", 32760)
            policy = "global"
        else:
            target_size = int(local_attn_size_value) * self.frame_seq_length
            policy = "local"

        updated_modules = []
        # Update root model if applicable
        if hasattr(self.generator.model, "max_attention_size"):
            try:
                prev = getattr(self.generator.model, "max_attention_size")
            except Exception:
                prev = None
            setattr(self.generator.model, "max_attention_size", target_size)
            updated_modules.append("<root_model>")

        # Update all child modules
        for name, module in self.generator.model.named_modules():
            if hasattr(module, "max_attention_size"):
                try:
                    prev = getattr(module, "max_attention_size")
                except Exception:
                    prev = None
                try:
                    setattr(module, "max_attention_size", target_size)
                    updated_modules.append(name if name else module.__class__.__name__)
                except Exception:
                    pass
