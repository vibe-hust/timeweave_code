import types
import os
from typing import List, Optional
import torch
from torch import nn

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.model import WanModel, RegisterTokens, GanAttentionBlock
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl
from wan.modules.causal_model import CausalWanModel
from wan.modules.causal_model_infer import InferCausalWanModel


def _is_ti2v_5b(model_name: str) -> bool:
    return "TI2V-5B" in str(model_name) or "5B" in str(model_name)


def _model_path(model_name: str, model_dir: Optional[str] = None) -> str:
    root = model_dir or os.environ.get("WAN_MODELS_ROOT", "wan_models")
    return os.path.join(root, model_name)


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_dir: Optional[str] = None) -> None:
        super().__init__()

        if _is_ti2v_5b(model_name):
            from wan_5b.modules.tokenizers import HuggingfaceTokenizer as Wan5BTokenizer
            from wan_5b.modules.t5 import umt5_xxl as umt5_xxl_5b

            model_root = _model_path(model_name, model_dir)
            self.text_encoder = umt5_xxl_5b(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            ).eval().requires_grad_(False)
            self.text_encoder.load_state_dict(
                torch.load(
                    os.path.join(model_root, "models_t5_umt5-xxl-enc-bf16.pth"),
                    map_location="cpu",
                    weights_only=False,
                )
            )
            self.tokenizer = Wan5BTokenizer(
                name=os.path.join(model_root, "google/umt5-xxl"),
                seq_len=512,
                clean="whitespace",
            )
            self.model_name = model_name
            return

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load(os.path.join(_model_path(model_name, model_dir), "models_t5_umt5-xxl-enc-bf16.pth"),
                       map_location='cpu', weights_only=False)
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(_model_path(model_name, model_dir), "google/umt5-xxl/"),
            seq_len=512,
            clean='whitespace')
        self.model_name = model_name

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {
            "prompt_embeds": context
        }


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_name: str = "Wan2.1-T2V-1.3B", model_dir: Optional[str] = None):
        super().__init__()
        if _is_ti2v_5b(model_name):
            from wan_5b.modules.vae2_2 import _video_vae as video_vae_5b

            self.mean = torch.tensor([
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.157,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.123,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.052,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667
            ], dtype=torch.float32)
            self.std = torch.tensor([
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.499,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.06,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744
            ], dtype=torch.float32)
            model_root = _model_path(model_name, model_dir)
            self.model = video_vae_5b(
                pretrained_path=os.path.join(model_root, "Wan2.2_VAE.pth"),
            ).eval().requires_grad_(False)
            self.model_name = model_name
            self.temporal_compression_ratio = 4
            self.spatial_compression_ratio = 16
            self.latent_channels = 48
            return

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = _video_vae(
            pretrained_path=os.path.join(_model_path(model_name, model_dir), "Wan2.1_VAE.pth"),
            z_dim=16,
        ).eval().requires_grad_(False)
        self.model_name = model_name
        self.temporal_compression_ratio = 4
        self.spatial_compression_ratio = 8
        self.latent_channels = 16

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
            self,
            model_name="Wan2.1-T2V-1.3B",
            timestep_shift=8.0,
            is_causal=False,
            local_attn_size=-1,
            sink_size=0,
            infer_only=False,
            checkpoint_path=None,  # Support direct safetensors loading
            model_dir=None,
    ):
        super().__init__()
        self.model_name = model_name
        self.model_dir = model_dir
        self.is_ti2v_5b = _is_ti2v_5b(model_name)

        # Load from safetensors checkpoint if provided
        if checkpoint_path is not None:
            from safetensors.torch import load_file
            print(f"Loading model from safetensors: {checkpoint_path}")
            state_dict = load_file(checkpoint_path, device="cpu")

            # Detect model size from state dict
            if "model.blocks.0.self_attn.k.weight" in state_dict:
                k_weight_dim = state_dict["model.blocks.0.self_attn.k.weight"].shape[0]
                if k_weight_dim == 1536:
                    detected_model = "Wan2.1-T2V-1.3B"
                    model_config = {
                        'dim': 1536,
                        'ffn_dim': 8192,
                        'num_heads': 16,
                        'num_layers': 32,
                    }
                elif k_weight_dim == 5120:
                    detected_model = "Wan2.1-T2V-14B"
                    model_config = {
                        'dim': 5120,
                        'ffn_dim': 13824,
                        'num_heads': 40,
                        'num_layers': 40,
                    }
                else:
                    raise ValueError(f"Unknown model size with k.weight dim: {k_weight_dim}")
                print(f"Detected model: {detected_model}")
                model_name = detected_model

            # Initialize model architecture directly from config
            if is_causal:
                model_cls = InferCausalWanModel if infer_only else CausalWanModel
                self.model = model_cls(
                    model_type='t2v',
                    patch_size=(1, 2, 2),
                    text_len=512,
                    in_dim=16,
                    out_dim=16,
                    freq_dim=256,
                    text_dim=4096,
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    **model_config
                )
            else:
                self.model = WanModel(
                    model_type='t2v',
                    patch_size=(1, 2, 2),
                    text_len=512,
                    in_dim=16,
                    out_dim=16,
                    freq_dim=256,
                    text_dim=4096,
                    qk_norm=True,
                    cross_attn_norm=True,
                    eps=1e-6,
                    **model_config
                )

            # Load weights from safetensors
            # Remove 'model.' prefix from state_dict keys if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v  # Remove 'model.' prefix
                else:
                    new_state_dict[k] = v

            missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
            if missing_keys:
                print(f"Warning: Missing keys: {len(missing_keys)} keys")
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {len(unexpected_keys)} keys")
            print(f"Successfully loaded weights from {checkpoint_path}")
        else:
            # Original loading from pretrained directory
            model_root = _model_path(model_name, model_dir)
            if self.is_ti2v_5b:
                from wan_5b.modules.causal_model import CausalWanModel as Wan5BCausalWanModel
                from wan_5b.modules.model import WanModel as Wan5BWanModel

                model_cls = Wan5BCausalWanModel if is_causal else Wan5BWanModel
                self.model = model_cls.from_pretrained(
                    model_root,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                ) if is_causal else model_cls.from_pretrained(model_root)
            else:
                if is_causal:
                    model_cls = InferCausalWanModel if infer_only else CausalWanModel
                    self.model = model_cls.from_pretrained(
                        model_root, local_attn_size=local_attn_size, sink_size=sink_size)
                else:
                    self.model = WanModel.from_pretrained(model_root)

        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        # Token geometry must match the loaded backbone. Wan2.2-TI2V-5B uses
        # 44x80 latent spatial maps and 880 tokens per latent frame.
        if self.is_ti2v_5b:
            self.temporal_compression_ratio = 4
            self.spatial_compression_ratio = 16
            # Wan2.2-TI2V-5B VAE and transformer use 48 latent channels.
            self.latent_channels = 48
            self.frame_seq_length = 880
            self.seq_len = 36080  # 41 latent frames * 880 tokens (TIE-SF)
        else:
            self.temporal_compression_ratio = 4
            self.spatial_compression_ratio = 8
            self.latent_channels = 16
            self.frame_seq_length = 1560
            self.seq_len = 32760  # [1, 21, 16, 60, 104]
        self.post_init()

    def _call_model(self, *args, **kwargs):
        """Call the backbone through a hook used by the TIE wrapper."""
        return self.model(*args, **kwargs)

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class)
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt,
                                                        self.scheduler.sigmas,
                                                        self.scheduler.timesteps]
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt,
                                                      scheduler.sigmas,
                                                      scheduler.timesteps]
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor, conditional_dict: dict,
        timestep: torch.Tensor, kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        scene_cut: bool = False
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            model_kwargs = dict(
                t=input_timestep, context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
            )
            # The Wan2.1 causal model has a scene_cut argument; the native
            # Wan2.2-TI2V-5B model does not.  Passing it would fail before the
            # first training step, so keep the compatibility argument local to
            # the Wan2.1 path.
            if not self.is_ti2v_5b:
                model_kwargs["scene_cut"] = scene_cut
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                **model_kwargs
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep, context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self.model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep, context=prompt_embeds,
                        seq_len=self.seq_len
                    ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1)
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
