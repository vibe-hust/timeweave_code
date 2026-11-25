import os
from pathlib import Path
import torch
from utils.tie import decoded_frames

def _stream_decode_to_mp4(vae, latent, *, device, fps, path,
                          expected_latents):
    """Decode a long causal VAE sequence without materializing all pixels."""
    import av
    from wan_5b.modules.vae2_2 import unpatchify

    if latent.shape[0] != 1 or latent.shape[1] != expected_latents:
        raise ValueError("Visualization latent shape violates the clock contract")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.mp4")
    model = vae.model
    frame_count = 0
    model.clear_cache()
    try:
        with av.open(str(temporary), mode="w") as container:
            stream = container.add_stream("libx264", rate=int(fps))
            stream.width = int(latent.shape[-1] * 16)
            stream.height = int(latent.shape[-2] * 16)
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "preset": "medium"}
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.bfloat16, cache_enabled=False):
                mean = vae.mean.to(device=device, dtype=torch.bfloat16)
                inverse_std = (1.0 / vae.std).to(
                    device=device, dtype=torch.bfloat16)
                for index in range(expected_latents):
                    piece = latent[:, index:index + 1].to(
                        device=device, dtype=torch.bfloat16)
                    z = piece.permute(0, 2, 1, 3, 4)
                    z = (z / inverse_std.view(1, -1, 1, 1, 1)
                         + mean.view(1, -1, 1, 1, 1))
                    hidden = model.conv2(z)
                    model._conv_idx = [0]
                    decoded = model.decoder(
                        hidden, feat_cache=model._feat_map,
                        feat_idx=model._conv_idx, first_chunk=(index == 0))
                    decoded = unpatchify(decoded, patch_size=2)
                    expected_chunk_frames = 1 if index == 0 else 4
                    if decoded.shape[2] != expected_chunk_frames:
                        raise RuntimeError(
                            "Streaming VAE produced an unexpected chunk length")
                    frames = ((decoded[0].float().clamp(-1, 1) + 1)
                              * 127.5).byte().permute(
                                  1, 2, 3, 0).contiguous().cpu()
                    for array in frames.numpy():
                        frame = av.VideoFrame.from_ndarray(
                            array, format="rgb24")
                        for packet in stream.encode(frame):
                            container.mux(packet)
                    frame_count += int(frames.shape[0])
                    del piece, z, hidden, decoded, frames
                    if (index + 1) % 8 == 0:
                        torch.cuda.empty_cache()
            for packet in stream.encode():
                container.mux(packet)
        if frame_count != decoded_frames(expected_latents):
            raise RuntimeError("Streaming VAE decoded an unexpected length")
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        model.clear_cache()
        torch.cuda.empty_cache()
