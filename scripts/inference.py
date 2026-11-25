"""Portable entrypoint adapted from the production SF-600 inference runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from protocol import sha256, write_json, event_plan
from inference_contract import configure_relative_rope, preserve_raw_key_precision, runtime_contract

REPOSITORY = Path(__file__).resolve().parents[1]
ROOT = REPOSITORY / "outputs"
sys.path.insert(0, str(REPOSITORY / "runtime"))


def tensor_hash(x):
    return hashlib.sha256(x.detach().contiguous().cpu().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()


def tokenizer_preflight():
    from wan_5b.modules.tokenizers import HuggingfaceTokenizer
    from utils.tie import pack_tokens
    cfg = json.loads((ROOT / 'config/run.json').read_text())
    path = Path(cfg['base_training_config']['tie_sf']['model_dir']) / 'Wan2.2-TI2V-5B/google/umt5-xxl'
    tokenizer = HuggingfaceTokenizer(name=str(path), seq_len=512, clean='whitespace', local_files_only=True)
    rows = json.loads((ROOT / 'data/generation.json').read_text())
    maximum = 0
    for row in rows:
        for block in row['blocks']:
            _, mask, _ = pack_tokens(block['prompt'], tokenizer)
            maximum = max(maximum, sum(mask))
    report = {'status': 'passed', 'videos': len(rows), 'blocks': sum(len(r['blocks']) for r in rows),
              'max_tokens': maximum, 'text_limit': 512, 'no_truncation': True}
    write_json(ROOT / 'reports/token_preflight.json', report)
    print(json.dumps(report), flush=True)


def build_pipeline(cfg, variant, device):
    import torch
    from peft import PeftModel
    from model_helpers import _merge_tie_sf_lora
    from utils.tie_models import build_model, configure_attention, TIEDiffusionWrapper, TIETextEncoder
    from utils.wan_wrapper import WanVAEWrapper
    from pipeline import CausalInferencePipeline
    spec = cfg['base_training_config']['tie_sf']
    models = Path(spec['model_dir'])
    configure_attention('sdpa')
    # Match the training loader: merge raw SF LoRA into FP32 ODE weights,
    # THEN cast the frozen base to BF16 and attach the separately trained RL adapter.
    backbone = build_model(models / spec['model_name'] / 'config.json', spec['base_model'],
                           causal=True, block_frames=6, local_frames=41, sink_frames=5,
                           dtype=torch.float32)
    backbone = _merge_tie_sf_lora(backbone, spec['lora_path'], rank=128, alpha=128,
        checkpoint_step=600, checkpoint_final=True, source_run_id=spec['source_run_id'],
        source_ode_step=3000)
    backbone.to(dtype=torch.bfloat16)
    backbone.first_block_frames = 5
    backbone.independent_first_frame = False
    configure_relative_rope(backbone, cfg["generation"])
    backbone.gradient_checkpointing = False
    if variant != 'sf600_base':
        adapter = Path(cfg['variants'][variant]['path'])
        if sha256(adapter / 'adapter_model.bin') != cfg['variants'][variant]['sha256']:
            raise ValueError('RL adapter identity changed')
        backbone = PeftModel.from_pretrained(backbone, str(adapter), is_trainable=False,
                                             autocast_adapter_dtype=True)
        from peft import get_peft_model_state_dict
        actual = get_peft_model_state_dict(backbone)
        expected = torch.load(adapter / 'adapter_model.bin', map_location='cpu', weights_only=True)
        if set(actual) != set(expected) or not all(torch.equal(actual[k].cpu(), expected[k].to(actual[k].dtype)) for k in actual):
            raise ValueError('EMA adapter did not load exactly')
        if not any(torch.count_nonzero(v) for k,v in actual.items() if 'lora_B' in k):
            raise ValueError('RL adapter is a no-op')
        del actual, expected
    generator = TIEDiffusionWrapper(backbone, causal=True, timestep_shift=5.,
                                   cross_attention_mode='global_untimed_joint_softmax')
    generator.to(device).eval().requires_grad_(False)
    encoder = TIETextEncoder(models / spec['model_name'], device='cpu')
    vae = WanVAEWrapper(model_name=spec['model_name'], model_dir=str(models)).to(device, dtype=torch.bfloat16)
    args = SimpleNamespace(first_block_frames=5, context_noise=0,
        model_kwargs={'local_attn_size': 41, 'sink_size': 5},
        denoising_step_list=[1000,750,500,250], warp_denoising_step=True,
        num_frame_per_block=6, independent_first_frame=False, tie_enabled=True, tie_video_fps=16.)
    return CausalInferencePipeline(args, device, generator=generator, text_encoder=encoder, vae=vae)


def generate_one(p, row, variant):
    import torch
    import av
    from utils.tie import to_device
    from single_event.rollout import EventRollout
    from stream_decode import _stream_decode_to_mp4
    dest = ROOT / 'artifacts/videos' / variant / row['video']
    report_path = ROOT / 'artifacts/generation' / variant / (Path(row['video']).stem + '.json')
    run_hash = sha256(ROOT / 'config/run.json')
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old['run_hash'] != run_hash or sha256(dest) != old['video_sha256']:
            raise ValueError('Existing video has a different contract')
        return old
    device = torch.device('cuda:0')
    started = time.time()
    p.release_inference_caches()
    p.text_encoder.to(device)
    encoded = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        for block in row['blocks']:
            key = tuple(block['indices'])
            if key not in encoded:
                cond = p.text_encoder.encode_tie([{'prompt': block['prompt']}], video_fps=16.)
                encoded[key] = to_device(cond, 'cpu', torch.bfloat16)
    p.text_encoder.to('cpu')
    del cond
    torch.cuda.empty_cache()
    # Reset both initial noise and every re-noising draw for each method.
    torch.manual_seed(row['seed'])
    torch.cuda.manual_seed(row['seed'])
    rng = torch.Generator(device=device).manual_seed(row['seed'])
    noise = torch.randn((1,row['latents'],48,44,80), device=device, dtype=torch.bfloat16, generator=rng)
    noise_hash = tensor_hash(noise)
    engine = EventRollout(p)
    engine.reset_cache(noise)
    preserve_raw_key_precision(p)
    inference_settings = runtime_contract(p, row)
    output = torch.empty_like(noise)
    block_hashes = []
    previous = None
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
        for block in row['blocks']:
            lo, hi = block['start'], block['stop']
            key = tuple(block['indices'])
            if previous != key:
                # Self-attention history stays intact; text KV must follow the new event set.
                p._initialize_crossattn_cache(1, noise.dtype, device)
                condition = to_device(encoded[key], device, torch.bfloat16)
                previous = key
            assert condition['tie']['token_start_s'].dtype == torch.float64
            assert condition['tie']['token_end_s'].dtype == torch.float64
            assert condition['tie']['token_is_global'].dtype == torch.bool
            noisy = noise[:,lo:hi]
            for i,t in enumerate(p.denoising_step_list):
                clean = engine.predict(noisy, condition, t, lo)
                if i + 1 < len(p.denoising_step_list):
                    noisy = p.scheduler.add_noise(clean.flatten(0,1), torch.randn_like(clean.flatten(0,1)),
                        torch.full((hi-lo,), float(p.denoising_step_list[i+1]), device=device,
                                   dtype=torch.float32)).unflatten(0, clean.shape[:2])
            output[:,lo:hi] = clean
            engine.predict(clean, condition, 0, lo)
            block_hashes.append({'start':lo, 'stop':hi, 'sha256':tensor_hash(clean)})
    del noise, noisy, clean, condition, encoded
    p.release_inference_caches()
    latent_hash = tensor_hash(output)
    _stream_decode_to_mp4(p.vae, output, device=device, fps=16, path=dest, expected_latents=row['latents'])
    del output
    with av.open(str(dest)) as video:
        stream = video.streams.video[0]
        assert stream.frames == row['frames'] and stream.width == 1280 and stream.height == 704
        assert float(stream.average_rate) == 16
    report = {'status':'completed', 'variant':variant, 'sample_id':row['sample_id'],
              'source_group':row['source_group'], 'seed':row['seed'], 'run_hash':run_hash,
              'initial_noise_sha256':noise_hash, 'latent_sha256':latent_hash,
              'blocks':block_hashes, 'frames':row['frames'], 'fps':16, 'width':1280, 'height':704,
              'video':str(dest), 'video_sha256':sha256(dest), 'video_bytes':dest.stat().st_size,
              'seconds':time.time()-started, 'inference_settings':inference_settings,
              'actual_denoising_timesteps':[float(t) for t in p.denoising_step_list]}
    write_json(report_path, report)
    print(json.dumps({k:v for k,v in report.items() if k!='blocks'}), flush=True)
    return report


def main():
    global ROOT
    parser = argparse.ArgumentParser(description="TimeWeave causal generation from an action schedule")
    parser.add_argument('--prompt', required=True, type=Path, help="JSON schedule (see examples)")
    parser.add_argument('--model-root', type=Path, help="Directory containing Wan2.2-TI2V-5B")
    parser.add_argument('--ode-checkpoint', type=Path, help="TimeWeave ODE-3000 full model")
    parser.add_argument('--sf-checkpoint', type=Path, help="Matching raw SF-600 generator adapter")
    parser.add_argument('--output', type=Path, default=Path('outputs/demo'))
    parser.add_argument('--seed', type=int, default=4042)
    parser.add_argument('--plan-only', action='store_true', help="Validate and print the block schedule without Torch or weights")
    args = parser.parse_args()
    row = json.loads(args.prompt.read_text())
    duration = float(row['duration'])
    if duration <= 0 or duration > 60 or abs(duration * 4 - round(duration * 4)) > 1e-9:
        parser.error('duration must be a positive multiple of 0.25 seconds, at most 60')
    columns = row['prompt']
    if len(columns) != 4 or not columns[0] or len({len(x) for x in columns}) != 1:
        parser.error('prompt must have four nonempty equal-length arrays: starts, ends, texts, roles')
    for start, end, text, role in zip(*columns):
        if not (0 <= start <= end <= duration) or not isinstance(text,str) or not text.strip():
            parser.error('Each condition needs a nonempty text and interval within the duration')
        if not isinstance(role,int) or role < 0:
            parser.error('role must be 0 for untimed global context or a positive participant ID')
    row.update(video='sample.mp4', sample_id='demo', source_group='user_schedule', seed=args.seed,
               latents=round(duration*4)+1, frames=round(duration*16)+1)
    row['blocks'] = event_plan(row,total=row['latents'])
    if args.plan_only:
        print(json.dumps(row,indent=2)); return
    if not all([args.model_root,args.ode_checkpoint,args.sf_checkpoint]):
        parser.error('Generation requires --model-root, --ode-checkpoint and --sf-checkpoint; weights are not bundled')
    for p in [args.model_root,args.ode_checkpoint,args.sf_checkpoint]:
        if not p.exists(): parser.error(f'Missing required asset: {p}')
    # Bind the public entrypoint to the inspected source checkpoint pair.
    expected = [('ODE', args.ode_checkpoint, '67f49bd8d051ea0dd07f2f6e9542fc1bd313151e2b9de6228adc651124d47ac3'),
                ('SF', args.sf_checkpoint, 'f05041a5bb8e3d5cfd700d97961bf37546f1575abcd405ce075d4ab3b40b1047')]
    for kind,p,digest in expected:
        if sha256(p)!=digest: parser.error(f'{kind} checkpoint does not match the released inference contract')
    ROOT=args.output.resolve()
    cfg={'base_training_config':{'tie_sf':{'model_dir':str(args.model_root.resolve()),
         'model_name':'Wan2.2-TI2V-5B','base_model':str(args.ode_checkpoint.resolve()),
         'lora_path':str(args.sf_checkpoint.resolve()),
         'source_run_id':'adact-tie-gu-e48-officialfloat-f5b6-ode3k-lora128-globalnorm-sf600-20261010'}},
         'generation':{'relative_rope':False,'rolling_radius_seconds':4,'global_untimed':True,'rolling_action_timestamps':'absolute_unclipped'},
         'variants':{'sf600_base':{'path':None}}}
    config_path=ROOT/'config/run.json'
    if config_path.exists() and json.loads(config_path.read_text())!=cfg:
        parser.error('Output contains a different configuration; choose a new --output directory')
    row_path=ROOT/'data/generation.json'
    if row_path.exists() and json.loads(row_path.read_text())!=[row]:
        parser.error('Output contains a different prompt or seed; choose a new --output directory')
    write_json(config_path,cfg);write_json(row_path,[row])
    import torch
    if not torch.cuda.is_available(): parser.error('CUDA is required for video generation')
    tokenizer_preflight()
    p=build_pipeline(cfg,'sf600_base',torch.device('cuda:0'))
    generate_one(p,row,'sf600_base')

if __name__ == '__main__':
    main()
