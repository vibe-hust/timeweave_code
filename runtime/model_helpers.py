import torch
from peft import LoraConfig, get_peft_model
from utils.tie_checkpoint import load_generator_lora_checkpoint

def _configure_lora_for_wan(transformer, rank=256, alpha=256, dropout=0.0, adapter_target_modules=None):
    """Configure LoRA for a WanDiffusionWrapper model"""
    if adapter_target_modules is None:
        adapter_target_modules = ['CausalWanAttentionBlock']
    target_linear_modules = set()

    for name, module in transformer.named_modules():
        if module.__class__.__name__ in adapter_target_modules:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)

    target_linear_modules = list(target_linear_modules)

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_linear_modules,
        bias="none",
        task_type=None
    )

    lora_model = get_peft_model(transformer, peft_config)
    return lora_model

def _merge_tie_sf_lora(
    model,
    checkpoint_path,
    *,
    rank=128,
    alpha=128,
    dropout=0.0,
    target_modules=None,
    checkpoint_step=800,
    checkpoint_final=False,
    source_run_id=None,
    source_ode_step=None,
):
    """Load the frozen TIE-SF generator adapter and merge it into the base.

    The returned module is a plain CausalWanModel.  NFT then attaches its own
    adapter to that merged module, so TIE-SF weights are part of the fixed
    input and cannot accidentally be updated by the NFT optimizer.
    """
    state = load_generator_lora_checkpoint(
        checkpoint_path,
        expected_rank=rank,
        expected_step=checkpoint_step,
        expected_final=checkpoint_final,
        expected_source_run_id=source_run_id,
        expected_ode_step=source_ode_step,
    )
    tie_adapter = _configure_lora_for_wan(
        model,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        adapter_target_modules=target_modules or ["CausalWanAttentionBlock"],
    )
    # The reviewed Self-Forcing checkpoint was saved from the outer generator
    # wrapper, so its keys start with ``model.base_model``.  ``tie_adapter``
    # itself starts at ``base_model``.  Load its native state-dict keys
    # directly: PEFT's helper is intended for adapter exports without an
    # adapter-name component and would turn ``default`` into
    # ``default.default``, silently leaving every LoRA tensor unloaded.
    native_state = {
        key.removeprefix("model."): value for key, value in state.items()
    }
    load_result = tie_adapter.load_state_dict(native_state, strict=False)
    unexpected = [
        key for key in getattr(load_result, "unexpected_keys", [])
        if "lora_" in key
    ]
    missing_lora = [
        key for key in getattr(load_result, "missing_keys", [])
        if "lora_" in key
    ]
    if unexpected or missing_lora:
        raise ValueError(
            "TIE-SF generator LoRA did not load exactly: "
            f"unexpected={unexpected[:8]}, missing_lora={missing_lora[:8]}"
        )
    try:
        merged = tie_adapter.merge_and_unload(safe_merge=True)
    except TypeError:
        # Older PEFT releases do not expose ``safe_merge``.
        merged = tie_adapter.merge_and_unload()
    merged._tie_sf_source_adapter_keys = len(state)
    merged._tie_sf_source_adapter_rank = rank
    merged.eval().requires_grad_(False)
    return merged
