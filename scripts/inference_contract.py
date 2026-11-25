"""Runtime guards for explicit absolute RoPE and rolling prompts on the GU models."""


def backbone_of(pipeline):
    model = pipeline.generator.model
    return model.get_base_model() if hasattr(model, 'get_base_model') else model


def configure_relative_rope(backbone, settings):
    if settings['relative_rope'] is not False:
        raise ValueError('This evaluation explicitly disables relative RoPE')
    if settings['rolling_radius_seconds'] != 4 or settings['global_untimed'] is not True:
        raise ValueError('Expected rolling +/-4 seconds and GlobalUntimed')
    backbone.use_relative_rope = False


def preserve_raw_key_precision(pipeline):
    """Absolute RoPE uses native rotated-key caches; no raw-K widening."""
    assert backbone_of(pipeline).use_relative_rope is False


def runtime_contract(pipeline, row):
    import torch
    model = backbone_of(pipeline)
    assert model.use_relative_rope is False
    assert pipeline.generator.cross_attention_mode == 'global_untimed_joint_softmax'
    roles = row['prompt'][3]
    globals_ = {i for i, role in enumerate(roles) if role == 0}
    for block in row['blocks']:
        assert globals_ <= set(block['indices'])
        for column, selected in zip(row['prompt'], block['prompt'], strict=True):
            assert [column[i] for i in block['indices']] == selected
    return dict(relative_rope=False, cross_attention=pipeline.generator.cross_attention_mode,
                rolling_radius_seconds=4, action_timestamps='absolute_unclipped',
                global_events_always_visible=True, raw_key_precision_preserved=False, key_cache_policy="native_rotated_keys",
                key_dtypes=sorted({str(c['k'].dtype) for c in pipeline.kv_cache1}),
                value_dtypes=sorted({str(c['v'].dtype) for c in pipeline.kv_cache1}))
