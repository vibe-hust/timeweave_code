"""Do not enable relative-RoPE extensions in the native SF / LL1 baselines."""
import inspect
import hashlib


def enforce_absolute_rope(pipeline, settings):
    assert settings['relative_rope'] is False
    model = pipeline.generator.model
    if hasattr(model, 'get_base_model'):
        model = model.get_base_model()
    touched = []
    for name, module in model.named_modules():
        if hasattr(module, 'use_relative_rope'):
            module.use_relative_rope = False
            touched.append(name)
    assert not any(getattr(m, 'use_relative_rope', False) for m in model.modules())
    source = inspect.getsourcefile(type(model))
    with open(source, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    return dict(relative_rope=False, native_backbone=type(model).__name__,
                explicit_flags_disabled=touched, backbone_source=source,
                backbone_source_sha256=digest)
