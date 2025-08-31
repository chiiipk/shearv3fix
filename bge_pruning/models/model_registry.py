from .composer_bge_m3 import ComposerBGEM3

COMPOSER_MODEL_REGISTRY = {

    # BGE-M3 embedding models
    'bge_m3_base': ComposerBGEM3,
    'bge_m3_75pct': ComposerBGEM3,
    'bge_m3_60pct': ComposerBGEM3,
    'bge_m3_50pct': ComposerBGEM3,
}

def get_model_class(model_name: str):
    """Get model class from registry"""
    if model_name in COMPOSER_MODEL_REGISTRY:
        model_class = COMPOSER_MODEL_REGISTRY[model_name]
        if isinstance(model_class, str):
            # Import dynamically for backward compatibility
            module_path, class_name = model_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        return model_class
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def register_model(name: str, model_class):
    """Register a new model class"""
    COMPOSER_MODEL_REGISTRY[name] = model_class
