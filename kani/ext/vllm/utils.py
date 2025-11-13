def max_context_size_from_autoconfig(model_id: str):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id)
    max_context_size = getattr(config, "model_max_len", getattr(config, "max_position_embeddings", None))

    if max_context_size is None:
        raise ValueError(
            "Could not infer the model's max context size from the config. Please pass the `max_context_size` arg."
        )
    return max_context_size
