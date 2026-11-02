"""Llama 3.2 3B model config (A100 single-batch)."""


class LlamaConfig:
    num_hidden_layers = 28
    hidden_size = 3072
    intermediate_size = 8192
    head_dim = 128
    num_attention_heads = 24
    num_key_value_heads = 8
    num_heads = num_attention_heads + num_key_value_heads * 2
    qkv_out_dim = num_heads * head_dim
    vocab_size = 128256
    rms_norm_eps = 1e-5
    attn_kv_block_size = 16
    attn_reduction_size = 3

    @classmethod
    def update_from_hf(cls, config):
        cls.num_hidden_layers = config.num_hidden_layers
        cls.hidden_size = config.hidden_size
        cls.intermediate_size = config.intermediate_size
        cls.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        cls.num_attention_heads = config.num_attention_heads
        cls.num_key_value_heads = config.num_key_value_heads
        cls.num_heads = cls.num_attention_heads + cls.num_key_value_heads * 2
        cls.qkv_out_dim = cls.num_heads * cls.head_dim
        cls.vocab_size = config.vocab_size
        cls.rms_norm_eps = config.rms_norm_eps
        if hasattr(config, "attn_kv_block_size"):
            cls.attn_kv_block_size = config.attn_kv_block_size
        if hasattr(config, "attn_reduction_size"):
            cls.attn_reduction_size = config.attn_reduction_size

    @classmethod
    def update_from_dict(cls, config: dict):
        cls.num_hidden_layers = config["num_hidden_layers"]
        cls.hidden_size = config["hidden_size"]
        cls.intermediate_size = config["intermediate_size"]
        cls.head_dim = config.get("head_dim") or (
            config["hidden_size"] // config["num_attention_heads"]
        )
        cls.num_attention_heads = config["num_attention_heads"]
        cls.num_key_value_heads = config.get(
            "num_key_value_heads",
            cls.num_attention_heads,
        )
        cls.num_heads = cls.num_attention_heads + cls.num_key_value_heads * 2
        cls.qkv_out_dim = cls.num_heads * cls.head_dim
        cls.vocab_size = config["vocab_size"]
        cls.rms_norm_eps = config.get("rms_norm_eps", 1e-5)
        if "attn_kv_block_size" in config:
            cls.attn_kv_block_size = config["attn_kv_block_size"]
        if "attn_reduction_size" in config:
            cls.attn_reduction_size = config["attn_reduction_size"]

    @classmethod
    def as_dict(cls):
        return {
            "num_hidden_layers": cls.num_hidden_layers,
            "hidden_size": cls.hidden_size,
            "intermediate_size": cls.intermediate_size,
            "head_dim": cls.head_dim,
            "num_attention_heads": cls.num_attention_heads,
            "num_key_value_heads": cls.num_key_value_heads,
            "num_heads": cls.num_heads,
            "qkv_out_dim": cls.qkv_out_dim,
            "vocab_size": cls.vocab_size,
            "rms_norm_eps": cls.rms_norm_eps,
            "attn_kv_block_size": cls.attn_kv_block_size,
            "attn_reduction_size": cls.attn_reduction_size,
        }


__all__ = ["LlamaConfig"]
