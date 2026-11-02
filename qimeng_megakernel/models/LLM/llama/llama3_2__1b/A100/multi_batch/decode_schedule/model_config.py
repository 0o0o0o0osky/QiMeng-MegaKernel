"""Multi-batch LlamaConfig.

Inherits everything from the single_batch config but overrides
attn_kv_block_size = 64 to match the multi-batch attention kernel which
uses KV_TILE_SIZE=64 (4 warps fully active, M16N64 layout).
"""
from ...single_batch.decode_schedule.model_config import LlamaConfig as _SingleBatchConfig


class LlamaConfig(_SingleBatchConfig):
    # Multi-batch attention kernel uses KV_TILE_SIZE=64 (4 warps, M16N64).
    attn_kv_block_size = 64

    # CUDA constant aliases (upper-case, used in multi_batch kernel headers)
    NUM_LAYERS = _SingleBatchConfig.num_hidden_layers
    HIDDEN_DIM = _SingleBatchConfig.hidden_size
    HEAD_DIM   = _SingleBatchConfig.head_dim
    NUM_Q_HEADS  = _SingleBatchConfig.num_attention_heads
    NUM_KV_HEADS = _SingleBatchConfig.num_key_value_heads
    NUM_HEADS    = _SingleBatchConfig.num_heads
    QKV_OUT_DIM  = _SingleBatchConfig.qkv_out_dim
    INTERMEDIATE_SIZE = _SingleBatchConfig.intermediate_size
    VOCAB_SIZE   = _SingleBatchConfig.vocab_size


ModelConfig = LlamaConfig

__all__ = ["LlamaConfig", "ModelConfig"]
