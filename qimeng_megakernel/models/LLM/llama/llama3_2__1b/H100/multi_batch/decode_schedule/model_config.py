"""H100 multi-batch uses the same model constants as the single-batch route."""

from ...single_batch.decode_schedule.model_config import LlamaConfig

ModelConfig = LlamaConfig

__all__ = ["LlamaConfig", "ModelConfig"]
