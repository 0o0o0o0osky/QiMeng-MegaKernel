"""Real-checkpoint fixtures shared by the Qwen2.5 3B A100 verifier stages."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import torch

from qimeng_megakernel.models.LLM.decode import _load_hf_model
from qimeng_megakernel.models.LLM.prefill import InferenceConfig, run_prefill
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.preprocess import (
    preprocess_decode,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.A100.single_batch.decode_schedule.model_config import (
    Qwen2Config,
)

QWEN_TEST_MODEL_ENV = "QWEN_TEST_MODEL"
_CONFIGURED_MODEL_PATH = os.environ.get(QWEN_TEST_MODEL_ENV, "").strip()
LOCAL_CHECKPOINT = (
    Path(_CONFIGURED_MODEL_PATH).expanduser() if _CONFIGURED_MODEL_PATH else None
)
LOCAL_CHECKPOINT_AVAILABLE = bool(
    LOCAL_CHECKPOINT is not None
    and LOCAL_CHECKPOINT.is_dir()
    and (LOCAL_CHECKPOINT / "config.json").is_file()
)
SHORT_PROMPTS = ["Hello", "Tell me", "Python is"]


def require_local_qwen_test_model(
    model_path: str | os.PathLike[str] | None = None,
) -> str:
    configured = (
        os.fspath(model_path)
        if model_path is not None
        else os.environ.get(QWEN_TEST_MODEL_ENV, "")
    )
    if not configured.strip():
        raise RuntimeError(
            f"{QWEN_TEST_MODEL_ENV} must name a local checkpoint directory"
        )
    root = Path(configured).expanduser()
    if not root.is_dir() or not (root / "config.json").is_file():
        raise FileNotFoundError(f"invalid local Qwen checkpoint: {configured!r}")
    return str(root.resolve())


def make_globs_from_real_prefill(
    prompt: str,
    model_path: str | os.PathLike[str] | None = None,
    device: str = "cuda:0",
    torch_dtype: str = "auto",
    max_new_tokens: int = 256,
    return_model: bool = False,
) -> Globals | tuple[Globals, Any]:
    model_path = require_local_qwen_test_model(model_path)
    config = InferenceConfig(
        model=model_path,
        device=device,
        torch_dtype=torch_dtype,
    )
    prefill = run_prefill(config, [prompt])
    dtype = prefill.logits.dtype
    model = _load_hf_model(model_path, device=torch.device(device), dtype=dtype)
    globs = preprocess_decode(
        prefill=prefill,
        model=model,
        max_new_tokens=max_new_tokens,
        for_mk=True,
    )
    Qwen2Config.update_from_hf(model.config)
    if return_model:
        return globs, model
    return globs


def backup_globs(globs: Globals) -> dict[str, Optional[torch.Tensor]]:
    backup: dict[str, Optional[torch.Tensor]] = {}
    mutable_names = (
        "hidden_states",
        "q_post_rope",
        "k_cache",
        "v_cache",
        "attn_out",
        "attn_lse_intermediates",
        "attn_out_intermediates",
        "silu_out",
        "logits",
        "barriers",
    )
    for name in mutable_names:
        value = getattr(globs, name, None)
        if isinstance(value, torch.Tensor):
            backup[name] = value.clone()
    return backup


def restore_globs(
    globs: Globals,
    backup: dict[str, Optional[torch.Tensor]],
) -> None:
    for name, value in backup.items():
        destination = getattr(globs, name, None)
        if (
            value is not None
            and isinstance(destination, torch.Tensor)
            and destination.shape == value.shape
        ):
            destination.copy_(value)
