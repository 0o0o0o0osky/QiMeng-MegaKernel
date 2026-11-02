"""Create Qwen1.5-MoE A100 test globs from a real checkpoint and prompt."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.prefill import (
    InferenceConfig,
    _dtype_from_str,
    get_or_load_hf_model,
    run_prefill,
)
from qimeng_megakernel.models.LLM.preprocess import (
    preprocess_decode as llm_preprocess_decode,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.model_config import (
    QwenMoeConfig,
)

DEFAULT_MODEL_PATH = os.environ.get("QWEN_MOE_A2_7B_MODEL", "")
LOCAL_CHECKPOINT = Path(DEFAULT_MODEL_PATH).expanduser() if DEFAULT_MODEL_PATH else None
LOCAL_CHECKPOINT_AVAILABLE = bool(
    LOCAL_CHECKPOINT is not None
    and LOCAL_CHECKPOINT.is_dir()
    and (LOCAL_CHECKPOINT / "config.json").is_file()
)

SHORT_PROMPTS = ["Hello", "Tell me", "Python is"]


def make_globs_from_real_prefill(
    prompt: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda:0",
    torch_dtype: str = "auto",
    max_new_tokens: int = 2,
    return_model: bool = False,
) -> Globals | tuple[Globals, Any]:
    """run_prefill -> A100 preprocess_decode -> hidden_states = embedding(first_gen_token)."""
    config = InferenceConfig(model=model_path, device=device, torch_dtype=torch_dtype)
    prefill = run_prefill(config, [prompt])

    # Reuse the same cache key as run_prefill so the model is not loaded twice.
    model = get_or_load_hf_model(
        config.model, device=config.device, torch_dtype=_dtype_from_str(config.torch_dtype)
    )
    globs = llm_preprocess_decode(
        model_path,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=max_new_tokens,
        hardware="A100",
    )
    QwenMoeConfig.update_from_hf(model.config)

    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    globs.hidden_states.copy_(
        emb.squeeze(0).squeeze(0).to(globs.hidden_states.dtype)
    )

    if return_model:
        return globs, model
    return globs


def backup_globs(globs: Globals) -> dict[str, torch.Tensor]:
    """Backup mutable decode buffers (weights and flat views excluded)."""
    names = (
        "hidden_states",
        "q_post_rope",
        "k_cache",
        "v_cache",
        "attn_out",
        "attn_out_intermediates",
        "attn_lse_intermediates",
        "normed_hidden_states",
        "shared_expert_silu_out",
        "shared_out",
        "routed_out",
        "routed_out_f32",
        "routed_silu_buffer",
        "router_logits_buf",
        "expert_indices",
        "expert_weights",
        "silu_out",
        "logits",
        "barriers",
    )
    out: dict[str, torch.Tensor] = {}
    for name in names:
        t = getattr(globs, name, None)
        if isinstance(t, torch.Tensor):
            out[name] = t.clone()
    return out


def restore_globs(globs: Globals, backup: dict[str, torch.Tensor]) -> None:
    for key, value in backup.items():
        dest = getattr(globs, key, None)
        if isinstance(dest, torch.Tensor) and dest.shape == value.shape:
            dest.copy_(value)
