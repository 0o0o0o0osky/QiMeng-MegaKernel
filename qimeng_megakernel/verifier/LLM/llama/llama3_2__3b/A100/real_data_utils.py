"""Real-checkpoint inputs for Llama 3.2 3B A100 verification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.llama.llama3_2__3b.A100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.prefill import InferenceConfig, run_prefill_fused
from qimeng_megakernel.verifier.common.real_data import (
    LONG_PROMPT,
    MEDIUM_PROMPT,
    SHORT_PROMPTS as COMMON_SHORT_PROMPTS,
    RealDataSpec,
    bind,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__3b.A100.verifier_config import CONFIG


def default_llama_3_2_3b_model_dir() -> str:
    """Resolve a local checkpoint only; verifier runs never download weights."""
    configured = os.environ.get("LLAMA_32_3B_MODEL_PATH", "").strip()
    path = Path(configured).expanduser() if configured else Path()
    if configured and path.is_dir() and (path / "config.json").is_file():
        return str(path.resolve())
    return ""


DEFAULT_MODEL_PATH = default_llama_3_2_3b_model_dir()
LOCAL_CHECKPOINT = (
    Path(DEFAULT_MODEL_PATH).expanduser() if DEFAULT_MODEL_PATH else Path()
)
LOCAL_CHECKPOINT_AVAILABLE = (
    bool(DEFAULT_MODEL_PATH)
    and LOCAL_CHECKPOINT.is_dir()
    and (LOCAL_CHECKPOINT / "config.json").is_file()
)

SHORT_PROMPTS = list(COMMON_SHORT_PROMPTS)
SPEC = RealDataSpec(
    default_model_path=DEFAULT_MODEL_PATH,
    hardware=CONFIG.hardware,
)
_bound = bind(SPEC)
make_globs_from_real_prefill = _bound["make_globs_from_real_prefill"]


_GLOBS_MUTABLE_TENSOR_FIELDS = (
    "hidden_states",
    "q_post_rope",
    "k_cache",
    "v_cache",
    "attn_out",
    "attn_out_intermediates",
    "attn_lse_intermediates",
    "attn_partial_valid",
    "silu_out",
    "logits",
    "barriers",
    "hidden_states_seq",
    "prefill_q_seq",
    "attn_out_seq",
    "position_ids",
    "kv_lengths",
    "active_mask",
    "eos_reached",
    "last_generated_tokens",
    "attn_rms_out",
    "mlp_rms_out",
    "lm_head_rms_out",
)


def detach_globs_mutable_tensors(globs: Any) -> None:
    """Detach tensors created under inference mode before mutable decode tests."""
    for name in _GLOBS_MUTABLE_TENSOR_FIELDS:
        value = getattr(globs, name, None)
        if isinstance(value, torch.Tensor):
            setattr(globs, name, value.clone())


def distinct_prompts_for_batch(batch_size: int) -> list[str]:
    return [
        f"{SHORT_PROMPTS[index % len(SHORT_PROMPTS)]} "
        f"{'token ' * (index % 5)}#{index}"
        for index in range(batch_size)
    ]


def make_globs_from_real_prefill_multi(
    prompts: list[str],
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda:0",
    torch_dtype: str = "auto",
    max_new_tokens: int = 8,
):
    """Retain the legacy A100 multi-batch fixture while B=1 uses common binding."""
    batch_size = len(prompts)
    if not 2 <= batch_size <= 16:
        raise ValueError(f"expected 2 <= batch_size <= 16, got {batch_size}")
    prefill = run_prefill_fused(
        InferenceConfig(model=model_path, device=device, torch_dtype=torch_dtype),
        prompts,
        hardware=CONFIG.hardware,
        max_new_tokens=max_new_tokens,
    )
    globs = prefill.globs
    if globs is None:
        raise RuntimeError("A100 fused prefill did not return Globals")

    with torch.inference_mode():
        globs.position_ids = prefill.pos_id_token_1.squeeze(1).clone()
        globs.position_id = int(prefill.pos_id_token_1.max().item())
        token_ids = prefill.first_gen_token.squeeze(-1)
        globs.hidden_states.copy_(
            globs.embed_tokens[token_ids].to(globs.hidden_states.dtype)
        )
        globs.barriers.zero_()
    detach_globs_mutable_tensors(globs)
    return globs


def backup_globs(globs: Any) -> dict[str, torch.Tensor | None]:
    """Snapshot all mutable fields used by the B=1 and legacy B>1 suites."""
    backup: dict[str, torch.Tensor | None] = {}
    for name in _GLOBS_MUTABLE_TENSOR_FIELDS:
        value = getattr(globs, name, None)
        if isinstance(value, torch.Tensor):
            backup[name] = value.clone()
        elif name in (
            "attn_partial_valid",
            "position_ids",
            "kv_lengths",
            "active_mask",
            "eos_reached",
            "last_generated_tokens",
        ):
            backup[name] = None
    return backup


def restore_globs(globs: Any, backup: dict[str, torch.Tensor | None]) -> None:
    """Restore snapshots without changing references to existing tensor buffers."""
    replaceable = {
        "attn_partial_valid",
        "position_ids",
        "kv_lengths",
        "active_mask",
        "eos_reached",
        "last_generated_tokens",
    }
    for name, value in backup.items():
        if name in replaceable:
            setattr(globs, name, None if value is None else value.clone())
        else:
            assert value is not None
            getattr(globs, name).copy_(value)


def backup_globs_multi(globs: Any) -> dict[str, torch.Tensor | None]:
    return backup_globs(globs)


def restore_globs_multi(
    globs: Any,
    backup: dict[str, torch.Tensor | None],
) -> None:
    restore_globs(globs, backup)


def print_globs_info(globs: Any, prefix: str = "") -> None:
    """Print compact state diagnostics for legacy multi-batch debugging."""
    print(f"{prefix}Globs info:")
    print(f"{prefix}  position_id: {globs.position_id}")
    position_ids = getattr(globs, "position_ids", None)
    print(
        f"{prefix}  position_ids: "
        f"{None if position_ids is None else position_ids.tolist()}"
    )
    print(
        f"{prefix}  hidden_states: shape={globs.hidden_states.shape}, "
        f"range=[{globs.hidden_states.min():.3f}, {globs.hidden_states.max():.3f}], "
        f"mean={globs.hidden_states.mean():.3f}"
    )


__all__ = [
    "DEFAULT_MODEL_PATH",
    "LOCAL_CHECKPOINT",
    "LOCAL_CHECKPOINT_AVAILABLE",
    "LONG_PROMPT",
    "MEDIUM_PROMPT",
    "SHORT_PROMPTS",
    "SPEC",
    "Globals",
    "backup_globs",
    "backup_globs_multi",
    "default_llama_3_2_3b_model_dir",
    "detach_globs_mutable_tensors",
    "distinct_prompts_for_batch",
    "make_globs_from_real_prefill",
    "make_globs_from_real_prefill_multi",
    "print_globs_info",
    "restore_globs",
    "restore_globs_multi",
]
