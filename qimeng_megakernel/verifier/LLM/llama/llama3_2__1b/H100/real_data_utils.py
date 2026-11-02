"""Real-checkpoint prefill inputs for Llama 3.2 1B H100 verification."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.globs import (
    Globals,
)
from qimeng_megakernel.models.LLM.prefill import (
    InferenceConfig,
    get_or_load_hf_model,
    run_prefill_fused,
)
from qimeng_megakernel.verifier.common.real_data import (
    RealDataSpec,
    bind,
)

DEFAULT_MODEL_PATH = os.environ.get("MODEL_LLAMA3_2_1B", "")

SHORT_PROMPTS = ["Hello", "Tell me", "Python is"]

SPEC = RealDataSpec(default_model_path=DEFAULT_MODEL_PATH, hardware="H100")

LOCAL_CHECKPOINT = (
    Path(DEFAULT_MODEL_PATH).expanduser() if DEFAULT_MODEL_PATH else Path()
)
LOCAL_CHECKPOINT_AVAILABLE = (
    bool(DEFAULT_MODEL_PATH)
    and LOCAL_CHECKPOINT.is_dir()
    and (LOCAL_CHECKPOINT / "config.json").is_file()
)
_bound = bind(SPEC)
make_globs_from_real_prefill = _bound["make_globs_from_real_prefill"]
backup_globs = _bound["backup_globs"]
restore_globs = _bound["restore_globs"]


def distinct_prompts_for_batch(batch_size: int) -> list[str]:
    return [
        f"{SHORT_PROMPTS[i % len(SHORT_PROMPTS)]} {'token ' * (i % 5)}#{i}"
        for i in range(batch_size)
    ]


@torch.inference_mode()
def make_globs_from_real_prefill_multi(
    prompts: list[str],
    *,
    model_path: str | None = None,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    max_new_tokens: int = 16,
    return_model: bool = False,
):
    """Build production H100 multi-batch Globals and seed the first decode input."""
    batch_size = len(prompts)
    if not 2 <= batch_size <= 16:
        raise ValueError(f"expected 2 <= batch_size <= 16, got {batch_size}")
    path = model_path or DEFAULT_MODEL_PATH
    prefill = run_prefill_fused(
        InferenceConfig(path, device=device, torch_dtype=torch_dtype),
        prompts,
        hardware="H100",
        max_new_tokens=max_new_tokens,
    )
    globs = prefill.globs
    if globs is None:
        raise RuntimeError("H100 fused prefill did not return Globals")
    token_ids = prefill.first_gen_token.squeeze(-1)
    globs.hidden_states.copy_(
        globs.embed_tokens[token_ids].to(globs.hidden_states.dtype)
    )
    globs.barriers.zero_()
    globs.validate_padded_storage()
    if return_model:
        model = get_or_load_hf_model(
            path,
            device=device,
            torch_dtype=torch.bfloat16,
        )
        return globs, model
    return globs


__all__ = [
    "DEFAULT_MODEL_PATH",
    "LOCAL_CHECKPOINT",
    "LOCAL_CHECKPOINT_AVAILABLE",
    "SHORT_PROMPTS",
    "SPEC",
    "Globals",
    "backup_globs",
    "distinct_prompts_for_batch",
    "make_globs_from_real_prefill",
    "make_globs_from_real_prefill_multi",
    "restore_globs",
]
