"""
Utilities for creating test globs with real prefill data (Qwen2.5 3B H100).

Used by Python VM (PyVM) versus MegaKernel tests that require a real prompt and
official framework prefill so decode receives real KV, RoPE, and embeddings.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from qimeng_megakernel.models.LLM.decode import _load_hf_model
from qimeng_megakernel.models.LLM.preprocess import preprocess_decode as llm_preprocess_decode
from qimeng_megakernel.models.LLM.prefill import InferenceConfig, run_prefill
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen2_5__3b.H100.single_batch.decode_schedule.model_config import Qwen2Config

DEFAULT_MODEL_PATH = os.environ.get("QWEN_TEST_MODEL", "")
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
) -> Globals | tuple[Globals, any]:
    """
    Create globs with real prefill data from a prompt, for PyVM vs MK tests.

    Runs: run_prefill -> llm_preprocess_decode(..., for_mk=True) -> set hidden_states
    to first_gen_token embedding. Intermediates and barriers are reshaped to match
    the verifier kernel (attn 4D, barriers 6 rows).
    """
    config = InferenceConfig(model=model_path, device=device, torch_dtype=torch_dtype)
    prefill = run_prefill(config, [prompt])
    dtype = prefill.logits.dtype
    model = _load_hf_model(model_path, device=torch.device(device), dtype=dtype)

    globs = llm_preprocess_decode(
        model_path,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=max_new_tokens,
        hardware="H100",
        for_mk=True,
    )

    Qwen2Config.update_from_hf(model.config)
    n_heads = Qwen2Config.num_attention_heads
    hd = Qwen2Config.head_dim
    # qkv_bias: preprocess returns (L, 2560); kernel expects (L, QKV_NUM_BLOCKS, 16)
    if globs.qkv_bias is not None and globs.qkv_bias.dim() == 2:
        globs.qkv_bias = globs.qkv_bias.view(globs.qkv_bias.size(0), -1, 16).contiguous()
    # Kernel expects barriers (L, 6, slots); preprocess may return (L, 10, slots)
    if globs.barriers.shape[1] > 6:
        globs.barriers = globs.barriers[:, :6, :].contiguous()
    # attn_out_intermediates: preprocess (n_heads, cols, hd) -> kernel (1, n_heads, cols, hd)
    if globs.attn_out_intermediates is not None and globs.attn_out_intermediates.dim() == 3:
        globs.attn_out_intermediates = globs.attn_out_intermediates.unsqueeze(0).contiguous()
    # attn_lse_intermediates: (n_heads, cols) -> (1, 1, n_heads, cols)
    if globs.attn_lse_intermediates is not None and globs.attn_lse_intermediates.dim() == 2:
        globs.attn_lse_intermediates = globs.attn_lse_intermediates.unsqueeze(0).unsqueeze(0).contiguous()

    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    # hidden_states: kernel expects 4D (1,1,1,H); 1D (H,) is padded by pyutils
    h = globs.hidden_states
    if h.dim() == 1:
        globs.hidden_states = emb.squeeze(0).squeeze(0).to(h.dtype).contiguous()
    else:
        globs.hidden_states.copy_(emb.view_as(h).to(h.dtype))

    if return_model:
        return globs, model
    return globs


def backup_globs(globs: Globals) -> dict[str, torch.Tensor]:
    """Backup mutable buffers in globs (for fused_to_pyvm tests)."""
    out = {}
    for name in ("hidden_states", "q_post_rope", "k_cache", "v_cache", "attn_out", "silu_out", "logits", "barriers"):
        t = getattr(globs, name, None)
        if t is not None and isinstance(t, torch.Tensor):
            out[name] = t.clone()
    if getattr(globs, "attn_out_intermediates", None) is not None:
        out["attn_out_intermediates"] = globs.attn_out_intermediates.clone()
    if getattr(globs, "attn_lse_intermediates", None) is not None:
        out["attn_lse_intermediates"] = globs.attn_lse_intermediates.clone()
    return out


def restore_globs(globs: Globals, backup: dict[str, torch.Tensor]) -> None:
    """Restore globs from backup. Skips tensors whose shape changed (e.g. intermediates)."""
    for key, value in backup.items():
        if hasattr(globs, key):
            dest = getattr(globs, key)
            if isinstance(dest, torch.Tensor) and isinstance(value, torch.Tensor) and dest.shape == value.shape:
                dest.copy_(value)
