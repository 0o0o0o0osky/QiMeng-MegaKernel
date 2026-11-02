"""Real-checkpoint prefill inputs for Llama 3.2 3B H100 verification."""
from __future__ import annotations

import os
from pathlib import Path

import torch

from qimeng_megakernel.models.LLM.decode import llm_preprocess_decode, _load_hf_model
from qimeng_megakernel.models.LLM.prefill import InferenceConfig, run_prefill
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.H100.single_batch.decode_schedule import model_config as _lc

DEFAULT_MODEL_PATH = os.environ.get("LLAMA_32_3B_MODEL_PATH", "")
LOCAL_CHECKPOINT = Path(DEFAULT_MODEL_PATH).expanduser() if DEFAULT_MODEL_PATH else Path()
LOCAL_CHECKPOINT_AVAILABLE = (
    bool(DEFAULT_MODEL_PATH)
    and LOCAL_CHECKPOINT.is_dir()
    and (LOCAL_CHECKPOINT / "config.json").is_file()
)

SHORT_PROMPTS = ["Hello", "Tell me", "Python is"]


def make_globs_from_real_prefill(
    prompt: str,
    model_path: str = DEFAULT_MODEL_PATH,
    device: str = "cuda:0",
    torch_dtype: str = "auto",
    max_new_tokens: int = 1,
    return_model: bool = False,
) -> Globals | tuple[Globals, any]:
    """
    Create globs with real prefill data from a prompt.

    Args:
        prompt: Input prompt string (keep SHORT - prefill time scales with prompt length!)
        model_path: Path to model
        device: Device to use
        torch_dtype: Torch dtype ("auto", "bfloat16", "float16", "float32")
        max_new_tokens: Max tokens for cache allocation (1 is sufficient for single-step decode tests)
        return_model: If True, also return the loaded model

    Returns:
        globs: Globals object with real prefill data
        model: (optional) Loaded model if return_model=True

    """
    # Run prefill
    config = InferenceConfig(model=model_path, device=device, torch_dtype=torch_dtype)
    prefill = run_prefill(config, [prompt])

    # Load model and preprocess for decode
    model = _load_hf_model(model_path, device=torch.device(device), dtype=prefill.logits.dtype)
    globs = llm_preprocess_decode(
        model_path,
        prefill=prefill,
        loaded_model=model,
        max_new_tokens=max_new_tokens,
        hardware="H100",
    )

    # Force 3B LlamaConfig so pyvm/fused use correct head_dim (128) and layer counts
    _lc.LlamaConfig.update_from_hf(model.config)
    _lc.LlamaConfig.head_dim = getattr(model.config, "head_dim", 128)
    _lc.LlamaConfig.num_attention_heads = getattr(model.config, "num_attention_heads", 24)
    _lc.LlamaConfig.num_key_value_heads = getattr(model.config, "num_key_value_heads", 8)
    _lc.LlamaConfig.hidden_size = getattr(model.config, "hidden_size", 3072)
    _lc.LlamaConfig.intermediate_size = getattr(model.config, "intermediate_size", 8192)
    _lc.LlamaConfig.num_hidden_layers = getattr(model.config, "num_hidden_layers", 28)

    # Initialize hidden_states with first token embedding (as in real decode)
    with torch.no_grad():
        emb = model.model.embed_tokens(prefill.first_gen_token)
    globs.hidden_states.copy_(emb.squeeze(0).squeeze(0).to(globs.hidden_states.dtype))

    if return_model:
        return globs, model
    return globs


def backup_globs(globs: Globals) -> dict[str, torch.Tensor]:
    """Backup all mutable buffers in globs."""
    return {
        "hidden_states": globs.hidden_states.clone(),
        "q_post_rope": globs.q_post_rope.clone(),
        "k_cache": globs.k_cache.clone(),
        "v_cache": globs.v_cache.clone(),
        "attn_out": globs.attn_out.clone(),
        "attn_out_intermediates": globs.attn_out_intermediates.clone(),
        "attn_lse_intermediates": globs.attn_lse_intermediates.clone(),
        "silu_out": globs.silu_out.clone(),
        "logits": globs.logits.clone(),
        "barriers": globs.barriers.clone(),
    }


def restore_globs(globs: Globals, backup: dict[str, torch.Tensor]) -> None:
    """Restore globs from backup."""
    for key, value in backup.items():
        getattr(globs, key).copy_(value)
