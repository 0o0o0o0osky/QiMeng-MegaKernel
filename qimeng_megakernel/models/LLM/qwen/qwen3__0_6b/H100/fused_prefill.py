from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.prefill import InferenceConfig, PrefillOutput
from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.qwen import load_weights as qwen_load_weights
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule.model_config import (
    QWEN3_H100_ATTENTION_INTERMEDIATE_COLS,
    Qwen3Config,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule.fused import layer_scheduler as fused_layer
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.H100.single_batch.decode_schedule.fused import operations as fused_ops


def _build_globs_for_prefill_from_weights(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    prompt_len: int,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int | None = None,
) -> Globals:
    Qwen3Config.update_from_dict(config)
    nlayers = config["num_hidden_layers"]
    nheads = config["num_attention_heads"]
    nkv = config.get("num_key_value_heads", nheads)
    head_dim = config.get("head_dim") or (config["hidden_size"] // nheads)
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]

    kv_seq_len = prompt_len + (max_new_tokens or 0)
    # These globs flow directly into the H100 megakernel after prefill.  Its
    # kv_cache_t is bf16, so the cache must already satisfy that native ABI;
    # central decode intentionally reuses prefill.globs without preprocessing.
    cache_dtype = torch.bfloat16
    k_cache = torch.zeros(
        (nlayers, 1, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=cache_dtype,
    )
    k_cache = _interleave_k_cache(k_cache, head_dim).contiguous()
    v_cache = torch.zeros(
        (nlayers, 1, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=cache_dtype,
    ).contiguous()

    def buf(shape, dt=dtype):
        return torch.zeros(shape, device=device, dtype=dt)

    def buf32(shape):
        return torch.zeros(shape, device=device, dtype=torch.float32)

    return Globals(
        position_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
        qkv_bias=None,
        o_proj_weights=weights["o_proj_weights"].contiguous(),
        attn_norm_weights=weights["attn_norm_weights"].contiguous(),
        q_norm_weights=weights["q_norm_weights"].contiguous(),
        k_norm_weights=weights["k_norm_weights"].contiguous(),
        mlp_norm_weights=weights["mlp_norm_weights"].contiguous(),
        up_proj_weights=weights["up_proj_weights"].contiguous(),
        gate_proj_weights=weights["gate_proj_weights"].contiguous(),
        down_proj_weights=weights["down_proj_weights"].contiguous(),
        lm_head_norm_weights=weights["lm_head_norm_weights"].contiguous(),
        lm_head_weights=weights["lm_head_weights"].contiguous(),
        embed_tokens=weights["embed_tokens"].contiguous(),
        rope_cos=weights["rope_cos"].contiguous(),
        rope_sin=weights["rope_sin"].contiguous(),
        hidden_states=buf(hidden_size),
        qkv_raw=buf(nheads * head_dim + 2 * nkv * head_dim),
        q_post_rope=buf(nheads * head_dim),
        attn_out=buf(nheads * head_dim),
        attn_out_intermediates=buf32(
            [nheads, QWEN3_H100_ATTENTION_INTERMEDIATE_COLS, head_dim]
        ),
        attn_lse_intermediates=buf32(
            [nheads, QWEN3_H100_ATTENTION_INTERMEDIATE_COLS]
        ),
        silu_out=buf(intermediate_size),
        logits=buf(vocab_size),
        barriers=torch.zeros([nlayers, 10, nheads + nkv * 2], dtype=torch.int32, device=device),
        hidden_states_seq=torch.zeros((prompt_len, hidden_size), device=device, dtype=dtype),
        prefill_qkv_raw=torch.zeros(
            (prompt_len, nheads * head_dim + 2 * nkv * head_dim), device=device, dtype=torch.float32
        ),
        # Q in float32 to avoid bf16 rounding (was max_abs~27 vs HF when stored in bf16).
        prefill_q_seq=torch.zeros((prompt_len, nheads, head_dim), device=device, dtype=torch.float32),
        attn_out_seq=torch.zeros((prompt_len, nheads * head_dim), device=device, dtype=dtype),
    )


def _resolve_to_disk_path(model_id_or_path: str) -> str | None:
    p = Path(model_id_or_path)
    if p.exists() and (p / "config.json").exists():
        return str(p)
    try:
        from huggingface_hub import snapshot_download

        resolved = snapshot_download(model_id_or_path, local_files_only=True)
        if Path(resolved, "config.json").exists():
            return resolved
    except Exception:
        return None
    return None


@torch.inference_mode()
def run_fused_prefill(
    config: InferenceConfig,
    prompts: list[str],
    *,
    loaded_model: Any = None,
    tokenizer: Any = None,
    hardware: str = "H100",
    max_new_tokens: int | None = None,
) -> PrefillOutput:
    del hardware
    from qimeng_megakernel.models.LLM.prefill import _dtype_from_str

    torch.cuda.set_device(config.device)
    device = torch.device(config.device)

    disk_path = _resolve_to_disk_path(config.model)
    if disk_path is None:
        raise NotImplementedError(
            "fused_prefill requires local/cached model files (config + safetensors); "
            f"cannot resolve disk weights for {config.model!r}."
        )

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(disk_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    tok = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=config.max_input_len is not None,
        max_length=config.max_input_len,
        add_special_tokens=config.add_special_tokens,
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    batch_size, prompt_len = input_ids.shape[0], input_ids.shape[1]
    if batch_size != 1:
        raise NotImplementedError("fused prefill only supports batch size 1")

    dtype = _dtype_from_str(config.torch_dtype)
    if dtype == "auto":
        dtype = torch.bfloat16

    dtype = dtype or torch.bfloat16
    config_dict, weights = qwen_load_weights.load_qwen3_weights_from_disk(
        disk_path, device=device, dtype=dtype
    )

    globs = _build_globs_for_prefill_from_weights(
        weights, config_dict, prompt_len, device, dtype, max_new_tokens=max_new_tokens
    )
    globs.hidden_states_seq.copy_(
        weights["embed_tokens"][input_ids.squeeze(0)].to(globs.hidden_states_seq.dtype)
    )

    for layer_idx in range(Qwen3Config.num_hidden_layers):
        fused_layer.run_fused_prefill_layer(globs, layer_idx, prompt_len)

    globs.hidden_states.copy_(globs.hidden_states_seq[prompt_len - 1].to(globs.hidden_states.dtype))
    fused_ops.fused_rms_lmhead_gemv(globs)

    logits = globs.logits.unsqueeze(0).unsqueeze(0)
    prompt_lens = attention_mask.sum(dim=-1)
    first_gen_token = torch.argmax(globs.logits, dim=-1).unsqueeze(0).unsqueeze(-1)
    pos_id_token_1 = prompt_lens.to(dtype=torch.long).unsqueeze(1)
    kv_seq_len_before = prompt_lens.to(dtype=torch.long)
    kv_seq_len_after = kv_seq_len_before + 1
    past_key_values = (globs.k_cache, globs.v_cache)

    return PrefillOutput(
        input_ids=input_ids,
        attention_mask=attention_mask,
        prompt_lens=prompt_lens,
        logits=logits,
        past_key_values=past_key_values,
        first_gen_token=first_gen_token,
        pos_id_token_1=pos_id_token_1,
        kv_seq_len_before=kv_seq_len_before,
        kv_seq_len_after=kv_seq_len_after,
        globs=globs,
    )
