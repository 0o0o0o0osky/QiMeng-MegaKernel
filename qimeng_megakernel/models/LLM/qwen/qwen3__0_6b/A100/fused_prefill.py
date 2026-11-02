# A100 (Ampere) fused prefill for Qwen3-0.6B: tokenize + our fused ops only (no HF forward).
# Layout matches A100 preprocess: KV 5D interleaved, route-owned attention workspace,
# and six opcode-barrier rows.
# Weights: from disk when possible (local dir or HF cache path); only load full HF model when no disk path available.
# Qwen3 differences vs Qwen2.5: q_norm/k_norm per head, no qkv_bias, tie_word_embeddings=True.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.prefill import InferenceConfig, PrefillOutput
from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.preprocess import _stack_qwen3_weights_from_hf
from qimeng_megakernel.models.LLM.qwen import load_weights as qwen_load_weights
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.globs import Globals
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.model_config import Qwen3Config
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.instructions.instruction_defs import (
    ATTENTION_REDUCTION_WORKSPACE_CAPACITY,
)
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.fused import layer_scheduler as fused_layer
from qimeng_megakernel.models.LLM.qwen.qwen3__0_6b.A100.single_batch.decode_schedule.fused import operations as fused_ops


def _get_sm_count(device):
    if getattr(device, "type", None) != "cuda":
        return 1
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1


def _build_globs_for_prefill_from_weights(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    prompt_len: int,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int | None = None,
) -> Globals:
    """Build A100 Globals from pre-loaded Qwen3 weights."""
    Qwen3Config.update_from_dict(config)
    nlayers = config["num_hidden_layers"]
    nheads = config["num_attention_heads"]
    nkv = config.get("num_key_value_heads", nheads)
    head_dim = config.get("head_dim") or (config["hidden_size"] // nheads)
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]

    kv_seq_len = prompt_len + (max_new_tokens or 0)
    k_cache = torch.zeros(
        (nlayers, 1, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=dtype,
    )
    k_cache = _interleave_k_cache(k_cache, head_dim).contiguous()
    v_cache = torch.zeros(
        (nlayers, 1, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=dtype,
    ).contiguous()

    max_attn = ATTENTION_REDUCTION_WORKSPACE_CAPACITY

    def buf(s, dt=dtype):
        return torch.zeros(s, device=device, dtype=dt)

    def buf32(s):
        return torch.zeros(s, device=device, dtype=torch.float32)

    return Globals(
        position_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
        q_norm_weights=weights["q_norm_weights"].contiguous(),
        k_norm_weights=weights["k_norm_weights"].contiguous(),
        o_proj_weights=weights["o_proj_weights"].contiguous(),
        attn_norm_weights=weights["attn_norm_weights"].contiguous(),
        mlp_norm_weights=weights["mlp_norm_weights"].contiguous(),
        up_proj_weights=weights["up_proj_weights"].contiguous(),
        gate_proj_weights=weights["gate_proj_weights"].contiguous(),
        down_proj_weights=weights["down_proj_weights"].contiguous(),
        lm_head_norm_weights=weights["lm_head_norm_weights"].contiguous(),
        lm_head_weights=weights["lm_head_weights"].contiguous(),
        rope_cos=weights["rope_cos"].contiguous(),
        rope_sin=weights["rope_sin"].contiguous(),
        hidden_states=buf(hidden_size),
        qkv_raw=buf(nheads * head_dim + 2 * nkv * head_dim),
        q_post_rope=buf(nheads * head_dim),
        attn_out=buf(nheads * head_dim),
        attn_out_intermediates=buf32([nheads, max_attn, head_dim]),
        attn_lse_intermediates=buf32([nheads, max_attn]),
        silu_out=buf(intermediate_size),
        logits=buf(vocab_size),
        barriers=torch.zeros([nlayers, 6, nheads + nkv * 2], dtype=torch.int32, device=device),
        hidden_states_seq=torch.zeros((prompt_len, hidden_size), device=device, dtype=dtype),
        prefill_q_seq=torch.zeros((prompt_len, nheads, head_dim), device=device, dtype=dtype),
        attn_out_seq=torch.zeros((prompt_len, nheads * head_dim), device=device, dtype=dtype),
    )


@torch.no_grad()
def run_fused_prefill(
    config: InferenceConfig,
    prompts: list[str],
    *,
    loaded_model: Any = None,
    tokenizer: Any = None,
    hardware: str = "A100",
    max_new_tokens: int | None = None,
) -> PrefillOutput:
    """
    Run prefill with fused ops only (no HF forward). Prefer loading weights from disk
    (local dir or HF cache path); only load full HF model when no such path is available.
    Builds A100 globs (max_attn = device SM count, barriers 6 rows), runs prefill layers,
    returns PrefillOutput with globs so decode reuses (single copy, no double-load).
    """
    from qimeng_megakernel.models.LLM.prefill import _dtype_from_str, get_or_load_hf_model

    torch.cuda.set_device(config.device)
    device = torch.device(config.device)

    def _can_load_from_disk(model_path: str) -> bool:
        p = Path(model_path)
        return p.exists() and (p / "config.json").exists()

    def _resolve_to_disk_path(model_id: str) -> str | None:
        """If model_id is a local dir or HF id already in cache, return path with config.json; else None."""
        if _can_load_from_disk(model_id):
            return model_id
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(model_id, local_files_only=True)
        except Exception:
            return None

    disk_path = _resolve_to_disk_path(config.model)
    if tokenizer is None:
        from transformers import AutoTokenizer
        load_path = disk_path if disk_path is not None else config.model
        kw = {"local_files_only": True} if disk_path is not None else {}
        tokenizer = AutoTokenizer.from_pretrained(load_path, **kw)
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

    # Prefer disk weights (no HF model instantiation)
    if disk_path is not None and _can_load_from_disk(disk_path):
        config_dict, weights = qwen_load_weights.load_qwen3_weights_from_disk(
            disk_path, device=device, dtype=dtype
        )
    else:
        model = get_or_load_hf_model(config.model, device=config.device, torch_dtype=dtype)
        weights = _stack_qwen3_weights_from_hf(model)
        weights["embed_tokens"] = model.model.embed_tokens.weight.detach().clone()
        config_dict = {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "intermediate_size": model.config.intermediate_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
            "vocab_size": model.config.vocab_size,
            "rms_norm_eps": getattr(model.config, "rms_norm_eps", 1e-6),
            "head_dim": getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads),
        }
        del model
        torch.cuda.empty_cache()

    globs = _build_globs_for_prefill_from_weights(
        weights, config_dict, prompt_len, device, dtype, max_new_tokens=max_new_tokens
    )
    globs.embed_tokens = weights["embed_tokens"]
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
