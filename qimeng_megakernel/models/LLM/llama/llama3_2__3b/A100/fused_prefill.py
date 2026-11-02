# A100 (Ampere) fused prefill for Llama 3.2 3B: tokenize + our fused ops only (no HF forward).
# One forward per layer over the full sequence, then lm_head on the last position only.
# Weights/tokenizer: from disk when possible (local dir or HF cache path); only load full HF when no disk path.
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.prefill import InferenceConfig, PrefillOutput
from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache, _stack_llama_weights_from_hf
from qimeng_megakernel.models.LLM.llama import load_weights as llama_load_weights
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.A100.single_batch.decode_schedule.globs import (
    Globals as SingleBatchGlobals,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.A100.single_batch.decode_schedule.model_config import LlamaConfig
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.A100.single_batch.decode_schedule.fused import (
    layer_scheduler as single_fused_layer,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__3b.A100.single_batch.decode_schedule.fused import (
    operations as single_fused_ops,
)


def _get_sm_count(device):
    if getattr(device, "type", None) != "cuda":
        return 1
    try:
        return torch.cuda.get_device_properties(device).multi_processor_count
    except Exception:
        return 1


def _trace_memory(phase: str, device: torch.device) -> None:
    """Print GPU allocated/peak when LLAMA_TRACE_MEMORY=1 (debug OOM)."""
    if os.environ.get("LLAMA_TRACE_MEMORY", "").strip() not in ("1", "true", "yes"):
        return
    if not torch.cuda.is_available() or device.type != "cuda":
        return
    try:
        torch.cuda.synchronize(device)
        alloc = torch.cuda.memory_allocated(device) / 1024**2
        peak = torch.cuda.max_memory_allocated(device) / 1024**2
        total = torch.cuda.get_device_properties(device).total_memory / 1024**2
        print(
            f"[3B prefill memory] {phase}: allocated={alloc:.2f} MiB  peak={peak:.2f} MiB  total={total:.2f} MiB",
            flush=True,
        )
    except Exception as e:
        print(f"[3B prefill memory] {phase}: {e}", flush=True)


def _build_globs_for_prefill_from_weights(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    prompt_len: int,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int | None = None,
    *,
    batch_size: int = 1,
) -> SingleBatchGlobals:
    """Build single-batch prefill globals from disk/HF weights."""
    if batch_size != 1:
        raise NotImplementedError("Llama 3.2 3B A100 supports single-batch decode only")
    LlamaConfig.update_from_dict(config)
    nlayers = config["num_hidden_layers"]
    nheads = config["num_attention_heads"]
    nkv = config.get("num_key_value_heads", nheads)
    head_dim = config.get("head_dim") or (config["hidden_size"] // nheads)
    hidden_size = config["hidden_size"]
    intermediate_size = config["intermediate_size"]
    vocab_size = config["vocab_size"]

    kv_seq_len = prompt_len + (max_new_tokens or 0)
    k_cache = torch.zeros(
        (nlayers, batch_size, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=dtype,
    )
    k_cache = _interleave_k_cache(k_cache, head_dim).contiguous()
    v_cache = torch.zeros(
        (nlayers, batch_size, kv_seq_len, nkv, head_dim),
        device=device,
        dtype=dtype,
    ).contiguous()

    max_attn = _get_sm_count(device)

    def buf(s, dt=dtype):
        return torch.zeros(s, device=device, dtype=dt)

    def buf32(s):
        return torch.zeros(s, device=device, dtype=torch.float32)

    hidden_states = buf(hidden_size)
    q_post_rope = buf(hidden_size)
    attn_out = buf(hidden_size)
    silu_out = buf(intermediate_size)
    logits = buf(vocab_size)
    attn_int = buf32([nheads, max_attn, head_dim])
    attn_lse = buf32([nheads, max_attn])
    hidden_states_seq = torch.zeros((prompt_len, hidden_size), device=device, dtype=dtype)
    prefill_q_seq = torch.zeros((prompt_len, nheads, head_dim), device=device, dtype=dtype)
    attn_out_seq = torch.zeros((prompt_len, nheads * head_dim), device=device, dtype=dtype)

    globs = SingleBatchGlobals(
        position_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
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
        hidden_states=hidden_states,
        q_post_rope=q_post_rope,
        attn_out=attn_out,
        attn_out_intermediates=attn_int,
        attn_lse_intermediates=attn_lse,
        silu_out=silu_out,
        logits=logits,
        barriers=torch.zeros(
            [nlayers, 10, nheads + nkv * 2], dtype=torch.int32, device=device
        ),
        hidden_states_seq=hidden_states_seq,
        prefill_q_seq=prefill_q_seq,
        attn_out_seq=attn_out_seq,
    )
    globs.embed_tokens = weights["embed_tokens"]
    return globs


def _can_load_from_disk(model_path: str) -> bool:
    p = Path(model_path)
    return p.exists() and (p / "config.json").exists()


def _resolve_hf_id_to_cache_path(model_id: str):
    """If model_id is an HF id and the model is already in the HF cache, return the snapshot path. Otherwise None."""
    if _can_load_from_disk(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(model_id, local_files_only=True)
    except Exception:
        return None


_HF_LOAD_FAIL_HINT = (
    "Use a local model path: --model /path/to/model (directory with config.json and safetensors). "
    "Add a mapping in qimeng_megakernel/models/LLM/registry.local.json if needed."
)


@torch.inference_mode()
def run_fused_prefill(
    config: InferenceConfig,
    prompts: list[str],
    *,
    loaded_model: Any = None,
    tokenizer: Any = None,
    hardware: str = "A100",
    max_new_tokens: int | None = None,
    trace_memory: bool = False,
) -> PrefillOutput:
    """
    Run prefill with fused ops only: tokenize -> embed -> one forward per layer over
    full sequence -> lm_head on last position.
    No HF model. Returns PrefillOutput with globs set so decode can reuse.
    Weights: from disk (local path or HF cache) or from HF when not cached.
    """
    from qimeng_megakernel.models.LLM.prefill import _dtype_from_str

    torch.cuda.set_device(config.device)
    device = torch.device(config.device)

    # Resolve once: use disk path for both tokenizer and weights when available
    model_path = config.model
    if not _can_load_from_disk(model_path):
        model_path = _resolve_hf_id_to_cache_path(config.model)
    disk_path = model_path if (model_path and _can_load_from_disk(model_path)) else None

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
    prompt_lens = attention_mask.sum(dim=-1).to(device=device, dtype=torch.long)
    if batch_size != 1:
        raise NotImplementedError("Llama 3.2 3B A100 supports single-batch decode only")

    dtype = _dtype_from_str(config.torch_dtype)
    if dtype == "auto":
        dtype = torch.bfloat16
    if disk_path is not None:
        config_dict, weights = llama_load_weights.load_llama_weights_from_disk(
            disk_path, device=device, dtype=dtype
        )
        _trace_memory("after load_weights", device)
    else:
        from qimeng_megakernel.models.LLM.prefill import get_or_load_hf_model
        try:
            model = get_or_load_hf_model(config.model, device=config.device, torch_dtype=dtype)
        except Exception as e:
            raise ValueError(
                "Failed to load model '%s' from HuggingFace (network unreachable, gated model without token, or not found): %s. %s"
                % (config.model, e, _HF_LOAD_FAIL_HINT)
            ) from e
        weights = _stack_llama_weights_from_hf(model)
        weights["embed_tokens"] = model.model.embed_tokens.weight.detach().clone()
        config_dict = {k: getattr(model.config, k, None) for k in (
            "num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads",
            "num_key_value_heads", "vocab_size", "rms_norm_eps", "head_dim"
        )}
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        if "head_dim" not in config_dict:
            config_dict["head_dim"] = config_dict["hidden_size"] // config_dict["num_attention_heads"]
        del model
        torch.cuda.empty_cache()
        _trace_memory("after load_weights (from HF)", device)
    globs = _build_globs_for_prefill_from_weights(
        weights,
        config_dict,
        prompt_len,
        device,
        dtype,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    embed = weights["embed_tokens"][input_ids].to(globs.hidden_states_seq.dtype)
    globs.hidden_states_seq.copy_(embed.squeeze(0))
    fused_layer = single_fused_layer
    fused_ops = single_fused_ops
    for layer_idx in range(LlamaConfig.num_hidden_layers):
        fused_layer.run_fused_prefill_layer(globs, layer_idx, prompt_len)

    globs.hidden_states.copy_(
        globs.hidden_states_seq[int(prompt_lens[0].item()) - 1].to(globs.hidden_states.dtype)
    )
    fused_ops.fused_rms_lmhead_gemv(globs)

    logits = globs.logits.unsqueeze(0).unsqueeze(0)
    first_gen_token = torch.argmax(globs.logits, dim=-1).unsqueeze(0).unsqueeze(-1)
    pos_id_token_1 = prompt_lens.to(dtype=torch.long).unsqueeze(1)
    kv_seq_len_before = prompt_lens.to(dtype=torch.long)
    kv_seq_len_after = kv_seq_len_before + 1
    globs.position_id = int(prompt_lens.max().item())

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
