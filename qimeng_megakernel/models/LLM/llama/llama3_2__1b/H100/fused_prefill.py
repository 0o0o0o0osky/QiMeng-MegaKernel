# H100 (Hopper) fused prefill: tokenize + our fused ops only (no HF forward).
# One forward per layer over the full sequence, then lm_head on the last position only.
# Weights/tokenizer: from disk when possible (local dir or HF cache path); only load full HF when no disk path.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.llama import load_weights as llama_load_weights
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule.globs import (
    BATCH_CAPACITY,
    MAX_BATCH_SIZE,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule.globs import (
    Globals as MultiBatchGlobals,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.fused import (
    layer_scheduler as fused_layer,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.fused import (
    operations as fused_ops,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.globs import (
    Globals as SingleBatchGlobals,
)
from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.single_batch.decode_schedule.model_config import (
    LlamaConfig,
)
from qimeng_megakernel.models.LLM.llama.preprocess import (
    _interleave_k_cache,
    _stack_llama_weights_from_hf,
)
from qimeng_megakernel.models.LLM.prefill import InferenceConfig, PrefillOutput


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
    *,
    batch_size: int = 1,
) -> SingleBatchGlobals | MultiBatchGlobals:
    """Build Globals from pre-loaded weight dict.

    Normal inference uses SingleBatchGlobals for B=1 and MultiBatchGlobals for B>1.
    """
    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise NotImplementedError(
            f"H100 fused prefill supports batch_size in [1, {MAX_BATCH_SIZE}], got {batch_size}"
        )
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

    max_attn = min(_get_sm_count(device), 132)

    def buf(s, dt=dtype):
        return torch.zeros(s, device=device, dtype=dt)

    def buf32(s):
        return torch.zeros(s, device=device, dtype=torch.float32)

    attn_partial_valid: torch.Tensor | None = None
    GlobalsCls = SingleBatchGlobals
    globs_kw: dict[str, Any] = {}
    if batch_size == 1:
        hidden_states = buf(hidden_size)
        q_post_rope = buf(hidden_size)
        attn_out = buf(hidden_size)
        silu_out = buf(intermediate_size)
        logits = buf(vocab_size)
        attn_int = buf32([nheads, max_attn, head_dim])
        attn_lse = buf32([nheads, max_attn])
        hidden_states_seq = torch.zeros(
            (prompt_len, hidden_size), device=device, dtype=dtype
        )
        prefill_q_seq = torch.zeros(
            (prompt_len, nheads, head_dim), device=device, dtype=dtype
        )
        attn_out_seq = torch.zeros(
            (prompt_len, nheads * head_dim), device=device, dtype=dtype
        )
    else:
        hidden_states_backing = buf((BATCH_CAPACITY, hidden_size))
        q_post_rope_backing = buf((BATCH_CAPACITY, hidden_size))
        attn_out_backing = buf((BATCH_CAPACITY, hidden_size))
        silu_out_backing = buf((BATCH_CAPACITY, intermediate_size))
        logits_backing = buf((BATCH_CAPACITY, vocab_size))
        attn_rms_out_backing = buf((BATCH_CAPACITY, hidden_size))
        mlp_rms_out_backing = buf((BATCH_CAPACITY, hidden_size))
        lm_head_rms_out_backing = buf((BATCH_CAPACITY, hidden_size))
        down_partials = buf32((4, BATCH_CAPACITY, hidden_size))
        mk_barriers = torch.zeros(
            (nlayers, 10, 48), device=device, dtype=torch.int32
        )
        k_cache_decode = torch.zeros(
            (nlayers, BATCH_CAPACITY, nkv, kv_seq_len, head_dim),
            device=device,
            dtype=dtype,
        )
        v_cache_decode = torch.zeros_like(k_cache_decode)
        hidden_states = hidden_states_backing[:batch_size]
        q_post_rope = q_post_rope_backing[:batch_size]
        attn_out = attn_out_backing[:batch_size]
        silu_out = silu_out_backing[:batch_size]
        logits = logits_backing[:batch_size]
        attn_int = buf32([batch_size, nheads, max_attn, head_dim])
        attn_lse = buf32([batch_size, nheads, max_attn])
        attn_partial_valid = torch.zeros(
            (batch_size, nheads, max_attn), device=device, dtype=torch.bool
        )
        attn_rms_out = attn_rms_out_backing[:batch_size]
        mlp_rms_out = mlp_rms_out_backing[:batch_size]
        lm_head_rms_out = lm_head_rms_out_backing[:batch_size]
        hidden_states_seq = torch.zeros(
            (batch_size, prompt_len, hidden_size), device=device, dtype=dtype
        )
        prefill_q_seq = torch.zeros(
            (batch_size, prompt_len, nheads, head_dim), device=device, dtype=dtype
        )
        attn_out_seq = torch.zeros(
            (batch_size, prompt_len, nheads * head_dim), device=device, dtype=dtype
        )
        GlobalsCls = MultiBatchGlobals
        globs_kw = {
            "attn_partial_valid": attn_partial_valid,
            "attn_rms_out": attn_rms_out,
            "mlp_rms_out": mlp_rms_out,
            "lm_head_rms_out": lm_head_rms_out,
            "logical_batch_size": batch_size,
            "batch_capacity": BATCH_CAPACITY,
            "hidden_states_backing": hidden_states_backing,
            "q_post_rope_backing": q_post_rope_backing,
            "attn_out_backing": attn_out_backing,
            "silu_out_backing": silu_out_backing,
            "logits_backing": logits_backing,
            "attn_rms_out_backing": attn_rms_out_backing,
            "mlp_rms_out_backing": mlp_rms_out_backing,
            "lm_head_rms_out_backing": lm_head_rms_out_backing,
            "k_cache_decode": k_cache_decode,
            "v_cache_decode": v_cache_decode,
            "down_partials": down_partials,
            "mk_barriers": mk_barriers,
        }

    globs = GlobalsCls(
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
            [nlayers, 10, (nheads + nkv * 2) * batch_size],
            dtype=torch.int32,
            device=device,
        ),
        hidden_states_seq=hidden_states_seq,
        prefill_q_seq=prefill_q_seq,
        attn_out_seq=attn_out_seq,
        **globs_kw,
    )
    globs.embed_tokens = weights["embed_tokens"]
    if batch_size > 1:
        from qimeng_megakernel.models.LLM.llama.llama3_2__1b.H100.multi_batch.decode_schedule.instructions import (
            scheduler as mb_scheduler,
        )

        mb_scheduler.tensorize_instructions(
            globs,
            mb_scheduler.build_instruction_queues(
                globs, batch_size=batch_size
            ),
        )
        globs.validate_padded_storage()
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


def _trace_memory(enabled: bool, step: str, device: torch.device) -> None:
    """Print allocated and peak GPU memory when explicitly requested."""
    if not enabled or not torch.cuda.is_available():
        return
    dev = torch.device(device) if isinstance(device, str) else device
    alloc = torch.cuda.memory_allocated(dev) / 1024**2
    peak = torch.cuda.max_memory_allocated(dev) / 1024**2
    print(
        f"[fused_prefill memory] {step}: allocated {alloc:.2f} MiB  "
        f"peak {peak:.2f} MiB",
        file=sys.stderr,
        flush=True,
    )


@torch.inference_mode()
def run_fused_prefill(
    config: InferenceConfig,
    prompts: list[str],
    *,
    loaded_model: Any = None,
    tokenizer: Any = None,
    hardware: str = "H100",
    max_new_tokens: int | None = None,
    trace_memory: bool = False,
) -> PrefillOutput:
    """
    Run prefill with fused ops only: tokenize -> embed -> one forward per layer over
    full sequence (one causal SDPA per layer) -> lm_head on last position.
    No HF model.forward(). Returns PrefillOutput with globs set so decode can reuse.
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
    _trace_memory(trace_memory, "after tokenize", device)

    batch_size, prompt_len = input_ids.shape[0], input_ids.shape[1]
    if batch_size < 1 or batch_size > MAX_BATCH_SIZE:
        raise NotImplementedError(
            f"H100 fused prefill supports batch_size in [1, {MAX_BATCH_SIZE}], got {batch_size}"
        )
    prompt_lens = attention_mask.sum(dim=-1)

    dtype = _dtype_from_str(config.torch_dtype)
    if dtype == "auto":
        dtype = torch.bfloat16
    if disk_path is not None:
        config_dict, weights = llama_load_weights.load_llama_weights_from_disk(
            disk_path, device=device, dtype=dtype
        )
        _trace_memory(trace_memory, "after loading weights", device)
    else:
        from qimeng_megakernel.models.LLM.prefill import get_or_load_hf_model

        try:
            model = get_or_load_hf_model(
                config.model, device=config.device, torch_dtype=dtype
            )
        except Exception as e:
            raise ValueError(
                "Failed to load model '%s' from HuggingFace (network unreachable, gated model without token, or not found): %s. %s"
                % (config.model, e, _HF_LOAD_FAIL_HINT)
            ) from e
        weights = _stack_llama_weights_from_hf(model)
        weights["embed_tokens"] = model.model.embed_tokens.weight.detach().clone()
        config_dict = {
            k: getattr(model.config, k, None)
            for k in (
                "num_hidden_layers",
                "hidden_size",
                "intermediate_size",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
                "rms_norm_eps",
                "head_dim",
            )
        }
        config_dict = {k: v for k, v in config_dict.items() if v is not None}
        if "head_dim" not in config_dict:
            config_dict["head_dim"] = (
                config_dict["hidden_size"] // config_dict["num_attention_heads"]
            )
        del model
        torch.cuda.empty_cache()
        _trace_memory(trace_memory, "after releasing HF model", device)
    globs = _build_globs_for_prefill_from_weights(
        weights,
        config_dict,
        prompt_len,
        device,
        dtype,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
    )
    _trace_memory(trace_memory, "after building prefill state", device)
    globs.embed_tokens = weights["embed_tokens"]
    emb = weights["embed_tokens"][input_ids].to(globs.hidden_states_seq.dtype)
    if (
        batch_size == 1
        and globs.hidden_states_seq.ndim == 2
        and globs.hidden_states.ndim == 1
    ):
        globs.hidden_states_seq.copy_(emb.squeeze(0))
    else:
        globs.hidden_states_seq.copy_(emb)

    # One forward per layer over the full sequence
    for layer_idx in range(LlamaConfig.num_hidden_layers):
        fused_layer.run_fused_prefill_layer(globs, layer_idx, prompt_len)

    # Last position -> hidden_states, then lm_head -> logits
    if batch_size == 1 and globs.hidden_states.ndim == 1:
        globs.hidden_states.copy_(
            globs.hidden_states_seq[prompt_len - 1].to(globs.hidden_states.dtype)
        )
    else:
        row_ids = torch.arange(batch_size, device=device)
        last_real = (prompt_lens.to(device=device, dtype=torch.long) - 1).clamp_min_(0)
        globs.hidden_states.copy_(
            globs.hidden_states_seq[row_ids, last_real].to(globs.hidden_states.dtype)
        )
    fused_ops.fused_rms_lmhead_gemv(globs)

    # logits from last step are in globs.logits
    if batch_size == 1 and globs.logits.ndim == 1:
        logits = globs.logits.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size]
    else:
        logits = globs.logits.unsqueeze(1)  # [B, 1, vocab_size]
    pl = prompt_lens.to(device=device, dtype=torch.long)
    is_multi_batch_globs = globs.hidden_states.ndim == 2
    if is_multi_batch_globs:
        globs.k_cache_decode[:, :batch_size].copy_(
            globs.k_cache.permute(0, 1, 3, 2, 4)
        )
        globs.v_cache_decode[:, :batch_size].copy_(
            globs.v_cache.permute(0, 1, 3, 2, 4)
        )
        globs.position_ids = pl.clone()
        globs.kv_lengths = globs.position_ids + 1
        globs.active_mask = torch.ones(
            batch_size, dtype=torch.bool, device=device
        )
        globs.eos_reached = torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
        globs.last_generated_tokens = torch.zeros(
            batch_size, dtype=torch.long, device=device
        )
    globs.position_id = int(pl.max().item())
    if (
        is_multi_batch_globs
        and tokenizer is not None
        and getattr(tokenizer, "eos_token_id", None) is not None
    ):
        globs.eos_token_id = int(tokenizer.eos_token_id)

    if batch_size == 1 and globs.logits.ndim == 1:
        first_gen_token = torch.argmax(globs.logits, dim=-1).unsqueeze(0).unsqueeze(-1)
    else:
        first_gen_token = torch.argmax(globs.logits, dim=-1).unsqueeze(-1)  # [B, 1]
    pos_id_token_1 = prompt_lens.to(dtype=torch.long).unsqueeze(1)
    kv_seq_len_before = prompt_lens.to(dtype=torch.long)
    kv_seq_len_after = kv_seq_len_before + 1

    # past_key_values: decode preprocess expects this when not using prefill.globs; we pass (k_cache, v_cache) for compatibility
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
