# A100 fused prefill for Qwen1.5-MoE-A2.7B: tokenize + fused ops only (no HF forward).
# Weights: from disk when possible; only load full HF model when no disk path available.
# KV and seq buffers allocated for prompt_len + max_new_tokens / prompt_len only.
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from qimeng_megakernel.models.LLM.prefill import InferenceConfig, PrefillOutput
from qimeng_megakernel.models.LLM.llama.preprocess import _interleave_k_cache
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.weights import (
    _load_hf_qwen_moe_model,
    _qwen_moe_weights_from_hf,
)
from qimeng_megakernel.models.LLM.qwen import load_weights as qwen_load_weights
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.preprocess import (
    _flat_expert_view,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.globs import (
    DECODE_ALLOCATED_FIELDS,
    Globals,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.model_config import (
    QwenMoeConfig,
    get_num_partials_for_device,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.fused import (
    layer_scheduler as fused_layer,
)
from qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.fused import (
    operations as fused_ops,
)


def _trace_memory(step: str, device: torch.device) -> None:
    if not torch.cuda.is_available():
        return
    try:
        dev = torch.device(device) if isinstance(device, str) else device
        alloc = torch.cuda.memory_allocated(dev) / 1024**2
        peak = torch.cuda.max_memory_allocated(dev) / 1024**2
        print(
            "[qwen1.5-moe a100 prefill memory] %s: allocated %.2f MiB  peak %.2f MiB"
            % (step, alloc, peak),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass


def _build_globs_for_prefill_from_weights(
    weights: dict[str, torch.Tensor],
    config: dict[str, Any],
    prompt_len: int,
    device: torch.device,
    dtype: torch.dtype,
    max_new_tokens: int | None = None,
) -> Globals:
    """Build A100 Globals for prefill from pre-loaded MoE weight dict. KV length = prompt_len + max_new_tokens; seq buffers length = prompt_len."""
    QwenMoeConfig.update_from_dict(config)
    device = torch.device(device) if isinstance(device, str) else device
    nlayers = config["num_hidden_layers"]
    nheads = config["num_attention_heads"]
    nkv = config.get("num_key_value_heads", nheads)
    head_dim = config.get("head_dim") or (
        config["hidden_size"] // config["num_attention_heads"]
    )
    hidden_size = config["hidden_size"]
    vocab_size = config["vocab_size"]
    num_experts = config.get("num_experts", 60)
    num_experts_per_tok = config.get("num_experts_per_tok", 4)
    shared_expert_intermediate_size = config.get(
        "shared_expert_intermediate_size",
        config.get("intermediate_size", 5632),
    )

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

    # Use same num_partials as decode scheduler so attn buffers match instruction schedule
    num_partials = get_num_partials_for_device(device)
    n_opcode_slots = QwenMoeConfig.num_opcode_rows

    def buf(shape, dt=dtype):
        if isinstance(shape, int):
            return torch.zeros(shape, device=device, dtype=dt)
        return torch.zeros(shape, device=device, dtype=dt)

    def buf32(shape):
        return torch.zeros(shape, device=device, dtype=torch.float32)

    qkv_bias = weights.get("qkv_bias")
    if qkv_bias is None:
        qkv_bias = buf((nlayers, QwenMoeConfig.qkv_out_dim))
    else:
        qkv_bias = qkv_bias.contiguous()

    expert_gate_up = weights["expert_gate_up_proj_weights"].contiguous()
    expert_down = weights["expert_down_proj_weights"].contiguous()

    globs = Globals(
        position_id=0,
        k_cache=k_cache,
        v_cache=v_cache,
        qkv_weights=weights["qkv_weights"].contiguous(),
        qkv_bias=qkv_bias,
        o_proj_weights=weights["o_proj_weights"].contiguous(),
        attn_norm_weights=weights["attn_norm_weights"].contiguous(),
        mlp_norm_weights=weights["mlp_norm_weights"].contiguous(),
        router_weights=weights["router_weights"].contiguous(),
        router_logits_buf=buf32(num_experts),
        shared_expert_gate_proj_weights=weights[
            "shared_expert_gate_proj_weights"
        ].contiguous(),
        shared_expert_up_proj_weights=weights[
            "shared_expert_up_proj_weights"
        ].contiguous(),
        shared_expert_down_proj_weights=weights[
            "shared_expert_down_proj_weights"
        ].contiguous(),
        shared_expert_gate_weights=weights["shared_expert_gate_weights"].contiguous(),
        expert_gate_up_proj_weights=expert_gate_up,
        expert_down_proj_weights=expert_down,
        expert_gate_up_proj_weights_flat=_flat_expert_view(
            expert_gate_up, "expert_gate_up_proj_weights"
        ),
        expert_down_proj_weights_flat=_flat_expert_view(
            expert_down, "expert_down_proj_weights"
        ),
        lm_head_norm_weights=weights["lm_head_norm_weights"].contiguous(),
        lm_head_weights=weights["lm_head_weights"].contiguous(),
        rope_cos=weights["rope_cos"].contiguous(),
        rope_sin=weights["rope_sin"].contiguous(),
        hidden_states=buf(hidden_size),
        q_post_rope=buf(nheads * head_dim),
        attn_out=buf(nheads * head_dim),
        attn_out_intermediates=buf32([nheads, num_partials, head_dim]),
        attn_lse_intermediates=buf32([nheads, num_partials]),
        normed_hidden_states=buf(hidden_size),
        shared_expert_silu_out=buf(shared_expert_intermediate_size),
        shared_out=buf(hidden_size),
        routed_out=buf(hidden_size),
        routed_out_f32=buf32(hidden_size),
        routed_silu_buffer=buf(
            (num_experts_per_tok, QwenMoeConfig.moe_intermediate_size)
        ),
        expert_indices=torch.zeros(
            num_experts_per_tok, device=device, dtype=torch.int64
        ),
        expert_weights=buf32(num_experts_per_tok),
        silu_out=buf(QwenMoeConfig.moe_intermediate_size),
        logits=buf(vocab_size),
        barriers=torch.zeros(
            (nlayers, n_opcode_slots, QwenMoeConfig.barrier_slots),
            dtype=torch.int32,
            device=device,
        ),
        hidden_states_seq=torch.zeros(
            (prompt_len, hidden_size), device=device, dtype=dtype
        ),
        prefill_q_seq=torch.zeros(
            (prompt_len, nheads, head_dim), device=device, dtype=torch.float32
        ),
        attn_out_seq=torch.zeros(
            (prompt_len, nheads * head_dim), device=device, dtype=torch.float32
        ),
    )
    for name in DECODE_ALLOCATED_FIELDS:
        if getattr(globs, name) is None:
            raise RuntimeError(f"fused prefill left Globals.{name} unallocated")
    return globs


@torch.no_grad()
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
    Run prefill with fused ops only (no HF forward). Prefer loading weights from disk;
    only load full HF model when no disk path available. Builds globs with KV length
    = prompt_len + max_new_tokens and seq buffers length = prompt_len.
    """
    from qimeng_megakernel.models.LLM.prefill import (
        _dtype_from_str,
        get_or_load_hf_model,
    )

    torch.cuda.set_device(config.device)
    device = torch.device(config.device)

    def _can_load_from_disk(model_path: str) -> bool:
        p = Path(model_path)
        return p.exists() and (p / "config.json").exists()

    def _resolve_to_disk_path(model_id: str) -> str | None:
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

    if trace_memory:
        _trace_memory("after tokenize", device)

    batch_size, prompt_len = input_ids.shape[0], input_ids.shape[1]
    if batch_size != 1:
        raise NotImplementedError("fused prefill only supports batch size 1")

    dtype = _dtype_from_str(config.torch_dtype)
    if dtype == "auto":
        dtype = torch.bfloat16

    if disk_path is not None and _can_load_from_disk(disk_path):
        config_dict, weights = qwen_load_weights.load_qwen_moe_weights_from_disk(
            disk_path, device=device, dtype=dtype
        )
    else:
        model = get_or_load_hf_model(
            config.model, device=config.device, torch_dtype=dtype
        )
        weights = _qwen_moe_weights_from_hf(model)
        weights["embed_tokens"] = model.model.embed_tokens.weight.detach().clone()
        config_dict = {
            "num_hidden_layers": model.config.num_hidden_layers,
            "hidden_size": model.config.hidden_size,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": getattr(
                model.config,
                "num_key_value_heads",
                model.config.num_attention_heads,
            ),
            "vocab_size": model.config.vocab_size,
            "rms_norm_eps": getattr(model.config, "rms_norm_eps", 1e-6),
            "head_dim": getattr(
                model.config,
                "head_dim",
                model.config.hidden_size // model.config.num_attention_heads,
            ),
            "num_experts": getattr(model.config, "num_experts", 60),
            "num_experts_per_tok": getattr(model.config, "num_experts_per_tok", 4),
            "norm_topk_prob": getattr(model.config, "norm_topk_prob", False),
            "shared_expert_intermediate_size": getattr(
                model.config,
                "shared_expert_intermediate_size",
                getattr(model.config, "intermediate_size", 5632),
            ),
            "moe_intermediate_size": getattr(
                model.config, "moe_intermediate_size", 1408
            ),
        }
        del model
        torch.cuda.empty_cache()

    if trace_memory:
        _trace_memory(
            "after load model + stack weights + del model + empty_cache", device
        )

    globs = _build_globs_for_prefill_from_weights(
        weights, config_dict, prompt_len, device, dtype, max_new_tokens=max_new_tokens
    )
    globs.embed_tokens = weights["embed_tokens"]
    globs.hidden_states_seq.copy_(
        weights["embed_tokens"][input_ids.squeeze(0)].to(
            globs.hidden_states_seq.dtype
        )
    )

    if trace_memory:
        _trace_memory("after build globs + embed", device)

    for layer_idx in range(QwenMoeConfig.num_hidden_layers):
        fused_layer.run_fused_prefill_layer(globs, layer_idx, prompt_len)

    globs.hidden_states.copy_(
        globs.hidden_states_seq[prompt_len - 1].to(globs.hidden_states.dtype)
    )
    fused_ops.fused_rms_lmhead_gemv(globs)

    if trace_memory:
        _trace_memory("after prefill layers + lm_head", device)

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
