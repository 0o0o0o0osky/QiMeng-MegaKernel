#!/usr/bin/env python3
"""Print per-layer and total instruction counts for Qwen1.5-MoE decode schedule.

Run from repo root (as module so relative imports work), e.g.:
  cd /path/to/QiMeng-MegaKernel
  PYTHONPATH="$(pwd):$PYTHONPATH" python -m qimeng_megakernel.models.LLM.qwen.qwen1_5_moe_a2_7b.A100.single_batch.decode_schedule.instructions.print_instruction_counts
"""
from __future__ import annotations

import sys

import torch

# Ensure config is loaded
from ..model_config import QwenMoeConfig
from ..globs import Globals
from .scheduler import (
    build_instruction_list,
    get_num_partials_for_device,
    _get_sm_count,
)

OPCODE_NAMES = {
    0: "NoOp",
    1: "RMS_QKVProj_GEMV_RoPE",
    2: "DecodeAttentionSplitKV",
    3: "DecodeAttentionReduction",
    4: "OProj_GEMV_Residual",
    5: "(unused)",
    6: "(unused)",
    7: "RMS_LMHead_GEMV",
    8: "Router_GEMV",
    9: "Router_SoftmaxTopK",
    10: "SharedExpert_UpGate_SiLU",
    11: "SharedExpert_DownProj",
    12: "RoutedExpert_UpGate_SiLU",
    13: "RoutedExpert_DownProj",
    14: "MoE_Combine_Commit",
}


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sm_count = _get_sm_count(device)
    num_partials = get_num_partials_for_device(device)

    # Minimal globs so build_instruction_list can read device
    dummy = torch.zeros(1, device=device)
    globs = Globals(hidden_states=dummy)

    instructions = build_instruction_list(globs, num_layers=None)

    # Count by opcode (global)
    by_opcode: dict[int, int] = {}
    # Count by (layer_idx, opcode) for layer instructions
    by_layer_op: dict[tuple[int, int], int] = {}
    lm_head_opcode = 7

    for inst in instructions:
        op = inst.__class__.opcode()
        by_opcode[op] = by_opcode.get(op, 0) + 1
        layer_idx = getattr(inst, "layer_idx", None)
        if layer_idx is not None:
            by_layer_op[(layer_idx, op)] = by_layer_op.get((layer_idx, op), 0) + 1

    num_layers = QwenMoeConfig.num_hidden_layers
    print("Qwen1.5-MoE-A2.7B decode schedule instruction counts")
    print("=" * 60)
    print(f"Device: {device}  sm_count: {sm_count}  num_partials: {num_partials}")
    print(f"Num layers: {num_layers}")
    print()

    print("Per-opcode totals (all layers + LM head):")
    print("-" * 60)
    for op in sorted(by_opcode.keys()):
        name = OPCODE_NAMES.get(op, f"op{op}")
        print(f"  Op {op:2d}  {name:35s}  {by_opcode[op]:6d}")
    print(f"  {'TOTAL':40s}  {len(instructions):6d}")
    print()

    # Per-layer breakdown (use layer 0 as template; all layers same layout)
    print("Per-layer counts (layer 0 as representative):")
    print("-" * 60)
    layer0_ops: dict[int, int] = {}
    for (li, op), count in by_layer_op.items():
        if li == 0:
            layer0_ops[op] = count
    for op in sorted(layer0_ops):
        name = OPCODE_NAMES.get(op, f"op{op}")
        count = layer0_ops.get(op, 0)
        print(f"  Op {op:2d}  {name:35s}  {count:6d}")
    layer0_total = sum(layer0_ops.values())
    print(f"  {'Per-layer subtotal':40s}  {layer0_total:6d}")
    print()

    lm_head_count = by_opcode.get(lm_head_opcode, 0)
    expected_total = layer0_total * num_layers + lm_head_count
    print("Summary:")
    print(f"  Instructions per layer (attn + MoE):  {layer0_total}")
    print(f"  LM head instructions:                  {lm_head_count}")
    print(f"  Total (24 layers + lm_head):          {expected_total}")
    print(f"  Actual list length:                   {len(instructions)}")
    if len(instructions) != expected_total:
        print("  [MISMATCH]")
        sys.exit(1)
    print("  OK")


if __name__ == "__main__":
    main()
