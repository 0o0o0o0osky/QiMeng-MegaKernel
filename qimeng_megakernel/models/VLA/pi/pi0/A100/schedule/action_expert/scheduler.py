from __future__ import annotations

import os

import torch

from .config import (
    ACTION_DIM,
    ATTN_IN_DIM,
    DECODER_WIDTH,
    HEAD_DIM,
    K_SPLITS_DOWN,
    MLP_WIDTH,
    NUM_Q_HEADS,
    QKV_OUT_DIM,
    Pi0ActionExpertGlobals,
)
from .instructions import (
    BR,
    DINST_NO_BARRIER,
    DOP,
    SPLITK_REDUCE_PHASE,
    DecoderInstruction,
    decoder_barrier_byte_offset,
    serialize_instructions,
)

SM_COUNT = int(os.environ.get("PI0_AE_SM_COUNT", "108"))
BN = 64
NO_DEP = (DINST_NO_BARRIER, 0)


def num_n_tiles(extent: int, tile: int) -> int:
    if extent <= 0 or tile <= 0:
        return 0
    return (extent + tile - 1) // tile


class _Builder:
    def __init__(self, g: Pi0ActionExpertGlobals, sm_count: int):
        self.g = g
        self.sm = sm_count
        self.nls = g.num_layer_slots
        self.insts: list[DecoderInstruction] = []

    def off(self, fs: int, layer_slot: int, row: int, slot: int = 0) -> int:
        return decoder_barrier_byte_offset(fs, layer_slot, row, slot, self.nls)

    def emit_gridstride(self, opcode: int, fs: int, layer: int, dep: tuple[int, int], signal_row: int, *, layer_slot: int | None = None, k_split_idx: int = 0):
        ls = layer if layer_slot is None else layer_slot
        dep_off, dep_cnt = dep
        out_off = self.off(fs, ls, signal_row) if signal_row is not None else DINST_NO_BARRIER
        for _ in range(self.sm):
            self.insts.append(
                DecoderInstruction(
                    opcode=opcode,
                    layer_idx=layer,
                    flow_step=fs,
                    dep_counter_offset=dep_off,
                    dep_expected_count=dep_cnt,
                    out_counter_offset=out_off,
                    k_split_idx=k_split_idx,
                )
            )
        return self.sm

    def emit_gemm(self, opcode: int, fs: int, layer: int, M: int, N: int, K: int, dep: tuple[int, int], signal_row: int, *, layer_slot: int | None = None):
        ls = layer if layer_slot is None else layer_slot
        dep_off, dep_cnt = dep
        out_off = self.off(fs, ls, signal_row)
        nn = num_n_tiles(N, BN)
        for nt in range(nn):
            self.insts.append(
                DecoderInstruction(
                    opcode=opcode,
                    tile_m_start=0,
                    tile_m_end=1,
                    tile_n_start=nt,
                    tile_n_end=nt + 1,
                    M=M,
                    N=N,
                    K=K,
                    layer_idx=layer,
                    flow_step=fs,
                    dep_counter_offset=dep_off,
                    dep_expected_count=dep_cnt,
                    out_counter_offset=out_off,
                )
            )
        return nn

    def emit_gemm_2d(self, opcode: int, fs: int, layer: int, M: int, N: int, K: int, dep: tuple[int, int], signal_row: int, *, bm: int = 64, bn: int = BN, layer_slot: int | None = None):
        """Multi-M-tile GEMM (M > BM): emit ceil(M/bm) × ceil(N/bn) tiles."""
        ls = layer if layer_slot is None else layer_slot
        dep_off, dep_cnt = dep
        out_off = self.off(fs, ls, signal_row)
        nn = num_n_tiles(N, bn)
        nm = num_n_tiles(M, bm)
        for mt in range(nm):
            for nt in range(nn):
                self.insts.append(
                    DecoderInstruction(
                        opcode=opcode,
                        tile_m_start=mt,
                        tile_m_end=mt + 1,
                        tile_n_start=nt,
                        tile_n_end=nt + 1,
                        M=M, N=N, K=K,
                        layer_idx=layer,
                        flow_step=fs,
                        dep_counter_offset=dep_off,
                        dep_expected_count=dep_cnt,
                        out_counter_offset=out_off,
                    )
                )
        return nm * nn

    # Materialized-attention band geometry shared by scheduling decisions.
    QK_BM = 64
    PV_BM = 32
    ROWS_PER_SM_INST = 4

    def _r8(self):
        return self.g.rows_total * NUM_Q_HEADS            # 408

    def _num_qk_bands(self):
        return (self._r8() + self.QK_BM - 1) // self.QK_BM  # 7

    def _band_rows(self, b: int) -> tuple[int, int]:
        r0 = b * self.QK_BM
        return r0, min(r0 + self.QK_BM, self._r8())       # band 6 spans 24 rows: [384, 408)

    def _sm_insts_per_band(self, b: int) -> int:
        r0, r1 = self._band_rows(b)
        return (r1 - r0 + self.ROWS_PER_SM_INST - 1) // self.ROWS_PER_SM_INST

    def emit_attn(self, fs: int, layer: int, dep: tuple[int, int]):
        """Emit QK, row-softmax, and PV work with per-band dependencies."""
        R8 = self._r8()
        kvs = self.g.kv_stride
        nn_qk = num_n_tiles(kvs, self.QK_BM)               # 14 @ kv_stride=896
        nb = self._num_qk_bands()                          # 7
        dep_off, dep_cnt = dep

        # Keep the flat queue in topological order: all QK bands, row softmax,
        # then PV. Per-band counters still let ready work proceed independently.
        for mt in range(nb):
            out_off = self.off(fs, layer, BR.ATTN_QK_DONE, slot=mt)
            for nt in range(nn_qk):
                self.insts.append(
                    DecoderInstruction(
                        opcode=DOP.ATTN_QK,
                        tile_m_start=mt, tile_m_end=mt + 1,
                        tile_n_start=nt, tile_n_end=nt + 1,
                        M=R8, N=kvs, K=HEAD_DIM,
                        layer_idx=layer, flow_step=fs,
                        dep_counter_offset=dep_off,
                        dep_expected_count=dep_cnt,
                        out_counter_offset=out_off,
                    )
                )

        # Softmax instructions carry absolute packed-row ranges.
        for b in range(nb):
            r0, r1 = self._band_rows(b)
            dep_off_b = self.off(fs, layer, BR.ATTN_QK_DONE, slot=b)
            out_off_b = self.off(fs, layer, BR.ATTN_SM_DONE, slot=b)
            for r in range(r0, r1, self.ROWS_PER_SM_INST):
                self.insts.append(
                    DecoderInstruction(
                        opcode=DOP.ATTN_SOFTMAX,
                        tile_m_start=r,
                        tile_m_end=min(r + self.ROWS_PER_SM_INST, r1),
                        M=R8, layer_idx=layer, flow_step=fs,
                        dep_counter_offset=dep_off_b,
                        dep_expected_count=nn_qk,
                        out_counter_offset=out_off_b,
                    )
                )

        # Each PV tile waits only for its softmax band and publishes to one
        # aggregate attention-complete counter.
        n_pv = 0
        out_off = self.off(fs, layer, BR.ATTN_DONE)
        nm_pv = (R8 + self.PV_BM - 1) // self.PV_BM        # 13
        nn_pv = num_n_tiles(HEAD_DIM, 32)                  # 8
        for mt32 in range(nm_pv):
            b = (mt32 * self.PV_BM) // self.QK_BM          # Map 32-row PV tiles to 64-row QK bands.
            dep_off_b = self.off(fs, layer, BR.ATTN_SM_DONE, slot=b)
            exp_b = self._sm_insts_per_band(b)
            for nt in range(nn_pv):
                self.insts.append(
                    DecoderInstruction(
                        opcode=DOP.ATTN_PV,
                        tile_m_start=mt32, tile_m_end=mt32 + 1,
                        tile_n_start=nt, tile_n_end=nt + 1,
                        M=R8, N=HEAD_DIM, K=kvs,
                        layer_idx=layer, flow_step=fs,
                        dep_counter_offset=dep_off_b,
                        dep_expected_count=exp_b,
                        out_counter_offset=out_off,
                    )
                )
                n_pv += 1
        return n_pv

    def emit_splitk(self, opcode: int, fs: int, layer: int, M: int, N: int, K: int, k_splits: int, dep: tuple[int, int], partial_row: int, done_row: int, *, bn: int = BN):
        dep_off, dep_cnt = dep
        partial_off = self.off(fs, layer, partial_row)
        done_off = self.off(fs, layer, done_row)
        nn = num_n_tiles(N, bn)
        for nt in range(nn):
            for ks in range(k_splits):
                self.insts.append(
                    DecoderInstruction(
                        opcode=opcode,
                        tile_m_start=0,
                        tile_m_end=1,
                        tile_n_start=nt,
                        tile_n_end=nt + 1,
                        M=M,
                        N=N,
                        K=K,
                        layer_idx=layer,
                        flow_step=fs,
                        k_split_idx=ks,
                        k_splits=k_splits,
                        dep_counter_offset=dep_off,
                        dep_expected_count=dep_cnt,
                        out_counter_offset=partial_off,
                    )
                )
        n_partial = nn * k_splits
        for _ in range(self.sm):
            self.insts.append(
                DecoderInstruction(
                    opcode=opcode,
                    M=M,
                    N=N,
                    K=K,
                    layer_idx=layer,
                    flow_step=fs,
                    k_split_idx=SPLITK_REDUCE_PHASE,
                    k_splits=k_splits,
                    dep_counter_offset=partial_off,
                    dep_expected_count=n_partial,
                    out_counter_offset=done_off,
                )
            )
        return self.sm


def build_instruction_list(g: Pi0ActionExpertGlobals, sm_count: int = SM_COUNT) -> list[DecoderInstruction]:
    b = _Builder(g, sm_count)
    L = g.num_layers
    F = g.flow_steps
    R = g.rows_total
    C = g.chunk_size

    state_n = b.emit_gemm(DOP.STATE_IN, 0, 0, 1, DECODER_WIDTH, ACTION_DIM, NO_DEP, BR.STATE_IN_DONE, layer_slot=L)
    action_out_n = num_n_tiles(ACTION_DIM, BN)

    for fs in range(F):
        if fs == 0:
            dep = (b.off(0, L, BR.STATE_IN_DONE), state_n)
        else:
            dep = (b.off(fs - 1, L, BR.ACTION_OUT_DONE), action_out_n)
        n_action_in = b.emit_gemm(DOP.ACTION_IN, fs, 0, C, DECODER_WIDTH, ACTION_DIM, dep, BR.ACTION_IN_DONE, layer_slot=L)
        n_action_mlp = b.emit_gemm(
            DOP.ACTION_MLP,
            fs,
            0,
            C,
            DECODER_WIDTH,
            DECODER_WIDTH,
            (b.off(fs, L, BR.ACTION_IN_DONE), n_action_in),
            BR.ACTION_MLP_DONE,
            layer_slot=L,
        )

        for layer in range(L):
            if layer == 0:
                dep = (b.off(fs, L, BR.ACTION_MLP_DONE), n_action_mlp)
            else:
                dep = (b.off(fs, layer - 1, BR.DOWN_RESIDUAL_DONE), sm_count)
            # This route selects the row-scale producer/consumer slots around
            # QKV; the generic GEMM skeleton does not prescribe that placement.
            n_qkv = b.emit_gemm_2d(
                DOP.QKV_ROPE_KV,
                fs,
                layer,
                R,
                QKV_OUT_DIM,
                DECODER_WIDTH,
                dep,
                BR.QKV_ROPE_KV_DONE,
                bm=64,
                bn=32,
            )
            n_attn = b.emit_attn(fs, layer, (b.off(fs, layer, BR.QKV_ROPE_KV_DONE), n_qkv))
            # OProj uses a single-B GEMM; Down below uses split-K.
            n_oproj = b.emit_gemm_2d(
                DOP.OPROJ_RESIDUAL,
                fs,
                layer,
                R,
                DECODER_WIDTH,
                ATTN_IN_DIM,
                (b.off(fs, layer, BR.ATTN_DONE), n_attn),
                BR.OPROJ_RESIDUAL_DONE,
                bm=32, bn=32,
            )
            # Gate/Up selects the same-A dual-B skeleton and consumes the
            # row-scale slot before its nonlinear merge.
            n_gate = b.emit_gemm(
                DOP.GATE_UP_DUAL,
                fs,
                layer,
                R,
                MLP_WIDTH,
                DECODER_WIDTH,
                (b.off(fs, layer, BR.OPROJ_RESIDUAL_DONE), n_oproj),
                BR.GATE_UP_DUAL_DONE,
            )
            b.emit_splitk(
                DOP.DOWN_RESIDUAL,
                fs,
                layer,
                R,
                DECODER_WIDTH,
                MLP_WIDTH,
                K_SPLITS_DOWN,
                (b.off(fs, layer, BR.GATE_UP_DUAL_DONE), n_gate),
                BR.DOWN_PARTIAL_DONE,
                BR.DOWN_RESIDUAL_DONE,
            )

        # ActionOut consumes the final row-scale slot for action rows only.
        b.emit_gemm(
            DOP.ACTION_OUT,
            fs,
            L - 1,
            C,
            ACTION_DIM,
            DECODER_WIDTH,
            (b.off(fs, L - 1, BR.DOWN_RESIDUAL_DONE), sm_count),
            BR.ACTION_OUT_DONE,
            layer_slot=L,
        )
    return b.insts


def _round_robin(insts: list[DecoderInstruction], sm_count: int):
    queues = [[] for _ in range(sm_count)]
    for i, inst in enumerate(insts):
        queues[i % sm_count].append(inst)
    return queues


def schedule_and_serialize(g: Pi0ActionExpertGlobals, sm_count: int = SM_COUNT) -> torch.Tensor:
    flat = build_instruction_list(g, sm_count)
    tensor = serialize_instructions(_round_robin(flat, sm_count), device=g.device)
    g.instructions = tensor
    g.num_insts = int(tensor.size(1))
    return tensor
