from __future__ import annotations

import os

import torch

from .instructions import kDecBarrierRows, kDecBarrierSlots

DECODER_WIDTH = 1024
ATTN_IN_DIM = 2048
QKV_OUT_DIM = 2560
MLP_WIDTH = 4096
ACTION_DIM = 32
NUM_Q_HEADS = 8
HEAD_DIM = 256
STATE_ROWS = 1
Q_WIDTH = NUM_Q_HEADS * HEAD_DIM

K_SPLITS_DOWN = int(os.environ.get("PI0_KSPLIT_DOWN", "4"))


class Pi0ActionExpertGlobals:
    def __init__(
        self,
        *,
        num_layers: int = 18,
        flow_steps: int = 10,
        chunk_size: int = 50,
        keys_prefix: int = 0,
        device: str = "cuda",
    ):
        self.device = device
        self.num_layers = num_layers
        self.flow_steps = flow_steps
        self.chunk_size = chunk_size
        self.keys_prefix = keys_prefix
        self.rows_total = STATE_ROWS + chunk_size
        self.prefix_alloc = keys_prefix
        # Real key count = prefix + decoder rows; pad the KV-cache stride to 64 so the
        # materialized QK/PV GEMMs get a clean N/K (pad rows zeroed; softmax masks them).
        self.total_keys = self.prefix_alloc + self.rows_total
        self.kv_stride = ((self.total_keys + 63) // 64) * 64
        self.num_layer_slots = num_layers + 1
        self.instructions: torch.Tensor | None = None
        self.num_insts = 0

        d = device
        bf = torch.bfloat16
        C = chunk_size
        R = self.rows_total
        L = num_layers
        F = flow_steps

        def z(*shape, dtype=bf):
            return torch.zeros(shape, dtype=dtype, device=d)

        self.barriers_block = torch.zeros(
            F * self.num_layer_slots * kDecBarrierRows * kDecBarrierSlots,
            dtype=torch.int32,
            device=d,
        )

        self.observation_state_normalized = z(1, ACTION_DIM)
        self.decoder_state_buf = z(1, DECODER_WIDTH)
        self.decoder_x = z(R, DECODER_WIDTH)
        self.decoder_x_buf = z(C, DECODER_WIDTH)
        self.qkv = z(R, QKV_OUT_DIM)
        self.attn_out = z(R, Q_WIDTH)
        self.hidden = z(R, MLP_WIDTH)
        self.splitk_partial = z(K_SPLITS_DOWN, R, DECODER_WIDTH, dtype=torch.float32)
        scores_bytes = R * NUM_Q_HEADS * self.kv_stride * 2
        stats_bytes = 2 * 64 * 4
        workspace_bytes = ((scores_bytes + 255) // 256) * 256 + stats_bytes
        self.attention_workspace = z((workspace_bytes + 3) // 4, dtype=torch.float32)

        self.encoder_k = z(L, self.kv_stride, HEAD_DIM)
        self.encoder_v = z(L, self.kv_stride, HEAD_DIM)
        # Materialized cross-attention: packed rope'd Q (head folded into M) + V^T.
        self.q_packed = z(R * NUM_Q_HEADS, HEAD_DIM)
        self.encoder_vt = z(L, HEAD_DIM, self.kv_stride)

        self.diffusion_noise = z(C, ACTION_DIM)
        self.diffusion_snapshot = z(F, C, ACTION_DIM)

        self.qkv_w = z(L, QKV_OUT_DIM, DECODER_WIDTH)
        self.o_w = z(L, DECODER_WIDTH, ATTN_IN_DIM)
        self.gate_w = z(L, MLP_WIDTH, DECODER_WIDTH)
        self.up_w = z(L, MLP_WIDTH, DECODER_WIDTH)
        self.down_w = z(L, DECODER_WIDTH, MLP_WIDTH)

        self.rope_weights = z(R, HEAD_DIM)
        self.state_in_w = z(DECODER_WIDTH, ACTION_DIM)
        self.state_in_b = z(DECODER_WIDTH)
        self.action_in_w = z(DECODER_WIDTH, ACTION_DIM)
        self.action_time_biases = z(F, DECODER_WIDTH)
        self.action_mlp_w = z(DECODER_WIDTH, DECODER_WIDTH)
        self.action_mlp_b = z(DECODER_WIDTH)
        self.action_out_w = z(ACTION_DIM, DECODER_WIDTH)
        self.action_out_b = z(ACTION_DIM)

    def load_fixture(self, *, weights: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor], encoder_seq_len: int):
        self.observation_state_normalized.copy_(inputs["observation_state_normalized"].view(1, ACTION_DIM))
        self.diffusion_noise.copy_(inputs["diffusion_noise"])

        self.state_in_w.copy_(weights["decoder_state_in_proj_w"].transpose(0, 1).contiguous())
        self.state_in_b.copy_(weights["decoder_state_in_proj_b"])
        self.action_in_w.copy_(weights["decoder_action_fused_in_proj_w"].transpose(0, 1).contiguous())
        self.action_time_biases.copy_(weights["decoder_action_fused_time_biases"])
        self.action_mlp_w.copy_(weights["decoder_action_mlp_w"].transpose(0, 1).contiguous())
        self.action_mlp_b.copy_(weights["decoder_action_mlp_b"])
        self.action_out_w.copy_(weights["decoder_action_fused_out_proj_w"].transpose(0, 1).contiguous())
        self.action_out_b.copy_(weights["decoder_action_fused_out_proj_b"])

        self.qkv_w.copy_(weights["decoder_attn_qkv_w"].transpose(1, 2).contiguous())
        self.o_w.copy_(weights["decoder_attn_o_w"].transpose(1, 2).contiguous())
        self.gate_w.copy_(weights["decoder_ffn_gate_w"].transpose(1, 2).contiguous())
        self.up_w.copy_(weights["decoder_ffn_up_w"].transpose(1, 2).contiguous())
        self.down_w.copy_(weights["decoder_ffn_down_w"].transpose(1, 2).contiguous())

        pos = torch.arange(self.rows_total, device=self.device, dtype=torch.float32) + encoder_seq_len
        inv = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, device=self.device, dtype=torch.float32) / HEAD_DIM))
        phase = pos[:, None] * inv[None, :]
        rope = torch.zeros((self.rows_total, HEAD_DIM), device=self.device, dtype=torch.float32)
        rope[:, 0::2] = torch.cos(phase)
        rope[:, 1::2] = torch.sin(phase)
        self.rope_weights.copy_(rope.to(torch.bfloat16))

    def set_encoder_prefix(self, encoder_k: torch.Tensor, encoder_v: torch.Tensor):
        self.encoder_k.zero_()
        self.encoder_v.zero_()
        self.encoder_vt.zero_()
        self.encoder_k[:, : self.keys_prefix, :].copy_(encoder_k[:, : self.keys_prefix, :])
        self.encoder_v[:, : self.keys_prefix, :].copy_(encoder_v[:, : self.keys_prefix, :])
        # V^T prefix: [L, keys_prefix, 256] -> [L, 256, keys_prefix]
        self.encoder_vt[:, :, : self.keys_prefix].copy_(
            encoder_v[:, : self.keys_prefix, :].transpose(1, 2)
        )

    def launch_args(self, sm_count: int = 108):
        return (
            self.instructions,
            self.barriers_block,
            self.num_layers,
            self.flow_steps,
            self.chunk_size,
            self.keys_prefix,
            self.prefix_alloc,
            self.kv_stride,
            self.observation_state_normalized,
            self.decoder_state_buf,
            self.decoder_x,
            self.decoder_x_buf,
            self.qkv,
            self.attn_out,
            self.hidden,
            self.splitk_partial,
            self.attention_workspace,
            self.encoder_k,
            self.encoder_v,
            self.diffusion_noise,
            self.diffusion_snapshot,
            self.qkv_w,
            self.o_w,
            self.gate_w,
            self.up_w,
            self.down_w,
            self.rope_weights,
            self.state_in_w,
            self.state_in_b,
            self.action_in_w,
            self.action_time_biases,
            self.action_mlp_w,
            self.action_mlp_b,
            self.action_out_w,
            self.action_out_b,
            self.q_packed,
            self.encoder_vt,
            sm_count,
        )
