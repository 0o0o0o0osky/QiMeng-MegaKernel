#pragma once

namespace pi0 {

// SigLIP view-specialized MegaKernel opcodes.
#define OPCODE_SIGLIP_LN_ATTN   1   // pre-attention LayerNorm
#define OPCODE_SIGLIP_LN_FFN    2   // pre-FFN LayerNorm
#define OPCODE_SIGLIP_QKV       3   // GEMM + bias        -> qkv_buf
#define OPCODE_SIGLIP_ATTN      4
#define OPCODE_SIGLIP_OPROJ     5
#define OPCODE_SIGLIP_FC1       6   // GEMM + bias + GELU  -> fc1_buf
#define OPCODE_SIGLIP_FC2       7
#define OPCODE_SIGLIP_PATCH     8
#define OPCODE_SIGLIP_FINAL_NORM 9
#define OPCODE_SIGLIP_PROJECTOR 10  // 1152 -> 2048, vision -> encoder

#define BARRIER_ROW(opcode) ((opcode) - 1)

static constexpr int SIGLIP_NUM_BARRIER_ROWS = 10;

static constexpr int BARRIER_SLOTS = 64;

__device__ inline void mk_phase_release(uint32_t *bar, bool fused_release) {
    if (fused_release) {
        asm volatile("red.release.gpu.global.add.u32 [%0], %1;"
                     :: "l"(bar), "r"(1u) : "memory");
    } else {
        __threadfence();
        atomicAdd(bar, 1u);
    }
}

template <typename Config, typename Globals, int Opcode, int BarrierRow> struct siglip_ln;

template <typename Config, typename Globals, typename Epi,
          auto A_Ptr, auto B_Ptr, auto Bias_Ptr, auto Out_Ptr,
          int K_ITERS, int Opcode, int BarrierRow, auto Aux_Ptr,
          bool _ExplicitOut, auto CVec_Ptr, int _StatsTiles, int _TileN> struct siglip_gemm;
template <typename Config, typename Globals, int Opcode, int BarrierRow> struct siglip_attn;

}  // namespace pi0
