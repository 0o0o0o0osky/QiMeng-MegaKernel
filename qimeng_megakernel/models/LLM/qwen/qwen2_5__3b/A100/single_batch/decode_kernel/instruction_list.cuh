#pragma once

#define OPCODE_RMS_QKV_GEMV_ROPE   1
#define OPCODE_ATTENTION_PARTIAL   2
#define OPCODE_ATTENTION_REDUCTION 3
#define OPCODE_O_PROJ_RESIDUAL     4
#define OPCODE_RMS_UPGATE_SILU     5
#define OPCODE_DOWN_PROJ_RESIDUAL  6
#define OPCODE_RMS_LM_HEAD         7

namespace megakernel {

constexpr int FIRST_OPCODE = OPCODE_RMS_QKV_GEMV_ROPE;
constexpr int LAST_OPCODE = OPCODE_RMS_LM_HEAD;

__host__ __device__ constexpr int BARRIER_ROW(int opcode) {
    return opcode - FIRST_OPCODE;
}

struct QwenGlobals;
struct SmemLayout;

__device__ __forceinline__
void rms_qkv_gemv_rope(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void decode_attention_splitkv(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void decode_attention_reduction(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void oproj_gemv_residual(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void rms_upgate_gemv_silu(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void down_proj_gemv_residual(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void rms_lmhead_gemv(
    QwenGlobals& g,
    SmemLayout&   smem
);

}  // namespace megakernel
