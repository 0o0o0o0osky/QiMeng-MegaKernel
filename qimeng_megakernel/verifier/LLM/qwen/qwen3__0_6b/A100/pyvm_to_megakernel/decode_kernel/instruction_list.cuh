#pragma once

// Instruction opcodes for the decode MegaKernel.
// These values must stay in sync with the Python scheduler
// defined in decode_schedule/instructions/instruction_defs.py.
//
// RMS_QKV_GEMV
#define OPCODE_RMS_QKV_GEMV        1
#define OPCODE_ATTENTION_PARTIAL   2
#define OPCODE_ATTENTION_REDUCTION 3
#define OPCODE_O_PROJ_RESIDUAL     4
#define OPCODE_RMS_UPGATE_SILU     5
#define OPCODE_DOWN_PROJ_RESIDUAL  6
#define OPCODE_RMS_LM_HEAD         7

// Barrier rows are the zero-based opcodes, matching the H100 kernels.
#define BARRIER_ROW(opcode) ((opcode) - 1)

namespace megakernel {

struct QwenGlobals;
struct SmemLayout;

// RMS + QKV GEMV stage. Q/K norm and RoPE are applied in decode attention.
__device__ __forceinline__
void rms_qkv_gemv(
    QwenGlobals& g,
    SmemLayout&  smem
);

// Decode attention split-KV computation, including Q/K norm and RoPE.
__device__ __forceinline__
void decode_attention_splitkv(
    QwenGlobals& g,
    SmemLayout&  smem
);

// Decode attention reduction.
__device__ __forceinline__
void decode_attention_reduction(
    QwenGlobals& g,
    SmemLayout&  smem
);

// O projection GEMV + residual addition.
__device__ __forceinline__
void oproj_gemv_residual(
    QwenGlobals& g,
    SmemLayout&  smem
);

// RMS norm + Up/Gate projection GEMV + SiLU.
__device__ __forceinline__
void rms_upgate_gemv_silu(
    QwenGlobals& g,
    SmemLayout&  smem
);

// Down projection GEMV + residual (atomic add).
__device__ __forceinline__
void down_proj_gemv_residual(
    QwenGlobals& g,
    SmemLayout&  smem
);

// RMS norm + LM Head GEMV.
__device__ __forceinline__
void rms_lmhead_gemv(
    QwenGlobals& g,
    SmemLayout&  smem
);

}  // namespace megakernel
