#pragma once

// Instruction opcodes for the decode MegaKernel.
// These values must stay in sync with the Python scheduler
// defined in decode_schedule/instructions/instruction_defs.py.
//
// RMS_QKV_GEMV_RoPE
#define OPCODE_RMS_QKV_GEMV_ROPE   1
#define OPCODE_ATTENTION_PARTIAL   2
#define OPCODE_ATTENTION_REDUCTION 3
#define OPCODE_O_PROJ_RESIDUAL     4
#define OPCODE_RMS_UPGATE_SILU     5
#define OPCODE_DOWN_PROJ_RESIDUAL  6
#define OPCODE_RMS_LM_HEAD         7

// Barrier rows are the zero-based opcodes, matching the H100 kernels.
#define BARRIER_ROW(opcode) ((opcode) - 1)

namespace megakernel {

struct LlamaGlobals;
struct SmemLayout;

// RMS + QKV GEMV + RoPE stage (SlicedK is an implementation detail inside the .cu).
__device__ __forceinline__
void rms_qkv_gemv_rope(
    LlamaGlobals& g,
    SmemLayout&   smem
);

// Decode attention split-KV computation.
__device__ __forceinline__
void decode_attention_splitkv(
    LlamaGlobals& g,
    SmemLayout&   smem
);

// Decode attention reduction.
__device__ __forceinline__
void decode_attention_reduction(
    LlamaGlobals& g,
    SmemLayout&   smem
);

// O projection GEMV + residual addition.
__device__ __forceinline__
void oproj_gemv_residual(
    LlamaGlobals& g,
    SmemLayout&   smem
);

// RMS norm + Up/Gate projection GEMV + SiLU.
__device__ __forceinline__
void rms_upgate_gemv_silu(
    LlamaGlobals& g,
    SmemLayout&   smem
);

// Down projection GEMV + residual (atomic add).
__device__ __forceinline__
void down_proj_gemv_residual(
    LlamaGlobals& g,
    SmemLayout&   smem
);

// RMS norm + LM Head GEMV.
__device__ __forceinline__
void rms_lmhead_gemv(
    LlamaGlobals& g,
    SmemLayout&   smem
);

}  // namespace megakernel
