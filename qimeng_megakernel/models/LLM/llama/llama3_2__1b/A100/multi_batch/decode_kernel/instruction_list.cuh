#pragma once

// Instruction opcodes for the multi-batch decode MegaKernel.
// Must stay in sync with instruction_defs.py opcode() methods.
#define OPCODE_ATTN_RMS            1
#define OPCODE_QKV_GEMM_ROPE       2
#define OPCODE_ATTENTION_PARTIAL   3
#define OPCODE_ATTENTION_REDUCTION 4
#define OPCODE_O_PROJ_RESIDUAL     5
#define OPCODE_MLP_RMS             6
#define OPCODE_UPGATE_SILU         7
#define OPCODE_DOWN_PROJ_RESIDUAL  8
#define OPCODE_LMHEAD_RMS          9
#define OPCODE_LMHEAD_GEMM         10

namespace megakernel {

struct LlamaGlobals;
struct SmemLayout;

// Attention RMS selected by this instance; writes attn_rms_out.
__device__ __forceinline__
void decode_attn_rms(
    LlamaGlobals& g,
    SmemLayout& smem,
    int inst_idx = -1);

// QKV GEMM + RoPE; reads attn_rms_out.
__device__ __forceinline__
void qkv_gemm_rope(
    LlamaGlobals& g,
    SmemLayout& smem,
    int inst_idx = -1);

// Multi-batch attention split-KV (one partial tile).
__device__ __forceinline__
void decode_attention_splitkv(
    LlamaGlobals& g,
    SmemLayout& smem,
    int inst_idx = -1);

// Multi-batch attention reduction (merge partial tiles → final output).
__device__ __forceinline__
void decode_attention_reduction(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx = -1
);

// Multi-batch OProj GEMM with residual post-slot.
__device__ __forceinline__
void decode_oproj_gemm_residual(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx = -1
);

// MLP RMS selected by this instance; writes mlp_rms_out.
__device__ __forceinline__
void decode_mlp_rms(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx = -1
);

// Multi-batch dual Up/Gate GEMM + SiLU; reads mlp_rms_out.
__device__ __forceinline__
void decode_upgate_gemm_silu(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx = -1
);

// Multi-batch DownProj GEMM with residual post-slot.
__device__ __forceinline__
void decode_downproj_gemm_residual(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx = -1
);

// Final RMS selected by this instance; writes lm_head_rms_out.
__device__ __forceinline__
void decode_lmhead_rms(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx = -1
);

// Multi-batch LM-head GEMM; reads lm_head_rms_out.
__device__ __forceinline__
void decode_lmhead_gemm(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx = -1
);

}  // namespace megakernel
