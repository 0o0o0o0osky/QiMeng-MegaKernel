#pragma once

#define OPCODE_RMS_QKV_GEMV_ROPE   1
#define OPCODE_ATTENTION_PARTIAL   2
#define OPCODE_ATTENTION_REDUCTION 3
#define OPCODE_O_PROJ_RESIDUAL     4
#define OPCODE_ROUTER_GEMV         8
#define OPCODE_ROUTER_SOFTMAX_TOPK 9
#define OPCODE_SHARED_UPGATE_SILU 10
#define OPCODE_SHARED_DOWNPROJ   11
#define OPCODE_ROUTED_UPGATE_SILU 12
#define OPCODE_ROUTED_DOWNPROJ    13
#define OPCODE_MOE_COMBINE_COMMIT 14
#define OPCODE_RMS_LMHEAD_GEMV     7

namespace megakernel {

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
void router_gemv(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void router_softmax_topk(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void shared_expert_upgate_silu(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void shared_expert_downproj(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void routed_expert_upgate_silu(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void routed_expert_downproj(
    QwenGlobals& g,
    SmemLayout&   smem
);

__device__ __forceinline__
void moe_combine_commit(
    QwenGlobals& g,
    SmemLayout&   smem
);

}  // namespace megakernel
