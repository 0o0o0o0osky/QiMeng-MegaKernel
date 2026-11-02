#pragma once

#include "globals.cuh"
#include "barrier.cuh"

// Qwen1.5-MoE-A2.7B H100 opcode table, aligned with the L3 scheduler.
// Opcodes 5 and 6 are unused because the model's MoE block replaces the dense MLP.
// Opcode 0 is inserted by megakernel_wrapper.

#define OPCODE_RMS_QKVPROJ_GEMV_ROPE   1
#define OPCODE_DECODE_ATTENTION_SPLITKV 2
#define OPCODE_DECODE_ATTENTION_REDUCTION 3
#define OPCODE_OPROJ_GEMV_RESIDUAL     4
#define OPCODE_RMS_LMHEAD_GEMV         7
#define OPCODE_ROUTER_GEMV             8
#define OPCODE_ROUTER_SOFTMAX_TOPK     9
#define OPCODE_SHARED_EXPERT_UPGATE_SILU 10
#define OPCODE_SHARED_EXPERT_DOWNPROJ  11
#define OPCODE_ROUTED_EXPERT_UPGATE_SILU 12
#define OPCODE_ROUTED_EXPERT_DOWNPROJ  13
#define OPCODE_MOE_COMBINE_COMMIT      14

// ---------------------------------------------------------------------------
// Barrier rows follow the opcode-minus-one schedule ABI.
// ---------------------------------------------------------------------------
#define BARRIER_ROW(opcode) ((opcode) - 1)

// ---------------------------------------------------------------------------
// Forward declarations.  One struct per emitted opcode; implementations live in
// the matching `.cu` file, which mk.cu `#include`s (single translation unit —
// the Makefile only ever compiles mk.cu, so adding an op does NOT touch it).
//
//   opcode  struct                            file
//   ------  --------------------------------  ---------------------------------
//     1     rms_qkvproj_gemv_rope             rms_qkv_gemv_rope.cu
//     2     decode_attention_splitkv          decode_attention_splitkv.cu
//     3     decode_attention_reduction        decode_attention_reduction.cu
//     4     oproj_gemv_residual               oproj_gemv_residual.cu
//     7     rms_lmhead_gemv                   rms_lmhead_gemv.cu
//     8     router_gemv                       router_gemv.cu
//     9     router_softmax_topk               router_softmax_topk.cu
//    10     shared_expert_upgate_silu         shared_expert_upgate_silu.cu
//    11     shared_expert_downproj            shared_expert_downproj.cu
//    12     routed_expert_upgate_silu         routed_expert_upgate_silu.cu
//    13     routed_expert_downproj            routed_expert_downproj.cu
//    14     moe_combine_commit                moe_combine_commit.cu
// ---------------------------------------------------------------------------
template <typename Config = config, typename Globals = qwen_moe_globals>
struct rms_qkvproj_gemv_rope;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct decode_attention_splitkv;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct decode_attention_reduction;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct oproj_gemv_residual;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct rms_lmhead_gemv;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct router_gemv;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct router_softmax_topk;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct shared_expert_upgate_silu;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct shared_expert_downproj;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct routed_expert_upgate_silu;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct routed_expert_downproj;

template <typename Config = config, typename Globals = qwen_moe_globals>
struct moe_combine_commit;
