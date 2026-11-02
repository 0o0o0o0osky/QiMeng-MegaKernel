#pragma once

#include "globals.cuh"
#include "barrier.cuh"

#define OPCODE_RMS_QKV_GEMV_ROPE   1
#define OPCODE_DECODE_ATTENTION_SPLIT_KV   2
#define OPCODE_ATTENTION_REDUCTION 3
#define OPCODE_O_PROJ_RESIDUAL     4
#define OPCODE_RMS_UPGATE_SILU     5
#define OPCODE_DOWN_PROJ_RESIDUAL  6
#define OPCODE_RMS_LM_HEAD         7

// `Bar[layer, row, slot]` uses row == opcode - 1.  Keep that mapping behind
// one searchable name so all CUDA call sites share the Python scheduler's
// barrier-row invariant.
#define BARRIER_ROW(opcode) ((opcode) - 1)

template <typename config = config, typename globals = llama_3b_globals>
struct attention_partial;

template <typename config = config, typename globals = llama_3b_globals>
struct attention_reduction;

template <typename config = config, typename globals = llama_3b_globals>
struct rms_qkv_rope_append;

template <typename config = config, typename globals = llama_3b_globals>
struct downproj;

template <typename config = config, typename globals = llama_3b_globals>
struct o_proj;

template <typename config = config, typename globals = llama_3b_globals>
struct rms_upgate_silu;

template <typename config = config, typename globals = llama_3b_globals>
struct rms_lm_head;
