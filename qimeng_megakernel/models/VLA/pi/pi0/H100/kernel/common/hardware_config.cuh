#pragma once

#include "kittens.cuh"
#include "../include/config.cuh"

namespace pi0 {

// Backbone configuration for the SigLIP and Gemma encoders.
struct backbone_config {
    static constexpr int INSTRUCTION_PIPELINE_STAGES      = 2;
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];

    static constexpr int DYNAMIC_SEMAPHORES = 128;

    static constexpr int NUM_CONSUMER_WARPS = 8;                  // 2 warpgroup
    static constexpr int NUM_WARPS   = 4 + NUM_CONSUMER_WARPS;    // + loader/storer/launcher/controller
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;  // 384
    static constexpr int NUM_BLOCKS     = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;   // 227000

    // matmul_pipeline.cuh: static_assert(SCRATCH_BYTES >= 8192)

    static constexpr int SCRATCH_BYTES = 8192 + 2048;

    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + INSTRUCTION_WIDTH * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int PAGE_SIZE = 32768;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;

    static_assert(NUM_PAGES == 6, "backbone_config: matmul_pipeline requires exactly 6 pages");

    // 8*32*208 + 4*32*64 = 53248 + 8192 = 61440 <= 65536
    static constexpr int CONSUMER_REGISTERS     = 208;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
    static_assert(NUM_CONSUMER_WARPS * 32 * CONSUMER_REGISTERS +
                      (NUM_WARPS - NUM_CONSUMER_WARPS) * 32 * NON_CONSUMER_REGISTERS <= 65536,
                  "backbone_config: register file over-subscribed");
};

// Action Expert configuration: M <= 51 is padded to 64 for one warpgroup.
struct action_expert_config {
    static constexpr int INSTRUCTION_PIPELINE_STAGES      = 2;
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];

    static constexpr int DYNAMIC_SEMAPHORES = 128;

    static constexpr int NUM_CONSUMER_WARPS = 4;                  // 1 warpgroup
    static constexpr int NUM_WARPS   = 4 + NUM_CONSUMER_WARPS;    // 8
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;  // 256
    static constexpr int NUM_BLOCKS     = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;

    static constexpr int SCRATCH_BYTES = 8192 + 2048;

    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + INSTRUCTION_WIDTH * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    static constexpr int PAGE_SIZE = 32768;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 6, "action_expert_config: matmul_pipeline requires exactly 6 pages");

    // 4*32*232 + 4*32*64 = 29696 + 8192 = 37888 <= 65536

    static constexpr int CONSUMER_REGISTERS     = 232;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
    static_assert(NUM_CONSUMER_WARPS * 32 * CONSUMER_REGISTERS +
                      (NUM_WARPS - NUM_CONSUMER_WARPS) * 32 * NON_CONSUMER_REGISTERS <= 65536,
                  "action_expert_config: register file over-subscribed");
};

static constexpr int H100_SM_COUNT = 132;

}  // namespace pi0
