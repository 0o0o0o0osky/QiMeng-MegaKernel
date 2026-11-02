#pragma once

#include "kittens.cuh"

namespace llama_mb {

// H100 B<=16 topology: canonical two-slot/five-role VM.
// Eight consumer warps plus loader/storer/launcher/controller give 384 threads.
struct h100_mb16_config {
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;
    static constexpr int INSTRUCTION_WIDTH = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    static constexpr int NUM_CONSUMER_WARPS = 8;
    static constexpr int NUM_CONSUMER_WG = NUM_CONSUMER_WARPS / 4;
    static constexpr int NUM_WARPS = NUM_CONSUMER_WARPS + 4;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 132;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;

    static constexpr int SCRATCH_BYTES = 4096;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + INSTRUCTION_WIDTH * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY =
        MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = true;

    static constexpr int CONSUMER_REGISTERS = 208;
    static constexpr int NON_CONSUMER_REGISTERS = 64;

    static_assert(NUM_THREADS == 384);
    static_assert(NUM_PAGES == 13);
    static_assert(NUM_CONSUMER_WARPS * ::kittens::WARP_THREADS *
                              CONSUMER_REGISTERS +
                          4 * ::kittens::WARP_THREADS *
                              NON_CONSUMER_REGISTERS <= 65536,
                  "register file over-subscribed");
};

}  // namespace llama_mb
