#pragma once

#include "kittens.cuh"

namespace megakernel {

struct llama_3b_config {
    // Instruction pipeline
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;

    // num bits required to represent num pipeline stages
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;

    static constexpr int INSTRUCTION_WIDTH = 32; // 128 bytes per instruction.
    using instruction_t = int[INSTRUCTION_WIDTH];

    // How many semaphores are available for dynamic use?
    static constexpr int DYNAMIC_SEMAPHORES = 32;

    // One controller warp, one load warp, one store warp, and one mma warp.
    static constexpr int NUM_CONSUMER_WARPS = 24;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;

    // Shared memory declared statically
    static constexpr int SCRATCH_BYTES = 5120;
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + INSTRUCTION_WIDTH * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int DYNAMIC_SHARED_MEMORY =
        ::kittens::MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;

    // Shared memory declared dynamically
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = DYNAMIC_SHARED_MEMORY / PAGE_SIZE;
    static_assert(NUM_PAGES == 13, "NUM_PAGES must be 13");

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    // Per-SM register limit on H100 is 65536. With 24 consumer warps (768 threads) and 4
    // non-consumer warps (128 threads × NON_CONSUMER_REGISTERS): total = 768*C + 128*64.
    // 768*C + 8192 <= 65536 => C <= 74.67; setmaxnreg requires multiple of 8 => max C = 72.
    static constexpr int CONSUMER_REGISTERS = 72;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
    static_assert(NUM_CONSUMER_WARPS * ::kittens::WARP_THREADS * CONSUMER_REGISTERS +
                      (NUM_WARPS - NUM_CONSUMER_WARPS) * ::kittens::WARP_THREADS * NON_CONSUMER_REGISTERS
                  <= 65536,
                  "Total block registers must not exceed 65536 per SM (H100 limit)");
};

using default_config = llama_3b_config;

template <typename config>
using instruction_layout = kittens::gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>;

} // namespace megakernel