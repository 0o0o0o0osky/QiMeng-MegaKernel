#pragma once

#include "kittens.cuh"

namespace megakernel {

// Qwen3 0.6B: 2-stage instruction pipeline, 13 pages, 16 warps for hidden_dim=1024.
struct qwen3_0_6b_config {
    static constexpr int INSTRUCTION_PIPELINE_STAGES = 2;
    static constexpr int INSTRUCTION_PIPELINE_STAGES_BITS = 1;
    static constexpr int INSTRUCTION_WIDTH = 32;
    using instruction_t = int[INSTRUCTION_WIDTH];

    static constexpr int DYNAMIC_SEMAPHORES = 32;

    static constexpr int NUM_CONSUMER_WARPS = 16;
    static constexpr int NUM_WARPS = 4 + NUM_CONSUMER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * ::kittens::WARP_THREADS;
    static constexpr int NUM_BLOCKS = 1;
    static constexpr int CLUSTER_BLOCKS = 1;
    static constexpr int MAX_SHARED_MEMORY = ::kittens::MAX_SHARED_MEMORY;

    static constexpr int SCRATCH_BYTES = 5120;
    // STATIC = 512 + 2*(5120+128+256) = 11520
    static constexpr int STATIC_SHARED_MEMORY =
        512 + INSTRUCTION_PIPELINE_STAGES *
                  (SCRATCH_BYTES + INSTRUCTION_WIDTH * 4 +
                   DYNAMIC_SEMAPHORES * 8);
    static constexpr int PAGE_SIZE = 16384;
    static constexpr int NUM_PAGES = 13;
    // megakernel.cuh aligns __shm to 1024B; need headroom so 13 pages fit after alignment.
    static constexpr int DYNAMIC_SHARED_MEMORY = NUM_PAGES * PAGE_SIZE + 1024;

    static constexpr bool GMEM_SPIN_LOOP_SLEEP_NANOS = 20;

    static constexpr int CONSUMER_REGISTERS = 104;
    static constexpr int NON_CONSUMER_REGISTERS = 64;
    static_assert(NUM_CONSUMER_WARPS * ::kittens::WARP_THREADS * CONSUMER_REGISTERS +
                      (NUM_WARPS - NUM_CONSUMER_WARPS) * ::kittens::WARP_THREADS * NON_CONSUMER_REGISTERS
                  <= 65536,
                  "Total block registers must not exceed 65536 per SM (H100 limit)");
};

using default_config = qwen3_0_6b_config;

template <typename config>
using instruction_layout = kittens::gl<int, 1, -1, -1, config::INSTRUCTION_WIDTH>;

template <typename config> void print_config() {
    std::cout << "---------------- CONFIG INFO ----------------" << std::endl;
    std::cout << "INSTRUCTION_PIPELINE_STAGES: "
              << config::INSTRUCTION_PIPELINE_STAGES << std::endl;
    std::cout << "INSTRUCTION_WIDTH: " << config::INSTRUCTION_WIDTH
              << std::endl;
    std::cout << "NUM_CONSUMER_WARPS: " << config::NUM_CONSUMER_WARPS
              << std::endl;
    std::cout << "NUM_WARPS: " << config::NUM_WARPS << std::endl;
    std::cout << "NUM_THREADS: " << config::NUM_THREADS << std::endl;
    std::cout << "NUM_BLOCKS: " << config::NUM_BLOCKS << std::endl;
    std::cout << "CLUSTER_BLOCKS: " << config::CLUSTER_BLOCKS << std::endl;
    std::cout << "MAX_SHARED_MEMORY: " << config::MAX_SHARED_MEMORY
              << std::endl;
    std::cout << "STATIC_SHARED_MEMORY: " << config::STATIC_SHARED_MEMORY
              << std::endl;
    std::cout << "PAGE_SIZE: " << config::PAGE_SIZE << std::endl;
    std::cout << "NUM_PAGES: " << config::NUM_PAGES << std::endl;
    std::cout << "SCRATCH_BYTES: " << config::SCRATCH_BYTES << std::endl;
    std::cout << "DYNAMIC_SEMAPHORES: " << config::DYNAMIC_SEMAPHORES
              << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
}

} // namespace megakernel
