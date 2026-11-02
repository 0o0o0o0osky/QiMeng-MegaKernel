#pragma once
#include <cstdint>

namespace megakernel {
    struct KernelConfig {
        static constexpr int32_t NUM_WARPS              = 16;
        static constexpr int32_t WARP_SIZE              = 32;
        static constexpr int32_t NUM_THREADS            = NUM_WARPS * WARP_SIZE;       // 512
        static constexpr int32_t NUM_SMS                = 108;
        static constexpr int32_t SHARED_MEMORY_SIZE     = 160 * 1024;
        static constexpr int32_t GMEM_SPIN_LOOP_SLEEP_NANOS = 20;
    };

    struct SmemLayout {
        alignas(128) uint8_t  main_workspace[KernelConfig::SHARED_MEMORY_SIZE];
        alignas(128) int32_t  instruction[32];
    };
}
