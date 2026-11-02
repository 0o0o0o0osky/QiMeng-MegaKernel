#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#ifndef NDEBUG
#include <cstdio>
#endif

#ifndef DEBUG_PRINT
#define DEBUG_PRINT 0
#endif

namespace ampere_barrier {

__device__ __forceinline__ uint32_t load_relaxed(
    const volatile int32_t* counter) {
    uint32_t value;
    asm volatile(
        "ld.relaxed.gpu.global.u32 %0, [%1];"
        : "=r"(value)
        : "l"(counter)
        : "memory");
    return value;
}

__device__ __forceinline__ uint32_t load_acquire(
    const volatile int32_t* counter) {
    uint32_t value;
    asm volatile(
        "ld.acquire.gpu.global.u32 %0, [%1];"
        : "=r"(value)
        : "l"(counter)
        : "memory");
    return value;
}

__device__ __forceinline__ void wait(
    volatile int32_t* counter,
    uint32_t expected,
#ifndef NDEBUG
    int cta_id,
    int pc,
    int opcode,
    int counter_offset,
    uint64_t timeout_cycles,
#endif
    int nanosleep) {
#ifndef NDEBUG
    const uint64_t start = clock64();
#endif
    while (load_relaxed(counter) < expected) {
#ifndef NDEBUG
        if (clock64() - start > timeout_cycles) {
#if DEBUG_PRINT
            printf(
                "AE barrier timeout: cta=%d pc=%d op=%d off=%d exp=%u got=%u\n",
                cta_id,
                pc,
                opcode,
                counter_offset,
                expected,
                load_relaxed(counter));
#endif
            asm volatile("trap;");
        }
#endif
        if (nanosleep > 0) {
            __nanosleep(nanosleep);
        }
    }
    while (load_acquire(counter) < expected) {
        if (nanosleep > 0) {
            __nanosleep(nanosleep);
        }
    }
}

__device__ __forceinline__ void publish(
    volatile int32_t* counter,
    uint32_t increment = 1) {
    asm volatile(
        "red.release.gpu.global.add.u32 [%0], %1;"
        :
        : "l"(counter), "r"(increment)
        : "memory");
}

}  // namespace ampere_barrier
