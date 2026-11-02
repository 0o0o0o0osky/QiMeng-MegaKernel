#pragma once

#include <cstdint>
#include "kittens.cuh"

// Shared wait interface for Hopper instruction-VM kernels.
//
// Producers publish generic stores with a device fence before atomic signalling;
// completed TMA stores carry their async-proxy completion ordering. Consumers spin
// with relaxed loads and close with ld.acquire.gpu before reading published data.
namespace megakernel {

__device__ __forceinline__ int mk_bar_load_relaxed(const int *p) {
    int v;
    asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ int mk_bar_load_acquire(const int *p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void
mk_barrier_spin_acquire(volatile int32_t *barrier_ptr, int expected_count) {
    const int *p = (const int *)barrier_ptr;
    while (mk_bar_load_relaxed(p) < expected_count) {
        /* spin */
    }
    while (mk_bar_load_acquire(p) < expected_count) {
        /* spin */
    }
}

} // namespace megakernel

#ifndef NDEBUG
#define MK_BARRIER_WATCHDOG 1
#include "debug/barrier_watchdog.cuh"
#endif

namespace megakernel {

// expected_count comes from the scheduler's barrier count.
__device__ __forceinline__ void mk_barrier_wait(volatile int32_t *barrier_ptr,
                                                int expected_count) {
#ifdef MK_BARRIER_WATCHDOG
    debug::barrier_wait_with_timeout(barrier_ptr, expected_count);
#else
    mk_barrier_spin_acquire(barrier_ptr, expected_count);
#endif
}

// Blocking CTA-local mbarrier wait used by release kernels.
template <typename Semaphore>
__device__ __forceinline__ void mk_semaphore_wait(Semaphore &sem, int bit) {
#ifdef MK_BARRIER_WATCHDOG
    debug::semaphore_wait_with_timeout(sem, bit);
#else
    kittens::wait(sem, bit);
#endif
}

} // namespace megakernel
