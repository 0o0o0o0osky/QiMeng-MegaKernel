#pragma once

// Debug-only timeout diagnostics for the shared barrier and semaphore waits.
// Lane 0 of every warp may report because waits run in consumer and role warps.
#define WATCHDOG_IS_REPORTER() ((threadIdx.x & 31) == 0)

#ifndef WATCHDOG_BARRIER_MAX_ITERS
#define WATCHDOG_BARRIER_MAX_ITERS 20000000
#endif
#ifndef WATCHDOG_SEMAPHORE_MAX_ITERS
#define WATCHDOG_SEMAPHORE_MAX_ITERS 20000000
#endif
#ifndef WATCHDOG_SLEEP_NANOS
#define WATCHDOG_SLEEP_NANOS 20
#endif

namespace megakernel {
namespace debug {

// Match the release path's relaxed-spin plus acquire-close ordering.
__device__ __forceinline__ void
barrier_wait_with_timeout(volatile int32_t *barrier_ptr, int expected_count) {
    int wait_count = 0;
    int32_t loaded_value;

    while (true) {
        loaded_value = mk_bar_load_relaxed((const int *)barrier_ptr);

        if (loaded_value >= expected_count) {
            loaded_value = mk_bar_load_acquire((const int *)barrier_ptr);
            if (loaded_value >= expected_count)
                break;
        }

        wait_count++;
        if (wait_count >= WATCHDOG_BARRIER_MAX_ITERS) {
            if (WATCHDOG_IS_REPORTER()) {
                printf("BARRIER TIMEOUT! SM %d bar %p | expected %d actual %d | "
                       "warp %d lane 0 (tid %d) | waited ~%.1fs\n",
                       (int)blockIdx.x, (void *)barrier_ptr, expected_count,
                       (int)loaded_value, (int)(threadIdx.x >> 5),
                       (int)threadIdx.x,
                       wait_count * (WATCHDOG_SLEEP_NANOS * 1e-9f));
            }
            __trap();
            return;
        }
        __nanosleep(WATCHDOG_SLEEP_NANOS);
    }
}

// Poll with backoff so the debug build can count and report the timeout.
template <typename Semaphore>
__device__ __forceinline__ void semaphore_wait_with_timeout(Semaphore &sem,
                                                            int bit) {
    int wait_count = 0;
    while (true) {
        int ready = kittens::test_wait(sem, bit);
        if (ready)
            break;
        wait_count++;
        if (wait_count >= WATCHDOG_SEMAPHORE_MAX_ITERS) {
            if (WATCHDOG_IS_REPORTER()) {
                printf("SEMAPHORE TIMEOUT! SM %d sem %p bit %d | "
                       "warp %d lane 0 (tid %d) | waited ~%.1fs\n",
                       (int)blockIdx.x, (void *)&sem, bit,
                       (int)(threadIdx.x >> 5), (int)threadIdx.x,
                       wait_count * (WATCHDOG_SLEEP_NANOS * 1e-9f));
            }
            __trap();
            return;
        }
        __nanosleep(WATCHDOG_SLEEP_NANOS);
    }
}

} // namespace debug
} // namespace megakernel
