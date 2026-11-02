#pragma once

#ifndef NDEBUG

#ifndef AMPERE_BARRIER_WATCHDOG_MAX_ITERS
#define AMPERE_BARRIER_WATCHDOG_MAX_ITERS 20000000
#endif

namespace megakernel {
namespace debug {

__device__ __forceinline__ void barrier_wait_with_timeout(
    const int32_t* barrier_ptr,
    int expected_count
) {
    const uint32_t expected = static_cast<uint32_t>(expected_count);
    uint32_t loaded_value = 0;
    int wait_count = 0;

    while (true) {
        loaded_value = barrier_load_relaxed(barrier_ptr);
        if (loaded_value >= expected) {
            loaded_value = barrier_load_acquire(barrier_ptr);
            if (loaded_value >= expected) {
                return;
            }
        }

        ++wait_count;
        if (wait_count >= AMPERE_BARRIER_WATCHDOG_MAX_ITERS) {
            if (threadIdx.x == 0) {
                printf(
                    "BARRIER TIMEOUT: sm=%d barrier=%p expected=%u actual=%u "
                    "iterations=%d\n",
                    static_cast<int>(blockIdx.x),
                    static_cast<const void*>(barrier_ptr),
                    expected,
                    loaded_value,
                    wait_count
                );
            }
            __trap();
            return;
        }
        __nanosleep(KernelConfig::GMEM_SPIN_LOOP_SLEEP_NANOS);
    }
}

} // namespace debug
} // namespace megakernel

#endif // NDEBUG
