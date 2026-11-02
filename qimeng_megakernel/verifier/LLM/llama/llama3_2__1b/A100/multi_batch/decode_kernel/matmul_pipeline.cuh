#pragma once

#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

// This instance's cp.async atom has a 128-thread ownership layout.  Keep the
// issue mask aligned with that layout so no thread duplicates a transaction.
template <typename Copy, typename Source, typename Destination>
__device__ __forceinline__ void issue_gemm_g2s_128(
    Copy const& copy,
    Source const& source,
    Destination&& destination)
{
    if (threadIdx.x < 128) {
        cute::copy(copy, source, destination);
    }
}

// Instance specialization of the Ampere GEMM stage ring. Operators own their
// input/output slots; this type only controls cp.async group lifetime.
template <int Stages>
struct CpAsyncStageRing {
    static_assert(Stages >= 2, "Ampere GEMM pipeline requires at least two stages");

    int next_tile = 0;
    int write_stage = 0;
    int read_stage = 0;

    __device__ __forceinline__ bool has_next(int tile_count) const {
        return next_tile < tile_count;
    }

    __device__ __forceinline__ void advance_write() {
        ++next_tile;
        write_stage = (write_stage + 1) % Stages;
    }

    // Commit once per logical issue step, including tail steps with no copy.
    // The empty groups preserve the wait-group distance for short-K/tail paths.
    __device__ __forceinline__ void commit_issue_step() const {
        cute::cp_async_fence();
    }

    __device__ __forceinline__ void wait_for_read_stage() const {
        cute::cp_async_wait<Stages - 2>();
        __syncthreads();
    }

    __device__ __forceinline__ void advance_read() {
        read_stage = (read_stage + 1) % Stages;
    }
};

}  // namespace megakernel
