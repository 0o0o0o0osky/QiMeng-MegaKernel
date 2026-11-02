#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

#include "common/barrier.cuh"
#include "common/instruction_list.cuh"
#include "common/pi0_config.cuh"

// Backoff, in nanoseconds, between dependency-counter polls. Zero spins.
#ifndef PI0_BARRIER_NANOSLEEP
#define PI0_BARRIER_NANOSLEEP 0
#endif

namespace pi0_a100 {

__device__ __forceinline__ volatile int32_t* barrier_ptr(
    int32_t* base, int layer, int row, int rows) {
    return base + (static_cast<int64_t>(layer) * rows + row) * kBarrierSlots;
}

__device__ __forceinline__ void wait_dependency(
    int32_t* barriers,
    int rows,
    int dep_layer,
    int dep_row,
    int dep_expected,
    int cta,
    int pc,
    int opcode) {
    if (dep_expected <= 0) return;
    if (threadIdx.x == 0) {
        ampere_barrier::wait(
            barrier_ptr(barriers, dep_layer, dep_row, rows),
            static_cast<uint32_t>(dep_expected),
#ifndef NDEBUG
            cta,
            pc,
            opcode,
            (dep_layer * rows + dep_row) * kBarrierSlots * sizeof(int32_t),
            4000000000ULL,
#endif
            PI0_BARRIER_NANOSLEEP);
    }
    __syncthreads();
}

__device__ __forceinline__ void publish_phase(
    int32_t* barriers, int rows, int layer, int opcode) {
    __syncthreads();
    if (threadIdx.x == 0) {
        ampere_barrier::publish(
            barrier_ptr(barriers, layer, barrier_row(opcode), rows));
    }
}

// Dependency-waiting CTAs must be resident together. Reject oversized grids.
template <typename Kernel>
inline void validate_launch_shape(Kernel kernel, int smem_bytes, int grid) {
    int active = 0;
    cudaError_t status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active, kernel, kThreads, smem_bytes);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    if (active < 1) throw std::runtime_error("Pi0 A100 megakernel has zero resident CTAs");
    if (grid > active * kSmCount) {
        throw std::runtime_error(
            "Pi0 A100 megakernel grid " + std::to_string(grid) +
            " exceeds resident capacity " + std::to_string(active * kSmCount) +
            " (" + std::to_string(active) + " CTAs/SM at " +
            std::to_string(smem_bytes) + " B of shared memory); "
            "a persistent kernel deadlocks above it");
    }
}

}  // namespace pi0_a100
