#pragma once

#include <cstdint>

#ifndef DECODE_GMEM_SPIN_LOOP_SLEEP_NANOS
#define DECODE_GMEM_SPIN_LOOP_SLEEP_NANOS 20
#endif

namespace megakernel {

// Whole-CTA tuning: shared by all operators in the block (RMS norm, GEMM, etc.).
// Per-GEMM pipeline knobs (kTileK, kStage, swizzle, MMA EU repeat M/K) live in
// utils.cuh — struct GemmSharedBase (edit there to retune the GEMM).
//
struct KernelConfig {
    static constexpr int NUM_WARPS   = 4;
    static constexpr int NUM_THREADS = NUM_WARPS * 32;
    static constexpr int TARGET_CTAS_PER_SM = 2;

    static constexpr int NUM_SMS                    = 108;
    static constexpr int NUM_WORKERS                = NUM_SMS * TARGET_CTAS_PER_SM;
    static constexpr int GMEM_SPIN_LOOP_SLEEP_NANOS = DECODE_GMEM_SPIN_LOOP_SLEEP_NANOS;

    // Scheduler / epilogue: M rows (max decode batch), base N-tile granularity.
    static constexpr int kTileM = 16;   // max batch size for decode (M dimension)
    static constexpr int kTileN = 16;   // weight rows per N-tile (single-tile config)

    // Per-CTA operator workspace.  The instruction words live in a separate
    // 128-byte member below. The workspace is sized for two resident CTAs.
    static constexpr int SHARED_MEMORY_SIZE = 64 * 1024;

    static_assert(SHARED_MEMORY_SIZE >= 48 * 1024,
                  "Ampere decode workspace is too small for the attention/GEMM overlays");
};

struct SmemLayout {
    alignas(128) uint8_t main_workspace[KernelConfig::SHARED_MEMORY_SIZE];
    alignas(128) int32_t instruction[32];
};

}  // namespace megakernel
