#pragma once

// Ampere GEMM execution profiles and CuTe tile aliases.  The profile describes
// launch/resource intent only; it does not prescribe an operation or fusion.

#include "common/gemm_config.cuh"

#ifndef DEBUG_PRINT
#define DEBUG_PRINT 0
#endif

namespace mk_gemm {

static constexpr int kAmpereMaxDynamicSmemPerBlock = 166912;

template <int NumWarps, int TargetCtasPerSm>
struct ExecutionProfile {
    static constexpr bool kSupported =
        (NumWarps == 4 && TargetCtasPerSm == 1) ||
        (NumWarps == 4 && TargetCtasPerSm == 2) ||
        (NumWarps == 8 && TargetCtasPerSm == 1);
    static_assert(
        kSupported,
        "Ampere GEMM supports 4w1CTA, 4w2CTA, or 8w1CTA");

    static constexpr int kNumWarps = NumWarps;
    static constexpr int kNumThreads = NumWarps * 32;
    static constexpr int kTargetCtasPerSm = TargetCtasPerSm;
};

template <int BM = 128, int BN = 128, int BK = 32, int Stages = 3>
using GemmConfig4Warp = GemmConfig<BM, BN, BK, Stages, 2, 2, 2>;

template <int BM = 128, int BN = 128, int BK = 64, int Stages = 3>
using GemmConfig8Warp = GemmConfig<BM, BN, BK, Stages, 4, 2, 2>;

}  // namespace mk_gemm
