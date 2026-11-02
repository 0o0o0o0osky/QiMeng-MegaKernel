// gemm_config.cuh — Unified GEMM config for A100 (sm_80) bf16
//
// Single template covers all thread counts and warp layouts:
//
//   kWarpM    — warps in M direction
//   kWarpN    — warps in N direction  (kWarpM × kWarpN × 32 = kNumThreads)
//   kAtomRepN — per-warp N-atom repeats (supported: 1 or 2)
//
// execution_config.cuh exposes canonical 4-warp and 8-warp aliases while this
// type retains the layout parameters needed by concrete instances.
//
// All derived quantities auto-computed (no hardcoded 32/64/128 literals):
//   kNumThreads = kWarpM × kWarpN × 32
//   kMmaPM      = kWarpM × 16
//   kMmaPN      = kWarpN × kAtomRepN × 8
//   G2S layout  = (kNumThreads / (BK/8),  BK/8)
//   S2G layout  = (kNumThreads / (kMmaPN/8), kMmaPN/8)
//   SmemLayoutC swizzle derived from kMmaPN
//
// Supported worker shapes contain either four or eight warps.

#pragma once

#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/copy_atom.hpp>

using namespace cute;

namespace mk_gemm {

constexpr int ct_log2(int n) { return (n <= 1) ? 0 : 1 + ct_log2(n / 2); }

// ---------------------------------------------------------------------------
// GemmConfig — 7-parameter flexible GEMM config for sm_80 bf16
// ---------------------------------------------------------------------------
template <
    int BM_,
    int BN_,
    int BK_,            // must be power of 2, ≥ 16
    int kStage_,
    int kWarpM_ = 2,    // warps in M direction
    int kWarpN_ = 2,    // warps in N direction
    int kAtomRepN_ = 2  // per-warp N-atom repeats; per-warp N tile = kAtomRepN × 8
>
struct GemmConfig {
    using T    = nv_bfloat16;
    using AccT = float;

    static constexpr int BM        = BM_;
    static constexpr int BN        = BN_;
    static constexpr int BK        = BK_;
    static constexpr int kStage    = kStage_;

    // BK=32 has two MMA K-fragments, so advance the stage wait one fragment
    // earlier. Deeper K tiles retain the boundary wait after their last fragment.
    static constexpr bool kEarlyWait = (BK_ <= 32);

    static constexpr int kWarpM    = kWarpM_;
    static constexpr int kWarpN    = kWarpN_;
    static constexpr int kAtomRepN = kAtomRepN_;

    // MMA atom: SM80_16x8x16_F32BF16BF16F32_TN
    using mma_op     = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;

    static constexpr int kAtomM = 16;
    static constexpr int kAtomN = 8;
    static constexpr int kAtomK = 16;

    // ── MMA partition tile ───────────────────────────────────────────────────
    // Total tile covered by all EUs (warps) in one TiledMMA invocation:
    //   M: kWarpM warps × 1 atom-M each × 16 = kWarpM × 16
    //   N: kWarpN warps × kAtomRepN atom-N each × 8 = kWarpN × kAtomRepN × 8
    //   K: 1 atom-K × 16 = 16
    static constexpr int kMmaPM = kWarpM * kAtomM;              // e.g. 2×16 = 32
    static constexpr int kMmaPN = kWarpN * kAtomRepN * kAtomN;  // e.g. 2×2×8 = 32
    static constexpr int kMmaPK = kAtomK;                       // = 16

    // Thread count = kWarpM × kWarpN warps × 32 threads/warp
    static constexpr int kNumThreads = kWarpM * kWarpN * 32;

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kWarpM>{}, Int<kWarpN>{}, Int<1>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using TiledMMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    // ── Static asserts ───────────────────────────────────────────────────────
    static_assert(kNumThreads % 32 == 0 && kNumThreads >= 32 && kNumThreads <= 1024,
        "kNumThreads must be a multiple of 32 in [32, 1024]");
    static_assert(kWarpM * kWarpN == 4 || kWarpM * kWarpN == 8,
        "Ampere GEMM worker shapes must contain four or eight warps");
    static_assert(kAtomRepN == 1 || kAtomRepN == 2,
        "kAtomRepN must be one or two");
    static_assert(BK_ >= 16 && (BK_ & (BK_ - 1)) == 0,
        "BK must be a power of 2 and >= 16 (swizzle uses ct_log2)");
    static_assert(kNumThreads % (BK_ / 8) == 0,
        "kNumThreads must be divisible by BK/8 for G2S uint128_t copy layout");
    static_assert(BM_ % kMmaPM == 0,
        "BM must be divisible by kMmaPM (= kWarpM × 16)");
    static_assert(BN_ % kMmaPN == 0,
        "BN must be divisible by kMmaPN (= kWarpN × kAtomRepN × 8)");
    static_assert(kMmaPN % 8 == 0,
        "kMmaPN must be divisible by 8 for S2GCopyC uint128_t alignment");
    static_assert(BM_ * BK_ >= kMmaPM * kMmaPN * 2,
        "SmemLayoutC must fit within one A-pipe stage: BM × BK >= kMmaPM × kMmaPN × 2");

    // ── A/B smem swizzle (derived from BK, same formula as execution_config.cuh) ─
    static constexpr int kSwizzleM  = 3;
    static constexpr int kSwizzleB  = ct_log2(BK_) - kSwizzleM;
    static constexpr int kSwizzleS  = (kSwizzleB > kSwizzleM) ? kSwizzleB : kSwizzleM;

    using SwizzleAB = Swizzle<kSwizzleB, kSwizzleM, kSwizzleS>;

    using SmemLayoutAtomA = decltype(composition(
        SwizzleAB{},
        make_layout(make_shape(Int<8>{}, Int<BK_>{}),
                    make_stride(Int<BK_>{}, Int<1>{}))
    ));
    using SmemLayoutAtomB = decltype(composition(
        SwizzleAB{},
        make_layout(make_shape(Int<8>{}, Int<BK_>{}),
                    make_stride(Int<BK_>{}, Int<1>{}))
    ));
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtomA{},
        make_shape(Int<BM_>{}, Int<BK_>{}, Int<kStage_>{})));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtomB{},
        make_shape(Int<BN_>{}, Int<BK_>{}, Int<kStage_>{})));

    // ── G2S copy: thread layout derived from kNumThreads and BK ─────────────
    // uint128_t load = 8 bf16 per thread; kG2sKThreads threads in K direction.
    static constexpr int kG2sKThreads = BK_ / 8;
    static constexpr int kG2sMThreads = kNumThreads / kG2sKThreads;

    using g2s_copy_op        = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_atom      = Copy_Atom<Copy_Traits<g2s_copy_op>, T>;
    using g2s_zfill_op       = SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>;
    using g2s_copy_atom_zfill = Copy_Atom<Copy_Traits<g2s_zfill_op>, T>;

    using G2SCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<kG2sMThreads>{}, Int<kG2sKThreads>{}),
                    make_stride(Int<kG2sKThreads>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyB = G2SCopyA;

    using G2SCopyA_ZFILL = decltype(make_tiled_copy(
        g2s_copy_atom_zfill{},
        make_layout(make_shape(Int<kG2sMThreads>{}, Int<kG2sKThreads>{}),
                    make_stride(Int<kG2sKThreads>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));
    using G2SCopyB_ZFILL = G2SCopyA_ZFILL;

    // S2R: ldmatrix for A (per-warp A = kAtomM × kAtomK = 16×16 = 256 bf16, always x4)
    using S2RCopyAtomA = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;

    // S2R: ldmatrix for B — variant depends on per-warp B tile size.
    // Per-warp B = kAtomRepN × kAtomN × kAtomK = kAtomRepN × 8 × 16 = kAtomRepN × 128 bf16.
    //   kAtomRepN=1 → 128 bf16/warp → SM75_U32x2_LDSM_N (2 uint32/thread × 32 = 128 bf16)
    //   kAtomRepN≥2 → 256+ bf16/warp → SM75_U32x4_LDSM_N (4 uint32/thread × 32 = 256 bf16)
    using S2RCopyAtomB = std::conditional_t<
        (kAtomRepN_ == 1),
        Copy_Atom<Copy_Traits<SM75_U32x2_LDSM_N>, T>,
        Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>
    >;

    // ── SmemLayoutC: epilogue C staging, aliased on A smem ──────────────────
    // Swizzle derived from kMmaPN (analogous to A/B swizzle from BK).
    // For kMmaPN < 8: kSwizzleCB would be negative — guarded against by static_assert.
    static constexpr int kSmemCBatch = 2;
    static constexpr int kSwizzleCB  = (ct_log2(kMmaPN) >= kSwizzleM)
                                        ? ct_log2(kMmaPN) - kSwizzleM : 0;
    static constexpr int kSwizzleCS  = (kSwizzleCB > kSwizzleM) ? kSwizzleCB : kSwizzleM;

    using SwizzleC = Swizzle<kSwizzleCB, kSwizzleM, kSwizzleCS>;

    using SmemLayoutAtomC = decltype(composition(
        SwizzleC{},
        make_layout(make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
                    make_stride(Int<kMmaPN>{}, Int<1>{}))
    ));
    using SmemLayoutC = decltype(tile_to_shape(
        SmemLayoutAtomC{},
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}, Int<kSmemCBatch>{})
    ));

    // ── S2G copy for C epilogue: thread layout derived from kMmaPN ───────────
    // Differs from G2S: writes along N direction of C tile (not K direction of A/B).
    // Logical S2G work per accumulator tile:
    //   kLogicalS2gVec = kMmaPM x kMmaPN / kNumThreads.
    // Physical store width stays in the existing uint64/uint128 family. Larger
    // logical vectors are represented as multiple physical rounds by the kernel
    // epilogue policy. Existing kLogicalS2gVec=4/8 configs remain single-round.
    // The former fixed 8-wide store silently mis-partitioned any config with
    // kMmaPM×kMmaPN < 8×kNumThreads (all kAtomRepN=1 / kMmaPN=16 variants):
    // the (kNumThreads/(kMmaPN/8)) M-thread layout overflowed the stage's M
    // extent → garbage output (see test_unified.cu "rep1"/"w141" cases).
    static constexpr int kLogicalS2gVec = (kMmaPM * kMmaPN) / kNumThreads;
    static_assert(kLogicalS2gVec >= 4 && (kLogicalS2gVec % 4) == 0,
        "S2G logical per-thread work must be a multiple of 4 bf16");
    static constexpr int kPhysicalS2gVec = (kLogicalS2gVec >= 8) ? 8 : 4;
    static constexpr int kS2gRounds = kLogicalS2gVec / kPhysicalS2gVec;
    static_assert(kS2gRounds >= 1 && kS2gRounds <= 8,
        "S2G multi-round policy expects a small static round count");
    static constexpr int kS2gVec = kPhysicalS2gVec;  // compatibility name for existing code.
    static constexpr int kS2gNThreads = kMmaPN / kPhysicalS2gVec;
    static constexpr int kS2gMThreads = kNumThreads / kS2gNThreads;
    static_assert(kS2gMThreads <= kMmaPM && (kMmaPM % kS2gMThreads) == 0,
        "S2G thread layout must tile a whole-number slice of one C stage");
    static_assert((kMmaPN % kPhysicalS2gVec) == 0,
        "Physical S2G vector must divide kMmaPN");

    using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
    using S2GStoreT = std::conditional_t<(kPhysicalS2gVec == 8), cute::uint128_t, cute::uint64_t>;
    using S2GCopyAtomC = Copy_Atom<UniversalCopy<S2GStoreT>, T>;
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(make_shape(Int<kS2gMThreads>{}, Int<kS2gNThreads>{}),
                    make_stride(Int<kS2gNThreads>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<kPhysicalS2gVec>{}))
    ));

    // ── Smem sizes ───────────────────────────────────────────────────────────
    static constexpr int shm_size_AB =
        (cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{})) * sizeof(T);
    static constexpr int shm_size_C =
        cute::cosize(SmemLayoutC{}) * sizeof(T);
    static constexpr int kShmSize = (shm_size_AB > shm_size_C) ? shm_size_AB : shm_size_C;
};

}  // namespace mk_gemm
