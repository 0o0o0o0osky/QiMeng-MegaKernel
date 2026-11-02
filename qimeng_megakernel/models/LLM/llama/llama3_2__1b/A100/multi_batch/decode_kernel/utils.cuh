#pragma once

#include "llama_config.cuh"
#include "kernel_config.cuh"

#include <algorithm>
#include <cuda_bf16.h>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cute/algorithm/clear.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/arch/copy_sm75.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "profiling.cuh"

#ifndef SPLITKV_KV_STAGES
#define SPLITKV_KV_STAGES 2
#endif

#ifndef DECODE_GEMM_SWIZZLE_B
#define DECODE_GEMM_SWIZZLE_B 3
#endif

#ifndef DECODE_GEMM_SWIZZLE_M
#define DECODE_GEMM_SWIZZLE_M 3
#endif

#ifndef DECODE_GEMM_SWIZZLE_S
#define DECODE_GEMM_SWIZZLE_S 3
#endif

// Swizzle for **MMA C scratch (swizzled plane)** in QKV / UpGate / etc. Buffers B/K operand
// swizzle (DECODE_GEMM_SWIZZLE_*). Using a slightly weaker C swizzle reduces shared **store**
// bank conflicts in Raw→Sw scatter (see NCU l1tex__..._op_st on mk_step_kernel).
#ifndef DECODE_GEMM_C_SWIZZLE_B
#define DECODE_GEMM_C_SWIZZLE_B 2
#endif
#ifndef DECODE_GEMM_C_SWIZZLE_M
#define DECODE_GEMM_C_SWIZZLE_M 2
#endif
#ifndef DECODE_GEMM_C_SWIZZLE_S
#define DECODE_GEMM_C_SWIZZLE_S 2
#endif

#ifndef DECODE_GEMM_STAGE_COUNT
#define DECODE_GEMM_STAGE_COUNT 5
#endif

#ifndef DECODE_GEMM_TILE_K
#define DECODE_GEMM_TILE_K 64
#endif

#ifndef DECODE_GEMM_MMA_REPEAT_M
#define DECODE_GEMM_MMA_REPEAT_M 1
#endif

#ifndef DECODE_GEMM_MMA_REPEAT_K
#define DECODE_GEMM_MMA_REPEAT_K 1
#endif

#ifndef DECODE_QKV_STAGED_A
#define DECODE_QKV_STAGED_A 0
#endif
#ifndef DECODE_OPROJ_STAGED_A
#define DECODE_OPROJ_STAGED_A 0
#endif
#ifndef DECODE_UPGATE_STAGED_A
#define DECODE_UPGATE_STAGED_A 0
#endif
#ifndef DECODE_LMHEAD_STAGED_A
#define DECODE_LMHEAD_STAGED_A 0
#endif

namespace megakernel {

using namespace cute;
using bfloat16_t = cutlass::bfloat16_t;

static_assert(DECODE_QKV_STAGED_A == 0 || DECODE_QKV_STAGED_A == 1);
static_assert(DECODE_OPROJ_STAGED_A == 0 || DECODE_OPROJ_STAGED_A == 1);
static_assert(DECODE_UPGATE_STAGED_A == 0 || DECODE_UPGATE_STAGED_A == 1);
static_assert(DECODE_LMHEAD_STAGED_A == 0 || DECODE_LMHEAD_STAGED_A == 1);

inline constexpr bool kQkvStagedA = DECODE_QKV_STAGED_A != 0;
inline constexpr bool kOprojStagedA = DECODE_OPROJ_STAGED_A != 0;
inline constexpr bool kUpgateStagedA = DECODE_UPGATE_STAGED_A != 0;
inline constexpr bool kLmheadStagedA = DECODE_LMHEAD_STAGED_A != 0;

// =============================================================================
// 1. Basic warp/block reduction helpers
// =============================================================================

template <typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    static_assert(sizeof(T) == 4, "warp_reduce_sum only supports 32-bit types.");
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

// =============================================================================
// 1b. Block reduce (same pattern as single_batch decode_kernel; RMS uses fp32)
// =============================================================================
template <int NumWarps, typename T, typename TensorSmem>
__device__ __forceinline__ T block_reduce_sum(T val, TensorSmem& t_smem) {
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    T warp_sum = warp_reduce_sum(val);

    if (lane_id == 0) {
        t_smem(warp_id) = warp_sum;
    }
    __syncthreads();

    T block_sum = T(0);
    if (warp_id == 0) {
        if (lane_id < NumWarps) {
            block_sum = t_smem(lane_id);
        }
        block_sum = warp_reduce_sum(block_sum);
    }
    if (tid == 0) {
        t_smem(0) = block_sum;
    }
    __syncthreads();

    return t_smem(0);
}

// =============================================================================
// 2. RoPE helper — interleaved rotation used for Q and K
// =============================================================================

template <int TileN>
__device__ __forceinline__ float apply_rope_interleaved(
    float val, float cos_val, float sin_val, int lane_id
) {
    static_assert(TileN > 0 && TileN <= 32, "TileN must be in (0, 32]");
    static_assert(TileN % 2 == 0, "TileN must be even for RoPE");
    constexpr unsigned mask = (TileN == 32) ? 0xffffffffu : (1u << TileN) - 1u;
    float neighbor = __shfl_xor_sync(mask, val, 1);
    float sign     = (lane_id % 2 == 0) ? -1.0f : 1.0f;
    return val * cos_val + sign * neighbor * sin_val;
}

// =============================================================================
// 3. Cross-CTA publication (SM80 GPU-scope release/publish and acquire/wait)
// =============================================================================

#define BARRIER_SLEEP_NANOS KernelConfig::GMEM_SPIN_LOOP_SLEEP_NANOS

__device__ __forceinline__ uint32_t barrier_load_relaxed(
    const volatile int32_t* barrier_ptr
) {
    uint32_t value;
    asm volatile(
        "ld.relaxed.gpu.global.u32 %0, [%1];"
        : "=r"(value)
        : "l"(barrier_ptr)
        : "memory"
    );
    return value;
}

__device__ __forceinline__ uint32_t barrier_load_acquire(
    const volatile int32_t* barrier_ptr
) {
    uint32_t value;
    asm volatile(
        "ld.acquire.gpu.global.u32 %0, [%1];"
        : "=r"(value)
        : "l"(barrier_ptr)
        : "memory"
    );
    return value;
}

__device__ __forceinline__ void barrier_wait(
    volatile int32_t* barrier_ptr,
    int expected_count,
    int sm_id,
    int opcode,
    int check_idx,
    int layer_idx = -1,
    LlamaGlobals* g = nullptr,
    int inst_idx = -1
) {
#ifdef ENABLE_PROFILING
    uint64_t wait_start = 0;
    if (threadIdx.x == 0 && g != nullptr && inst_idx >= 0) {
        wait_start = prof_clock_now();
    }
#endif
    const uint32_t expected = static_cast<uint32_t>(expected_count);
    while (barrier_load_relaxed(barrier_ptr) < expected) {
        __nanosleep(BARRIER_SLEEP_NANOS);
    }
    while (barrier_load_acquire(barrier_ptr) < expected) {
        __nanosleep(BARRIER_SLEEP_NANOS);
    }
#ifdef ENABLE_PROFILING
    if (threadIdx.x == 0 && g != nullptr && inst_idx >= 0) {
        uint64_t wait_end = prof_clock_now();
        uint64_t dt = wait_end - wait_start;
        PROF_ADD_WAIT_CYCLES((*g), sm_id, inst_idx, dt);
        PROF_ADD_STAGE_CYCLES((*g), sm_id, inst_idx, PROF_FIELD_BARRIER_SPIN, dt);
    }
#endif
}

__device__ __forceinline__ void barrier_publish(
    volatile int32_t* barrier_ptr,
    int increment
) {
    asm volatile(
        "red.release.gpu.global.add.u32 [%0], %1;"
        :
        : "l"(barrier_ptr), "r"(static_cast<uint32_t>(increment))
        : "memory"
    );
}

// =============================================================================
// 4. GemmSharedBase — common types shared by 1-tile and 2-tile configs
//
// Pipeline-specific tuning (K-tile, stages, swizzle, MMA EU repeat M/K): edit
// the constexprs below. Whole-CTA limits remain in KernelConfig.
// =============================================================================

struct GemmSharedBase {
    using T = bfloat16_t;

    // ── Swizzle parameters ─────────────────────────────────────────────────
    // SM80 operand swizzle selected for this instance.
    static constexpr int kShmLoadSwizzleB = DECODE_GEMM_SWIZZLE_B;
    static constexpr int kShmLoadSwizzleM = DECODE_GEMM_SWIZZLE_M;
    static constexpr int kShmLoadSwizzleS = DECODE_GEMM_SWIZZLE_S;

    // ── Pipeline depth ─────────────────────────────────────────────────────
    static constexpr int kStage = DECODE_GEMM_STAGE_COUNT;

    // ── K-tile granularity ─────────────────────────────────────────────────
    // Minimum: kTileK >= 2^(B+M) elements for Swizzle<B,M,S> + this dtype.
    static constexpr int kTileK = DECODE_GEMM_TILE_K;

    // ── MMA EU repeat (M, K); N is per GemmConfig{1,2}Tile ─────────────────
    static constexpr int kMmaEURepeatM = DECODE_GEMM_MMA_REPEAT_M;
    static constexpr int kMmaEURepeatK = DECODE_GEMM_MMA_REPEAT_K;

    static_assert(
        kTileK >= (1 << (kShmLoadSwizzleB + kShmLoadSwizzleM)),
        "kTileK too small for Swizzle<B,M,S>: need kTileK >= 2^(B+M) (elements)");

    // SwizzleAtom: 8 rows × kTileK columns, row-major with Swizzle applied.
    using SmemLayoutAtom = decltype(composition(
        Swizzle<kShmLoadSwizzleB, kShmLoadSwizzleM, kShmLoadSwizzleS>{},
        make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
                    make_stride(Int<kTileK>{}, Int<1>{}))
    ));

    // A smem: full normalized activation [kTileM, HIDDEN_DIM].
    // Resident-A GEMMs populate this layout from the route-owned RMS scratch.
    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<KernelConfig::kTileM>{}, Int<LlamaConfig::HIDDEN_DIM>{})
    ));

    template <int TileCount>
    using SmemLayoutATileRing = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(
            Int<KernelConfig::kTileM>{},
            Int<kTileK>{},
            Int<TileCount>{})
    ));

    using SmemLayoutAStaged = SmemLayoutATileRing<kStage>;

    // G2S: cp.async caching-global, 128-bit (16 bytes) per thread.
    using g2s_copy_op    = SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>;
    using g2s_copy_atom  = Copy_Atom<Copy_Traits<g2s_copy_op>, T>;

    // G2SCopyB: 128 threads cover one 16×kTileK B-tile per call.
    //   Thread layout (16, 8): 16 N-rows × 8 K-groups of threads.
    //   Val layout   (1,  8): each thread loads 8 consecutive bf16 elements.
    //   Coverage per call: (16, kTileK=64) elements = 2048 bytes.
    //   For 2-tile config (kTileN=32): same layout, 2 reps in N.
    using G2SCopyB = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<16>{}, Int<8>{}),
                    make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // S2R: ldmatrix.x4 — loads 4 matrix fragments (128 bits per thread).
    // TN GEMM: A is M×K, B is N×K in smem (B swizzled on the 8×kTileK atom). Both operands use
    // SM75_U32x4_LDSM_N with make_tiled_copy_{A,B}(..., tiled_mma), matching CuTe TN + swizzled B.
    // Transposed B (K-major in smem for a different layout) would use SM75_U16x8_LDSM_T (see
    // SplitKV S2RCopyAtomB_Trans below); that is **not** interchangeable here without a separate
    // GemmCfg fork — OProj/Downproj reuse this base.
    using s2r_copy_atom = Copy_Atom<Copy_Traits<SM75_U32x4_LDSM_N>, T>;
    using S2RCopyAtomA  = s2r_copy_atom;
    using S2RCopyAtomB  = s2r_copy_atom;

    // Epilogue R2S / S2G.
    using R2SCopyAtomC  = Copy_Atom<UniversalCopy<int>,       T>;
    using S2GCopyAtomC  = Copy_Atom<UniversalCopy<uint128_t>, T>;

    // S2GCopyC: 128-bit vectorized store to global, same thread layout as G2SCopyB.
    using S2GCopyC = decltype(make_tiled_copy(
        S2GCopyAtomC{},
        make_layout(make_shape(Int<16>{}, Int<8>{}),
                    make_stride(Int<8>{}, Int<1>{})),
        make_layout(make_shape(Int<1>{}, Int<8>{}))
    ));

    // QKV epilogue Raw→Sw: same tiled atom/layout as S2GCopyC for vectorized Raw→reg;
    // Swizzled dst uses logical sW(m,n) stores (ComposedLayout + uint128 retile_D fails).
    using SmemCSmemCopyC = S2GCopyC;
};

template <int Rows, int Cols>
using RowMajorTileLayout = decltype(make_layout(
    make_shape(Int<Rows>{}, Int<Cols>{}),
    LayoutRight{}));

template <int Rows, int Cols>
struct SharedOutputScratchLayout;

template <int Rows>
struct SharedOutputScratchLayout<Rows, 16> {
    using type = decltype(composition(
        Swizzle<1, 3, 3>{},
        make_layout(
            make_shape(Int<Rows>{}, Int<16>{}),
            make_stride(Int<16>{}, Int<1>{}))));
};

template <int Rows>
struct SharedOutputScratchLayout<Rows, 32> {
    using type = decltype(composition(
        Swizzle<2, 3, 3>{},
        make_layout(
            make_shape(Int<Rows>{}, Int<32>{}),
            make_stride(Int<32>{}, Int<1>{}))));
};

template <int Rows>
struct SharedOutputScratchLayout<Rows, 64> {
    using type = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(
            make_shape(Int<Rows>{}, Int<64>{}),
            make_stride(Int<64>{}, Int<1>{}))));
};

template <int Rows>
struct SharedOutputScratchLayout<Rows, 128> {
    using type = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(
            make_shape(Int<Rows>{}, Int<128>{}),
            make_stride(Int<128>{}, Int<1>{}))));
};

template <int Rows, int Cols>
using SharedOutputScratchLayoutT =
    typename SharedOutputScratchLayout<Rows, Cols>::type;

// =============================================================================
// 5. GemmConfig1Tile  — 1 output N-tile (kTileN=16), 1 MMA warp
//
//   kMmaEURepeatN=1 → size(MMA{})=32; warp 0 performs MMA.
//   N_REG_REPEAT=2  → two 16×8 atoms per warp → effective 16×16 N output.
//   kMmaPM=16, kMmaPN=16, kMmaPK=16.
// =============================================================================

struct GemmConfig1Tile : GemmSharedBase {
    static constexpr int kTileN_       = KernelConfig::kTileN;  // 16
    static constexpr int kMmaEURepeatN = 1;

    static_assert(
        (kTileN_ * kTileK) % (16 * 8 * 8) == 0,
        "kTileN * kTileK must be divisible by G2S thread-val coverage (1024 elements)");
    static_assert(
        (KernelConfig::kTileM * kTileN_) % KernelConfig::NUM_THREADS == 0,
        "kTileM * kTileN must be divisible by NUM_THREADS for epilogue scatter");

    using mma_op     = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;
    using mma_atom_shape = mma_traits::Shape_MNK;  // (16, 8, 16)

    // N_REG_REPEAT=2: each warp fires two 16×8 MMA atoms → 16×16 effective N.
    static constexpr int N_REG_REPEAT = 2;  // factor-2 in kMmaPN formula
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});             // 16
    static constexpr int kMmaPN = N_REG_REPEAT * kMmaEURepeatN * get<1>(mma_atom_shape{});  // 16
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});             // 16

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA     = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
    // size(MMA{}) = 1*1*32 = 32 (warp 0 only)

    // B smem: [kTileN, kTileK, kStage]
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileN_>{}, Int<kTileK>{}, Int<kStage>{})
    ));

    // C scratch: MMA writes compact row-major (Raw); QKV epilogue copies to Swizzled plane (see rms_qkv).
    using SmemLayoutCRaw = RowMajorTileLayout<kMmaPM, kMmaPN>;
    // kMmaPN=16 is too small for Swizzle<3,3,3>; use B=1.
    using SmemLayoutC = decltype(composition(
        Swizzle<1, 2, 1>{},
        make_layout(
            make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
            make_stride(Int<kMmaPN>{}, Int<1>{}))));
};

// =============================================================================
// 6. GemmConfig2Tile  — 2 output N-tiles (kTileN=32), 2 MMA warps
//
//   kMmaEURepeatN=2 → size(MMA{})=64; warps 0 and 1 perform MMA in parallel.
//   N_REG_REPEAT=2  → each warp covers 2×8 = 16 N columns → total 32 N.
//   kMmaPM=16, kMmaPN=32, kMmaPK=16.
// =============================================================================

struct GemmConfig2Tile : GemmSharedBase {
    static constexpr int kTileN_       = KernelConfig::kTileN * 2;  // 32
    static constexpr int kMmaEURepeatN = 2;  // 2 warps in N

    static_assert(
        (kTileN_ * kTileK) % (16 * 8 * 8) == 0,
        "kTileN * kTileK must be divisible by G2S thread-val coverage (1024 elements)");
    static_assert(
        (KernelConfig::kTileM * kTileN_) % KernelConfig::NUM_THREADS == 0,
        "kTileM * kTileN must be divisible by NUM_THREADS for epilogue scatter");

    using mma_op     = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;
    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int N_REG_REPEAT = 2;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});             // 16
    static constexpr int kMmaPN = N_REG_REPEAT * kMmaEURepeatN * get<1>(mma_atom_shape{});  // 32
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});             // 16

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA     = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));
    // size(MMA{}) = 1*2*32 = 64 (warps 0 and 1)

    // B smem: [kTileN, kTileK, kStage]
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileN_>{}, Int<kTileK>{}, Int<kStage>{})
    ));

    using SmemLayoutCRaw = RowMajorTileLayout<kMmaPM, kMmaPN>;
    using SmemLayoutC = decltype(composition(
        Swizzle<DECODE_GEMM_C_SWIZZLE_B, DECODE_GEMM_C_SWIZZLE_M, DECODE_GEMM_C_SWIZZLE_S>{},
        make_layout(
            make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
            make_stride(Int<kMmaPN>{}, Int<1>{}))));
};

// =============================================================================
// 6a. GemmConfig4Tile  — 4 output N-tiles (kTileN=64), 4 MMA warps
// =============================================================================
struct GemmConfig4Tile : GemmSharedBase {
    static constexpr int kTileN_       = KernelConfig::kTileN * 4;  // 64
    static constexpr int kMmaEURepeatN = 4;

    static_assert(
        (kTileN_ * kTileK) % (16 * 8 * 8) == 0,
        "kTileN * kTileK must be divisible by G2S thread-val coverage (1024 elements)");
    static_assert(
        (KernelConfig::kTileM * kTileN_) % KernelConfig::NUM_THREADS == 0,
        "kTileM * kTileN must be divisible by NUM_THREADS for epilogue scatter");

    using mma_op     = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;
    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int N_REG_REPEAT = 2;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = N_REG_REPEAT * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA     = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileN_>{}, Int<kTileK>{}, Int<kStage>{})
    ));

    using SmemLayoutCRaw = RowMajorTileLayout<kMmaPM, kMmaPN>;
    using SmemLayoutC = decltype(composition(
        Swizzle<DECODE_GEMM_C_SWIZZLE_B, DECODE_GEMM_C_SWIZZLE_M, DECODE_GEMM_C_SWIZZLE_S>{},
        make_layout(
            make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
            make_stride(Int<kMmaPN>{}, Int<1>{}))));
};

// =============================================================================
// 6b. GemmConfig8Tile — 8 output N-tiles (kTileN=128), 8 MMA warps
// =============================================================================
struct GemmConfig8Tile : GemmSharedBase {
    static constexpr int kTileN_       = KernelConfig::kTileN * 8;
    static constexpr int kMmaEURepeatN = 8;

    static_assert(
        (kTileN_ * kTileK) % (16 * 8 * 8) == 0,
        "kTileN * kTileK must be divisible by G2S thread-val coverage");
    static_assert(
        (KernelConfig::kTileM * kTileN_) % KernelConfig::NUM_THREADS == 0,
        "kTileM * kTileN must be divisible by NUM_THREADS for epilogue scatter");

    using mma_op     = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom   = MMA_Atom<mma_traits>;
    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int N_REG_REPEAT = 2;
    static constexpr int kMmaPM = kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = N_REG_REPEAT * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileN_>{}, Int<kTileK>{}, Int<kStage>{})));
    using SmemLayoutCRaw = RowMajorTileLayout<kMmaPM, kMmaPN>;
    using SmemLayoutC = decltype(composition(
        Swizzle<3, 3, 3>{},
        make_layout(
            make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
            make_stride(Int<kMmaPN>{}, Int<1>{}))));
};

// =============================================================================
// 6c. GEMM-family shared-memory layout helpers
// =============================================================================

static constexpr ptrdiff_t kGemmSmemAlignBytes = 128;

CUTE_HOST_DEVICE constexpr ptrdiff_t align_up_smem_bytes(
    ptrdiff_t offset,
    ptrdiff_t alignment = kGemmSmemAlignBytes)
{
    return ((offset + alignment - 1) / alignment) * alignment;
}

template <typename Layout>
inline constexpr ptrdiff_t kSmemLayoutElements =
    static_cast<ptrdiff_t>(decltype(cosize(Layout{}))::value);

template <typename Element, typename Layout>
inline constexpr ptrdiff_t kSmemLayoutBytes =
    kSmemLayoutElements<Layout> * static_cast<ptrdiff_t>(sizeof(Element));

template <typename Element, int Count>
inline constexpr ptrdiff_t kSmemArrayBytes =
    static_cast<ptrdiff_t>(Count) * static_cast<ptrdiff_t>(sizeof(Element));

template <typename T>
CUTE_HOST_DEVICE T* smem_ptr_at(
    uint8_t* smem_base,
    ptrdiff_t offset_bytes)
{
    return reinterpret_cast<T*>(smem_base + offset_bytes);
}

template <typename T>
CUTE_HOST_DEVICE const T* smem_ptr_at(
    const uint8_t* smem_base,
    ptrdiff_t offset_bytes)
{
    return reinterpret_cast<const T*>(smem_base + offset_bytes);
}

using WideGemmConfig = std::conditional_t<
    KernelConfig::NUM_WARPS == 8,
    GemmConfig8Tile,
    GemmConfig4Tile>;

template <bool Staged>
using ActivationSmemLayout = std::conditional_t<
    Staged,
    GemmSharedBase::SmemLayoutAStaged,
    GemmSharedBase::SmemLayoutA>;

struct OprojGemmSmemLayout {
    using SmemLayoutA = ActivationSmemLayout<kOprojStagedA>;
    using SmemLayoutB = typename WideGemmConfig::SmemLayoutB;
    // C scratch: MMA + epilogue use SmemLayoutCRaw (same cosize as swizzled; see QkvGemmSmemLayout).
    using SmemLayoutCScratch = typename WideGemmConfig::SmemLayoutCRaw;

    static_assert(
        kSmemLayoutBytes<bfloat16_t, typename WideGemmConfig::SmemLayoutCRaw> ==
        kSmemLayoutBytes<bfloat16_t, typename WideGemmConfig::SmemLayoutC>);

    static constexpr ptrdiff_t kOffsetA = 0;
    static constexpr ptrdiff_t kOffsetResidentA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedB = align_up_smem_bytes(
        kOffsetA + kSmemLayoutBytes<bfloat16_t, SmemLayoutA>);
    static constexpr ptrdiff_t kOffsetOutputScratch = align_up_smem_bytes(
        kOffsetStagedB + kSmemLayoutBytes<bfloat16_t, SmemLayoutB>);
    static constexpr ptrdiff_t kTotalBytes = align_up_smem_bytes(
        kOffsetOutputScratch + kSmemLayoutBytes<bfloat16_t, SmemLayoutCScratch>);

    static_assert(kTotalBytes <= KernelConfig::SHARED_MEMORY_SIZE, "oproj smem overflow");
};

struct UpgateGemmSmemLayout {
    using SmemLayoutA = ActivationSmemLayout<kUpgateStagedA>;
    using SmemLayoutB = GemmConfig2Tile::SmemLayoutB;
    using SmemLayoutOut = RowMajorTileLayout<KernelConfig::kTileM, GemmConfig2Tile::kTileN_>;

    static constexpr ptrdiff_t kOffsetA = 0;
    static constexpr ptrdiff_t kOffsetResidentA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedBUp = align_up_smem_bytes(
        kOffsetA + kSmemLayoutBytes<bfloat16_t, SmemLayoutA>);
    static constexpr ptrdiff_t kOffsetStagedBGate = align_up_smem_bytes(
        kOffsetStagedBUp + kSmemLayoutBytes<bfloat16_t, SmemLayoutB>);
    static constexpr ptrdiff_t kOffsetUpScratch = align_up_smem_bytes(
        kOffsetStagedBGate + kSmemLayoutBytes<bfloat16_t, SmemLayoutB>);
    static constexpr ptrdiff_t kOffsetGateScratch = align_up_smem_bytes(
        kOffsetUpScratch + kSmemLayoutBytes<float, SmemLayoutOut>);
    static constexpr ptrdiff_t kOffsetReduceWorkspace = align_up_smem_bytes(
        kOffsetGateScratch + kSmemLayoutBytes<float, SmemLayoutOut>);
    static constexpr ptrdiff_t kTotalBytes = align_up_smem_bytes(
        kOffsetReduceWorkspace + kSmemArrayBytes<float, KernelConfig::NUM_THREADS>);

    static_assert(kTotalBytes <= KernelConfig::SHARED_MEMORY_SIZE, "upgate smem overflow");
};

struct LmheadGemmSmemLayout {
    using SmemLayoutA = ActivationSmemLayout<kLmheadStagedA>;
    using SmemLayoutBMax = typename WideGemmConfig::SmemLayoutB;
    using SmemLayoutOutMax = SharedOutputScratchLayoutT<
        KernelConfig::kTileM, WideGemmConfig::kTileN_>;

    static constexpr ptrdiff_t kOffsetA = 0;
    static constexpr ptrdiff_t kOffsetResidentA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedB = align_up_smem_bytes(
        kOffsetA + kSmemLayoutBytes<bfloat16_t, SmemLayoutA>);
    static constexpr ptrdiff_t kOffsetOutputScratch = align_up_smem_bytes(
        kOffsetStagedB + kSmemLayoutBytes<bfloat16_t, SmemLayoutBMax>);
    static constexpr ptrdiff_t kOffsetReduceWorkspace = align_up_smem_bytes(
        kOffsetOutputScratch + kSmemLayoutBytes<float, SmemLayoutOutMax>);
    static constexpr ptrdiff_t kTotalBytes = align_up_smem_bytes(
        kOffsetReduceWorkspace + kSmemArrayBytes<float, KernelConfig::NUM_THREADS>);

    static_assert(kTotalBytes <= KernelConfig::SHARED_MEMORY_SIZE, "lmhead smem overflow");
};

struct DownprojGemmSmemLayout {
    using SmemLayoutA = decltype(tile_to_shape(
        GemmSharedBase::SmemLayoutAtom{},
        make_shape(Int<KernelConfig::kTileM>{}, Int<GemmSharedBase::kTileK>{}, Int<GemmSharedBase::kStage>{})
    ));
    using SmemLayoutB = typename WideGemmConfig::SmemLayoutB;
    using SmemLayoutOut = RowMajorTileLayout<
        KernelConfig::kTileM, WideGemmConfig::kTileN_>;

    static constexpr ptrdiff_t kOffsetStagedA = 0;
    static constexpr ptrdiff_t kOffsetStagedB = align_up_smem_bytes(
        kOffsetStagedA + kSmemLayoutBytes<bfloat16_t, SmemLayoutA>);
    static constexpr ptrdiff_t kOffsetOutputScratch = align_up_smem_bytes(
        kOffsetStagedB + kSmemLayoutBytes<bfloat16_t, SmemLayoutB>);
    static constexpr ptrdiff_t kTotalBytes = align_up_smem_bytes(
        kOffsetOutputScratch + kSmemLayoutBytes<float, SmemLayoutOut>);

    static_assert(kTotalBytes <= KernelConfig::SHARED_MEMORY_SIZE, "downproj smem overflow");
};

struct QkvGemmSmemLayout {
    using SmemLayoutA = ActivationSmemLayout<kQkvStagedA>;
    using SmemLayoutB = typename WideGemmConfig::SmemLayoutB;

    // Single Raw C plane: MMA partition_C stores row-major; Step2 loads vectorized from the same
    // plane (no Raw->Sw copy). kScratchCRawBytesMax is the union over 1/2/4-tile configs so that
    // all dispatch paths fit in the same slot.
    static constexpr ptrdiff_t kScratchCRawBytesFourWarps = std::max(
        kSmemLayoutBytes<bfloat16_t, typename GemmConfig1Tile::SmemLayoutCRaw>,
        std::max(
            kSmemLayoutBytes<bfloat16_t, typename GemmConfig2Tile::SmemLayoutCRaw>,
            kSmemLayoutBytes<bfloat16_t, typename GemmConfig4Tile::SmemLayoutCRaw>));
    static constexpr ptrdiff_t kScratchCRawBytesMax = std::max(
        kScratchCRawBytesFourWarps,
        kSmemLayoutBytes<bfloat16_t, typename WideGemmConfig::SmemLayoutCRaw>);

    static constexpr ptrdiff_t kOffsetA = 0;
    static constexpr ptrdiff_t kOffsetResidentA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedA = kOffsetA;
    static constexpr ptrdiff_t kOffsetStagedB = align_up_smem_bytes(
        kOffsetA + kSmemLayoutBytes<bfloat16_t, SmemLayoutA>);
    static constexpr ptrdiff_t kOffsetOutputScratch = align_up_smem_bytes(
        kOffsetStagedB + kSmemLayoutBytes<bfloat16_t, SmemLayoutB>);
    static constexpr ptrdiff_t kOffsetReduceWorkspace = align_up_smem_bytes(
        kOffsetOutputScratch + kScratchCRawBytesMax);
    static constexpr ptrdiff_t kTotalBytes = align_up_smem_bytes(
        kOffsetReduceWorkspace + kSmemArrayBytes<float, KernelConfig::NUM_THREADS>);

    static_assert(kTotalBytes <= KernelConfig::SHARED_MEMORY_SIZE, "qkv smem overflow");
};

// =============================================================================
// 6c. AttnScoreGemmConfig — SplitKV score: Q[16,64] x K^T -> S[16,64]
//
//   Four warps tile the N=64 dimension of the score (KV token axis), matching
//   FlashAttention-2 "warps along N" for the QK^T matmul (see docs/en/
//   splitkv_attention_layout_and_references.md).
//   size(MMA{}) = 4 * 32 = 128 == KernelConfig::NUM_THREADS.
//
//   Smem A/B are **plain row-major** tensors matching decode_attention_splitkv
//   arrays smem_Q[16][64] and smem_K[2][64][64] so strided KV cp.async loads
//   land in the layout CuTe's S2R + TN MMA expect. (Swizzled RMS-style atoms are
//   avoided here: duplicating K for a swizzle staging buffer would overflow the
//   160 KB shared budget.)
//
//   B operand for K^T: transposed ldmatrix — CuTe exposes this as
//   Copy_Traits<SM75_U16x8_LDSM_T> (x4.trans.m8n8), not a U32x4 "_T" trait.
// =============================================================================

struct AttnScoreGemmConfig : GemmSharedBase {
    static constexpr int kTileMScore = KernelConfig::kTileM;  // 16  (padded Q rows)
    static constexpr int kTileNScore = 64;   // KV tile width (= multi_batch attn_kv_block)
    static constexpr int kKVStage    = 2;    // double buffer for K tiles

    static_assert(
        kTileMScore * kTileNScore % KernelConfig::NUM_THREADS == 0,
        "");

    using mma_op         = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits     = MMA_Traits<mma_op>;
    using mma_atom       = MMA_Atom<mma_traits>;
    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int N_REG_REPEAT = 2;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = N_REG_REPEAT * 4 * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<4>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T        = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA            = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    // Physical order matches C arrays: Q[row][dim], K[stage][tok][dim].
    using SmemLayoutA = decltype(make_layout(
        make_shape(Int<kTileMScore>{}, Int<kTileK>{}),
        make_stride(Int<kTileK>{}, Int<1>{})
    ));
    using SmemLayoutB = decltype(make_layout(
        make_shape(Int<kKVStage>{}, Int<kTileNScore>{}, Int<kTileK>{}),
        make_stride(
            Int<kTileNScore * kTileK>{},
            Int<kTileK>{},
            Int<1>{})
    ));
    using SmemLayoutC = decltype(make_layout(
        make_shape(Int<kMmaPM>{}, Int<kMmaPN>{}),
        LayoutRight{}
    ));

    using S2RCopyAtomA = s2r_copy_atom;
    // PV / V operand (and CuTe examples): transposed ldmatrix.x4 — see copy_sm75.hpp.
    using S2RCopyAtomB_Trans = Copy_Atom<Copy_Traits<SM75_U16x8_LDSM_T>, T>;
    // FlashAttention-2 score path uses **non-transposed** U32x4 LDSM for K with **swizzled**
    // SmemLayoutKV — different from a plain [tok][dim] layout. Use this atom when
    // K is rewritten into swizzled smem and TiledMMA matches FA ValLayoutMNK.
    using S2RCopyAtomB_ScorePlainK_AsFA = s2r_copy_atom;

    // G2S for one (16 x kTileK) stripe (e.g. Q from gmem); KV tile uses strided cp.async.
    using G2SQ = G2SCopyB;
};

// PV GEMM config: keep N=64 output per warp-group, split K (64) across 4 warps.
// Each warp contributes one K slice to the SplitKV PV accumulation.
struct AttnPvGemmConfig : GemmSharedBase {
    using mma_op         = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits     = MMA_Traits<mma_op>;
    using mma_atom       = MMA_Atom<mma_traits>;
    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int N_REG_REPEAT = 8;  // 8 * 8 = 64 output cols
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});  // 16
    static constexpr int kMmaPN = N_REG_REPEAT * 1 * get<1>(mma_atom_shape{});   // 64
    static constexpr int kMmaPK = 4 * kMmaEURepeatK * get<2>(mma_atom_shape{});  // 64

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<1>{}, Int<4>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA     = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using S2RCopyAtomA      = s2r_copy_atom;
    using S2RCopyAtomB_Trans = Copy_Atom<Copy_Traits<SM75_U16x8_LDSM_T>, T>;
};

// =============================================================================
// 6d. One-row RMS to the route-owned global scratch.
//
// The RMS producer must not borrow the GEMM operand overlay. A full resident-A
// layout scatters even one logical row across most of the 64KB address range,
// which aliases staged-A/B rings in multi-CTA profiles. Each lane therefore
// keeps its input slice in registers and stores the normalized slice directly.
// =============================================================================
template <int NumWarps, typename OutTensor>
__device__ __forceinline__ void rms_norm_one_row_to_gmem(
    const bfloat16_t* __restrict__ row_in,
    const bfloat16_t* __restrict__ weight,
    float eps,
    float* reduce_ws,
    OutTensor& out,
    int batch_row,
    bool active)
{
    constexpr int HiddenDim = LlamaConfig::HIDDEN_DIM;
    constexpr int NumThreads = NumWarps * 32;
    constexpr int ElemsPerThread = HiddenDim / NumThreads;
    static_assert(HiddenDim % NumThreads == 0);

    const int col0 = int(threadIdx.x) * ElemsPerThread;
    float sum_sq = 0.0f;
    bfloat16_t regs_in[ElemsPerThread];
    bfloat16_t regs_w[ElemsPerThread];
    #pragma unroll
    for (int i = 0; i < ElemsPerThread; ++i) {
        regs_in[i] = row_in[col0 + i];
        regs_w[i] = weight[col0 + i];
        float x = static_cast<float>(regs_in[i]);
        sum_sq += x * x;
    }

    auto t_reduce = make_tensor(
        make_smem_ptr(reduce_ws),
        make_shape(Int<NumWarps>{}),
        LayoutRight{});
    float total_sum_sq = block_reduce_sum<NumWarps>(sum_sq, t_reduce);
    float rms_scale = rsqrtf((total_sum_sq / float(HiddenDim)) + eps);

    #pragma unroll
    for (int i = 0; i < ElemsPerThread; ++i) {
        bfloat16_t normalized = active
            ? bfloat16_t(static_cast<float>(regs_in[i]) * rms_scale)
            : bfloat16_t(0.0f);
        float value = active
            ? static_cast<float>(normalized) * static_cast<float>(regs_w[i])
            : 0.0f;
        out(batch_row, col0 + i) = bfloat16_t(value);
    }
}

// Vectorized cp.async from gmem [batch_size, HIDDEN_DIM] -> swizzled smem A tile.
// Thread mapping matches GemmSharedBase::G2SCopyB: thread(tid) owns row = tid/8, k-lane = tid%8.
// Per K-tile of kTileK=64 cols, each thread issues one 128-bit cp.async covering 8 bf16 columns.
// Predication: row_id >= batch_size -> 0-byte cp.async (gmem source addr is kept in-range).
// Caller must invoke cp_async_fence() + cp_async_wait<0>() + __syncthreads() before consuming smem.
template <int NUM_THREADS, int HIDDEN_DIM, typename SmemLayoutA_T>
__device__ __forceinline__
void load_batched_to_smem_a(
    const bfloat16_t* __restrict__ input_g,  // [batch_size, HIDDEN_DIM]
    bfloat16_t*       smem_A_ptr,
    int               batch_size
) {
    static_assert(NUM_THREADS >= 128, "load_batched_to_smem_a requires >= 128 threads");
    constexpr int kTileM = KernelConfig::kTileM;
    constexpr int kTileK = GemmSharedBase::kTileK;
    constexpr int VEC    = 8;   // 8 bf16 = 16 bytes per cp.async
    constexpr int KLanes = kTileK / VEC;     // 8
    constexpr int ActiveThreads = kTileM * KLanes;  // 128
    static_assert(kTileM == 16 && KLanes == 8 && ActiveThreads == 128,
                  "assumed (16 rows x 8 k-lanes) = 128 active threads");
    static_assert(HIDDEN_DIM % kTileK == 0, "HIDDEN_DIM must be a multiple of kTileK");
    constexpr int NTile = HIDDEN_DIM / kTileK;  // 32

    int tid = threadIdx.x;
    if (tid >= ActiveThreads) return;

    int row   = tid / KLanes;       // 0..15 (one row per thread)
    int klane = tid % KLanes;       // 0..7

    auto sA = make_tensor(make_smem_ptr(smem_A_ptr), SmemLayoutA_T{});
    bool row_active = (row < batch_size);
    // Clamp gmem source to row 0 for inactive rows (cp.async 0-byte won't read, but keep addr valid).
    long row_off = (long)(row_active ? row : 0) * HIDDEN_DIM;

    // Do NOT #pragma unroll this loop: unrolling 32 iterations inflates register pressure in the
    // persistent `mk_kernel` (spill jumped 100B -> 400B+). Each iteration is an independent cp.async
    // issue; the hardware already maintains high ILP via the cp.async pipeline.
    for (int ti = 0; ti < NTile; ++ti) {
        int col = ti * kTileK + klane * VEC;
        bfloat16_t* smem_dst = &sA(row, col);
        const bfloat16_t* gmem_src = input_g + row_off + col;
        unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_dst));
        if (row_active) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         :: "r"(smem), "l"(gmem_src));
        } else {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                         :: "r"(smem), "l"(gmem_src), "n"(0));
        }
    }
}

// =============================================================================
// 8. cp.async helpers — used by attention splitkv for double-buffered KV loads
// =============================================================================

// Issue a 16-byte cp.async.cg (cache-global) from global -> shared.
template <int SizeBytes>
__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
    unsigned smem = static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr));
    if constexpr (SizeBytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(smem), "l"(gmem_ptr));
    } else {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :: "r"(smem), "l"(gmem_ptr), "n"(SizeBytes));
    }
}

// Commit the current cp.async group (equivalent to cute::cp_async_fence).
__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

// Wait until at most N cp.async groups are outstanding.
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

// Stage one MxK activation tile into the shared A ring.  The movement mapping
// is deliberately independent of the CTA's MMA width: 128 lanes cover
// 16 rows x 8 vectors, while an 8-warp profile leaves its upper four warps
// available for the wider MMA mapping.
template <int ReductionExtent>
__device__ __forceinline__ void stage_activation_tile_async(
    const bfloat16_t* __restrict__ input,
    bfloat16_t* __restrict__ smem_a,
    int batch_size,
    int k_tile,
    int stage)
{
    constexpr int kVecElems = 8;
    constexpr int kKLanes = GemmSharedBase::kTileK / kVecElems;
    constexpr int kCopyThreads = KernelConfig::kTileM * kKLanes;
    static_assert(kCopyThreads == 128);
    static_assert(ReductionExtent % GemmSharedBase::kTileK == 0);

    const int tid = threadIdx.x;
    if (tid >= kCopyThreads) return;

    const int row = tid / kKLanes;
    const int k_lane = tid % kKLanes;
    const int k0 = k_tile * GemmSharedBase::kTileK + k_lane * kVecElems;
    auto sA = make_tensor(
        make_smem_ptr(smem_a),
        GemmSharedBase::SmemLayoutAStaged{});

    if (row < batch_size) {
        cp_async_cg<16>(
            &sA(row, k_lane * kVecElems, stage),
            input + static_cast<long>(row) * ReductionExtent + k0);
    } else {
        *reinterpret_cast<uint4*>(&sA(row, k_lane * kVecElems, stage)) =
            make_uint4(0, 0, 0, 0);
    }
}

// =============================================================================
// 9. Attention helpers (used by splitkv + reduction)
// =============================================================================

// Inline PTX exp2 (fast, FTZ). Used by splitkv softmax and reduction.
__device__ __forceinline__ float attn_ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Log2-domain sentinels shared by SplitKV partial + reduction paths.
constexpr float kSplitKvLseNonContributingLog2Leq = -79.f;
constexpr float kSplitKvLseMergedEmptyLog2 =
    kSplitKvLseNonContributingLog2Leq - 1.f;

__device__ __forceinline__ bool splitkv_lse_is_contributing(float lse) {
    return isfinite(lse) && lse > kSplitKvLseNonContributingLog2Leq;
}

template <int N>
__device__ __forceinline__ bool splitkv_merge_normalized_o_log2(
    float (&acc_o)[N], float& acc_lse, float const (&curr_o)[N], float curr_lse)
{
    if (!splitkv_lse_is_contributing(curr_lse)) {
        return false;
    }
    if (!splitkv_lse_is_contributing(acc_lse)) {
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            acc_o[i] = curr_o[i];
        }
        acc_lse = curr_lse;
        return true;
    }

    float max_lse = fmaxf(acc_lse, curr_lse);
    float acc_exp = attn_ptx_exp2(acc_lse - max_lse);
    float curr_exp = attn_ptx_exp2(curr_lse - max_lse);
    float denom = acc_exp + curr_exp;
    float acc_scale = acc_exp / denom;
    float curr_scale = curr_exp / denom;

    #pragma unroll
    for (int i = 0; i < N; ++i) {
        acc_o[i] = acc_o[i] * acc_scale + curr_o[i] * curr_scale;
    }
    acc_lse = max_lse + log2f(denom);
    return true;
}

// =============================================================================
// 10. SplitKV layouts and load (moved from splitkv_layouts.cuh, splitkv_load_cute.cuh)
// =============================================================================

struct SplitkvLayoutTraits {
    static constexpr int kKVTile    = 64;
    static constexpr int kHeadDim   = LlamaConfig::HEAD_DIM;
    static constexpr int kGQARatio  = LlamaConfig::NUM_Q_HEADS / LlamaConfig::NUM_KV_HEADS;
    static constexpr int kNThreads  = KernelConfig::NUM_THREADS;
    // Score/PV ownership remains a four-warp layout.  An 8-warp execution
    // profile may use its upper warps for cooperative loads, but must not
    // duplicate the four-warp softmax/MMA ownership map.
    static constexpr int kNWarps    = 4;
    static constexpr int kQRowsPad  = 16;
    static constexpr int kStage     = SPLITKV_KV_STAGES;
    static constexpr int kBlockKSmem = (kHeadDim % 64 == 0) ? 64 : 32;
    static constexpr int kSwizzle   = (kBlockKSmem == 32) ? 2 : 3;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(bfloat16_t);
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
};
static_assert(SplitkvLayoutTraits::kStage >= 2 && SplitkvLayoutTraits::kStage <= 4,
              "SplitKV stage count must stay in [2, 4]");

struct SplitkvSmemLayoutQ {
    using Traits = SplitkvLayoutTraits;
    using Shape  = decltype(make_shape(Int<Traits::kQRowsPad>{}, Int<Traits::kHeadDim>{}));
    using Stride = decltype(make_stride(Int<Traits::kHeadDim>{}, Int<1>{}));
    using Layout = decltype(make_layout(Shape{}, Stride{}));
    static constexpr int kCount = Traits::kQRowsPad * Traits::kHeadDim;
};

struct SplitkvSmemLayoutKV {
    using Traits = SplitkvLayoutTraits;
    using Shape  = decltype(make_shape(
        Int<Traits::kStage>{}, Int<Traits::kKVTile>{}, Int<Traits::kHeadDim>{}));
    using Stride = decltype(make_stride(
        Int<Traits::kKVTile * Traits::kHeadDim>{}, Int<Traits::kHeadDim>{}, Int<1>{}));
    using Layout = decltype(make_layout(Shape{}, Stride{}));
    static constexpr int kCountPerStage = Traits::kKVTile * Traits::kHeadDim;
    static constexpr int kCount         = Traits::kStage * kCountPerStage;
};

using SplitkvSmemLayoutKVPlane = decltype(make_layout(
    make_shape(Int<SplitkvLayoutTraits::kKVTile>{}, Int<SplitkvLayoutTraits::kHeadDim>{}),
    make_stride(Int<SplitkvLayoutTraits::kHeadDim>{}, Int<1>{})));

using SplitkvSmemLayoutAtomKVSwizzled = decltype(composition(
    Swizzle<SplitkvLayoutTraits::kSwizzle, 3, 3>{},
    make_layout(
        make_shape(Int<8>{}, Int<SplitkvLayoutTraits::kBlockKSmem>{}),
        make_stride(Int<SplitkvLayoutTraits::kBlockKSmem>{}, Int<1>{}))));

using SplitkvSmemLayoutKVPlaneSwizzled = decltype(tile_to_shape(
    SplitkvSmemLayoutAtomKVSwizzled{},
    make_shape(Int<SplitkvLayoutTraits::kKVTile>{}, Int<SplitkvLayoutTraits::kHeadDim>{})));

using SplitkvSmemLayoutVtransposed = decltype(composition(
    SplitkvSmemLayoutKVPlaneSwizzled{},
    make_layout(
        make_shape(Int<SplitkvLayoutTraits::kHeadDim>{}, Int<SplitkvLayoutTraits::kKVTile>{}),
        GenRowMajor{})));

// V^T view for P×V gemm (Flash pattern: sVtNoSwizzle). Same data as KVPlane, logical (dim, seq).
using SplitkvSmemLayoutVtransposedNoSwizzle =
    decltype(get_nonswizzle_portion(SplitkvSmemLayoutVtransposed{}));

using SplitkvSmemLayoutPAtomSwizzled = decltype(composition(
    Swizzle<SplitkvLayoutTraits::kSwizzle, 3, 3>{},
    make_layout(
        make_shape(Int<8>{}, Int<SplitkvLayoutTraits::kBlockKSmem>{}),
        make_stride(Int<SplitkvLayoutTraits::kBlockKSmem>{}, Int<1>{}))));

using SplitkvSmemLayoutPSwizzled = decltype(tile_to_shape(
    SplitkvSmemLayoutPAtomSwizzled{},
    make_shape(Int<SplitkvLayoutTraits::kQRowsPad>{}, Int<SplitkvLayoutTraits::kKVTile>{})));

using SplitkvSmemLayoutPNoSwizzle =
    decltype(get_nonswizzle_portion(SplitkvSmemLayoutPSwizzled{}));

using SplitkvSmemLayoutPStore = SplitkvSmemLayoutPSwizzled;
using SplitkvSmemLayoutPLogical = SplitkvSmemLayoutPNoSwizzle;

__device__ __forceinline__ void splitkv_load_q_cute(
    int tid, const bfloat16_t* q_row_base, bfloat16_t* smem_Q)
{
    using Tr = SplitkvLayoutTraits;
    auto sQ = make_tensor(make_smem_ptr(smem_Q), typename SplitkvSmemLayoutQ::Layout{});
    constexpr int kQElems = Tr::kGQARatio * Tr::kHeadDim;
    for (int e = tid; e < kQElems; e += Tr::kNThreads) {
        int h = e / Tr::kHeadDim, d = e % Tr::kHeadDim;
        sQ(h, d) = q_row_base[(long)h * Tr::kHeadDim + d];
    }
    constexpr int kPadElems = (Tr::kQRowsPad - Tr::kGQARatio) * Tr::kHeadDim;
    for (int e = tid; e < kPadElems; e += Tr::kNThreads) {
        int r = Tr::kGQARatio + e / Tr::kHeadDim, c = e % Tr::kHeadDim;
        sQ(r, c) = bfloat16_t(0.f);
    }
}

__device__ __forceinline__ void splitkv_load_k_tile_async(
    int tid, const bfloat16_t* k_base, long kv_stride_seq, int t_count, bfloat16_t* smem_K_stage)
{
    using Tr = SplitkvLayoutTraits;
    auto sK = make_tensor(make_smem_ptr(smem_K_stage), SplitkvSmemLayoutKVPlaneSwizzled{});
    constexpr int kVecsPerRow = Tr::kHeadDim / 8, total_vecs = Tr::kKVTile * kVecsPerRow;
    for (int i = tid; i < total_vecs; i += Tr::kNThreads) {
        int tok = i / kVecsPerRow, ck = i % kVecsPerRow, dim = ck * 8;
        int abs = min(tok, max(t_count - 1, 0));
        if (tok < t_count) cp_async_cg<16>(&sK(tok, dim), k_base + (long)abs * kv_stride_seq + dim);
        else               cp_async_cg<0>(&sK(tok, dim), k_base + (long)abs * kv_stride_seq + dim);
    }
}

__device__ __forceinline__ void splitkv_load_v_tile_async(
    int tid, const bfloat16_t* v_base, long kv_stride_seq, int t_count, bfloat16_t* smem_V_stage)
{
    using Tr = SplitkvLayoutTraits;
    auto sV = make_tensor(make_smem_ptr(smem_V_stage), SplitkvSmemLayoutKVPlaneSwizzled{});
    constexpr int kVecsPerRow = Tr::kHeadDim / 8, total_vecs = Tr::kKVTile * kVecsPerRow;
    for (int i = tid; i < total_vecs; i += Tr::kNThreads) {
        int tok = i / kVecsPerRow, ck = i % kVecsPerRow, dim = ck * 8;
        int abs = min(tok, max(t_count - 1, 0));
        if (tok < t_count) cp_async_cg<16>(&sV(tok, dim), v_base + (long)abs * kv_stride_seq + dim);
        else               cp_async_cg<0>(&sV(tok, dim), v_base + (long)abs * kv_stride_seq + dim);
    }
}

__device__ __forceinline__ void splitkv_load_kv_tile_async(
    int tid, const bfloat16_t* k_base, const bfloat16_t* v_base,
    long kv_stride_seq, int t_count, bfloat16_t* smem_K, bfloat16_t* smem_V)
{
    splitkv_load_k_tile_async(tid, k_base, kv_stride_seq, t_count, smem_K);
    splitkv_load_v_tile_async(tid, v_base, kv_stride_seq, t_count, smem_V);
    cp_async_fence();
}

__device__ __forceinline__ void splitkv_zero_p_tile(int tid, bfloat16_t* smem_P) {
    constexpr int kBytes = SplitkvLayoutTraits::kQRowsPad * SplitkvLayoutTraits::kKVTile * sizeof(bfloat16_t);
    static_assert(kBytes % sizeof(uint128_t) == 0, "P tile must be 128-bit aligned for vector clear");
    constexpr int kVecs = kBytes / sizeof(uint128_t);
    auto* smem_P_vec = reinterpret_cast<uint128_t*>(smem_P);
    uint128_t zero_vec{};
    for (int idx = tid; idx < kVecs; idx += SplitkvLayoutTraits::kNThreads) {
        smem_P_vec[idx] = zero_vec;
    }
}

// =============================================================================
// 11. SplitKV score MMA + write frag C (from splitkv_score_gemm_cute.cuh)
// =============================================================================

// Score MMA C → smem_score[row][col] mapping for softmax gather / PV bridge.
// Must match splitkv_write_frag_c_to_smem_float (partition_C(SmemLayoutC)) and
// Python splitkv_score_frag_coord_py used by the verifier coordinate mapping.
__device__ __forceinline__ void splitkv_score_smem_frag_coord(int tid, int fi, int& row, int& col) {
    const int lane_id = tid & 31;
    const int warp_id = tid >> 5;
    row = (lane_id >> 2) | (((fi & 3) >> 1) << 3);
    col = (warp_id << 4) | ((lane_id & 3) << 1) | ((fi >> 2) << 3) | (fi & 1);
}

template <typename TensorC>
__device__ __forceinline__ void splitkv_write_frag_c_to_smem_float(
    int tid, TensorC const& tCrC, float* smem_c_base)
{
    using SLC = typename AttnScoreGemmConfig::SmemLayoutC;
    typename AttnScoreGemmConfig::MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    auto sC = make_tensor(make_smem_ptr(smem_c_base), SLC{});
    auto tCsC = thr_mma.partition_C(sC);
    if (tid < KernelConfig::NUM_THREADS) {
        #pragma unroll
        for (int n = 0; n < (int)size(tCrC); ++n) tCsC(n) = float(tCrC(n));
    }
}

template <typename TensorC>
__device__ __forceinline__ void splitkv_score_mma_accumulate(
    int tid, bfloat16_t* smem_A_plain, bfloat16_t* smem_KV_plain, int stage, TensorC& tCrC)
{
    using CFG = AttnScoreGemmConfig;
    typename CFG::MMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);
    bfloat16_t* kv_plane = smem_KV_plain
        + (long)stage * (long)CFG::kTileNScore * (long)LlamaConfig::HEAD_DIM;
    typename CFG::S2RCopyAtomA s2r_a;
    typename CFG::S2RCopyAtomB_ScorePlainK_AsFA s2r_b;
    auto s2r_copy_a = make_tiled_copy_A(s2r_a, tiled_mma);
    auto s2r_copy_b = make_tiled_copy_B(s2r_b, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_slice(tid);
    auto s2r_thr_b = s2r_copy_b.get_slice(tid);

    // K: swizzled SMEM layout for bank-conflict-free LDSM
    auto sK_swiz = make_tensor(make_smem_ptr(kv_plane), SplitkvSmemLayoutKVPlaneSwizzled{});
    auto sK_logical = make_tensor(make_smem_ptr(kv_plane), SplitkvSmemLayoutKVPlane{});
    auto tCrB_full = thr_mma.partition_fragment_B(sK_logical);
    auto tBsK = s2r_thr_b.partition_S(sK_swiz);
    auto tCrB_copy = s2r_thr_b.retile_D(tCrB_full);

    // Q: plain layout (only 16 rows, minimal conflict)
    auto sQ = make_tensor(make_smem_ptr(smem_A_plain),
        make_layout(make_shape(Int<KernelConfig::kTileM>{}, Int<LlamaConfig::HEAD_DIM>{}),
                    make_stride(Int<LlamaConfig::HEAD_DIM>{}, Int<1>{})));
    auto tCrA_full = thr_mma.partition_fragment_A(sQ);
    auto tAsQ = s2r_thr_a.partition_S(sQ);
    auto tCrA_copy = s2r_thr_a.retile_D(tCrA_full);

    constexpr int nk = LlamaConfig::HEAD_DIM / CFG::kMmaPK;
    #pragma unroll
    for (int ik = 0; ik < nk; ++ik) {
        cute::copy(s2r_copy_a, tAsQ(_, _, ik), tCrA_copy(_, _, ik));
        cute::copy(s2r_copy_b, tBsK(_, _, ik), tCrB_copy(_, _, ik));
        cute::gemm(tiled_mma, tCrC, tCrA_full(_, _, ik), tCrB_full(_, _, ik), tCrC);
    }
}

// =============================================================================
// 11b. Layout conversion for accumulator (Flash pattern)
//      Convert (MMA=4, MMA_M, MMA_N) to (nrow, ncol) for row-based rescale.
// =============================================================================

template <typename Layout>
__device__ __forceinline__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});
    return make_layout(make_layout(get<1>(get<0>(l)), get<1>(l)),
                       make_layout(get<0>(get<0>(l)), get<2>(l)));
}

// Convert (nrow=(2,MMA_M), ncol=(2,MMA_N)) to A-register layout for m16n8k16.
template <typename MMA_traits, typename Layout>
__device__ __forceinline__ auto convert_layout_rowcol_Aregs(Layout rowcol_layout) {
    using X = Underscore;
    static_assert(decltype(size<0, 0>(rowcol_layout))::value == 2);
    static_assert(decltype(size<1, 0>(rowcol_layout))::value == 2);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    constexpr int MMA_N_divisor = (mma_shape_K == 8) ? 1 : 2;
    auto l = logical_divide(rowcol_layout, Shape<X, Shape<X, Int<MMA_N_divisor>>>{});
    return make_layout(
        make_layout(get<0>(get<1>(l)), get<0>(get<0>(l)), get<0>(get<1>(get<1>(l)))),
        get<1>(get<0>(l)), get<1>(get<1>(get<1>(l))));
}


template <typename To_type, typename Engine, typename Layout>
__device__ __forceinline__ auto splitkv_convert_type(Tensor<Engine, Layout> const& tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

template <typename PFragFloat>
__device__ __forceinline__ auto splitkv_make_score_frag_tensor(int tid, PFragFloat* x_frag_f) {
    using ScoreCFG = AttnScoreGemmConfig;
    typename ScoreCFG::MMA tiled_mma_score;
    auto thr_mma_score = tiled_mma_score.get_slice(tid);
    float score_layout_probe[8];
    auto gC_probe = make_tensor(
        make_rmem_ptr(&score_layout_probe[0]),
        make_shape(Int<SplitkvLayoutTraits::kQRowsPad>{}, Int<SplitkvLayoutTraits::kKVTile>{}),
        LayoutRight{});
    auto tCrProbe = thr_mma_score.partition_fragment_C(gC_probe);
    return make_tensor(make_rmem_ptr(x_frag_f), tCrProbe.layout());
}

// P×V gemm: P (softmax) in A-reg layout, V from smem (VtransposedNoSwizzle).
// Layout conversion from score fragment to A registers is done in-kernel so the
// score/PV bridge has one explicit ownership mapping.
template <typename AccTensor, typename PRegsTensor>
__device__ __forceinline__ void splitkv_pv_gemm_cute(
    int tid, AccTensor& acc_o, PRegsTensor const& tOrP, bfloat16_t* smem_V_stage,
    typename AttnPvGemmConfig::MMA const& tiled_mma)
{
    auto thr_mma = tiled_mma.get_slice(tid);
    auto sVt = make_tensor(make_smem_ptr(smem_V_stage), SplitkvSmemLayoutVtransposed{});
    auto sVtNoSwizzle = make_tensor(make_smem_ptr(smem_V_stage), SplitkvSmemLayoutVtransposedNoSwizzle{});
    auto smem_tiled_copy_V = make_tiled_copy_B(typename AttnPvGemmConfig::S2RCopyAtomB_Trans{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_slice(tid);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tCrB_copy_view = smem_thr_copy_V.retile_D(tOrVt);
    cute::copy(smem_tiled_copy_V, tOsVt(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < (int)size<2>(tOrP); ++i) {
        if (i < (int)size<2>(tOrP) - 1) {
            cute::copy(smem_tiled_copy_V, tOsVt(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tOrP(_, _, i), tOrVt(_, _, i), acc_o);
    }
}

// =============================================================================
// Vectorized epilogue helpers (128-bit gmem access)
// =============================================================================

// Vectorized residual-add epilogue for GEMM tiles that store bf16 results in smem.
// Loads 8 bf16 per thread from smem, loads 8 bf16 residual from gmem (128-bit),
// adds in float, stores 8 bf16 back to gmem (128-bit).
//
// smem_c:        [kTileM_ x kTileN_] row-major bf16 in shared memory
// gmem_inout:    base of hidden_state buffer (in-place residual add)
// gmem_stride:   elements between rows (HIDDEN_DIM)
// tile_n_offset: N column offset into the gmem row
// batch_size:    active M rows
// n_dim:         total N dimension for bounds check
template <int kTileM_, int kTileN_, int kNumThreads>
__device__ __forceinline__ void vec_epilogue_residual_add_bf16(
    const bfloat16_t* __restrict__ smem_c,
    bfloat16_t* __restrict__ gmem_inout,
    int gmem_stride,
    int tile_n_offset,
    int batch_size,
    int n_dim,
    const LlamaGlobals& g,
    int tid)
{
    constexpr int kVecElems = 8;
    constexpr int kVecsPerRow = kTileN_ / kVecElems;
    constexpr int kTotalVecs = kTileM_ * kVecsPerRow;

    static_assert(kTileN_ % kVecElems == 0,
        "kTileN_ must be a multiple of 8 for 128-bit vectorized epilogue");

    #pragma unroll
    for (int vid = tid; vid < kTotalVecs; vid += kNumThreads) {
        int m = vid / kVecsPerRow;
        int vec_idx = vid % kVecsPerRow;
        int n_local = vec_idx * kVecElems;
        int n_global = tile_n_offset + n_local;

        if (m >= batch_size || n_global + kVecElems > n_dim) continue;
        if (!row_active(g, m)) continue;

        // 128-bit load from smem (8 bf16)
        uint4 gemm_data = *reinterpret_cast<const uint4*>(
            &smem_c[m * kTileN_ + n_local]);

        // 128-bit load residual from gmem (8 bf16)
        bfloat16_t* gmem_row = gmem_inout + (long)m * gmem_stride + n_global;
        uint4 res_data = *reinterpret_cast<const uint4*>(gmem_row);

        // Unpack bf16x2 pairs, add in float, repack
        __nv_bfloat162* gemm_p = reinterpret_cast<__nv_bfloat162*>(&gemm_data);
        __nv_bfloat162* res_p  = reinterpret_cast<__nv_bfloat162*>(&res_data);
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            float2 gf = __bfloat1622float2(gemm_p[p]);
            float2 rf = __bfloat1622float2(res_p[p]);
            res_p[p] = __float22bfloat162_rn(
                make_float2(gf.x + rf.x, gf.y + rf.y));
        }

        // 128-bit store to gmem (8 bf16)
        *reinterpret_cast<uint4*>(gmem_row) = res_data;
    }
}

// Vectorized residual-add epilogue for GEMM tiles that store float results in smem.
// Loads 4 float per sub-vector from smem, loads bf16 residual from gmem,
// adds in float, stores bf16 to gmem.  Processes 8 output bf16 per iteration
// using two float4 smem reads + one uint4 gmem read/write.
template <int kTileM_, int kTileN_, int kNumThreads>
__device__ __forceinline__ void vec_epilogue_residual_add_float_smem(
    const float* __restrict__ smem_c,
    bfloat16_t* __restrict__ gmem_inout,
    int gmem_stride,
    int tile_n_offset,
    int batch_size,
    int n_dim,
    const LlamaGlobals& g,
    int tid)
{
    constexpr int kVecElems = 8;
    constexpr int kVecsPerRow = kTileN_ / kVecElems;
    constexpr int kTotalVecs = kTileM_ * kVecsPerRow;

    static_assert(kTileN_ % kVecElems == 0,
        "kTileN_ must be a multiple of 8 for vectorized epilogue");

    #pragma unroll
    for (int vid = tid; vid < kTotalVecs; vid += kNumThreads) {
        int m = vid / kVecsPerRow;
        int vec_idx = vid % kVecsPerRow;
        int n_local = vec_idx * kVecElems;
        int n_global = tile_n_offset + n_local;

        if (m >= batch_size || n_global + kVecElems > n_dim) continue;
        if (!row_active(g, m)) continue;

        // Two float4 loads from smem (8 floats = 32 bytes)
        const float* smem_row = &smem_c[m * kTileN_ + n_local];
        float4 f_lo = *reinterpret_cast<const float4*>(smem_row);
        float4 f_hi = *reinterpret_cast<const float4*>(smem_row + 4);

        // 128-bit load residual from gmem (8 bf16 = 16 bytes)
        bfloat16_t* gmem_row = gmem_inout + (long)m * gmem_stride + n_global;
        uint4 res_data = *reinterpret_cast<const uint4*>(gmem_row);
        __nv_bfloat162* res_p = reinterpret_cast<__nv_bfloat162*>(&res_data);

        // Add in float, repack to bf16x2
        float2 r0 = __bfloat1622float2(res_p[0]);
        float2 r1 = __bfloat1622float2(res_p[1]);
        float2 r2 = __bfloat1622float2(res_p[2]);
        float2 r3 = __bfloat1622float2(res_p[3]);

        res_p[0] = __float22bfloat162_rn(make_float2(f_lo.x + r0.x, f_lo.y + r0.y));
        res_p[1] = __float22bfloat162_rn(make_float2(f_lo.z + r1.x, f_lo.w + r1.y));
        res_p[2] = __float22bfloat162_rn(make_float2(f_hi.x + r2.x, f_hi.y + r2.y));
        res_p[3] = __float22bfloat162_rn(make_float2(f_hi.z + r3.x, f_hi.w + r3.y));

        // 128-bit store to gmem (8 bf16)
        *reinterpret_cast<uint4*>(gmem_row) = res_data;
    }
}

// Vectorized store-only epilogue (no residual): convert bf16 smem to gmem.
// Used when the output is written to a fresh buffer (e.g., silu_out in upgate).
template <int kTileM_, int kTileN_, int kNumThreads>
__device__ __forceinline__ void vec_epilogue_store_bf16(
    const bfloat16_t* __restrict__ smem_c,
    bfloat16_t* __restrict__ gmem_out,
    int gmem_stride,
    int tile_n_offset,
    int batch_size,
    int n_dim,
    const LlamaGlobals& g,
    int tid)
{
    constexpr int kVecElems = 8;
    constexpr int kVecsPerRow = kTileN_ / kVecElems;
    constexpr int kTotalVecs = kTileM_ * kVecsPerRow;

    static_assert(kTileN_ % kVecElems == 0,
        "kTileN_ must be a multiple of 8 for vectorized epilogue");

    #pragma unroll
    for (int vid = tid; vid < kTotalVecs; vid += kNumThreads) {
        int m = vid / kVecsPerRow;
        int vec_idx = vid % kVecsPerRow;
        int n_local = vec_idx * kVecElems;
        int n_global = tile_n_offset + n_local;

        if (m >= batch_size || n_global + kVecElems > n_dim) continue;
        if (!row_active(g, m)) continue;

        uint4 data = *reinterpret_cast<const uint4*>(
            &smem_c[m * kTileN_ + n_local]);

        bfloat16_t* gmem_row = gmem_out + (long)m * gmem_stride + n_global;
        *reinterpret_cast<uint4*>(gmem_row) = data;
    }
}

// Vectorized store from float smem to bf16 gmem (no residual).
// Reads floats from smem, converts to bf16, stores 128-bit to gmem.
// Used for upgate SiLU output.
template <int kTileM_, int kTileN_, int kNumThreads>
__device__ __forceinline__ void vec_epilogue_store_float_to_bf16_gmem(
    const float* __restrict__ smem_c,
    bfloat16_t* __restrict__ gmem_out,
    int gmem_stride,
    int tile_n_offset,
    int batch_size,
    int n_dim,
    const LlamaGlobals& g,
    int tid)
{
    constexpr int kVecElems = 8;
    constexpr int kVecsPerRow = kTileN_ / kVecElems;
    constexpr int kTotalVecs = kTileM_ * kVecsPerRow;

    static_assert(kTileN_ % kVecElems == 0,
        "kTileN_ must be a multiple of 8 for vectorized epilogue");

    auto sScratch = make_tensor(make_smem_ptr(const_cast<float*>(smem_c)),
                                SharedOutputScratchLayoutT<kTileM_, kTileN_>{});

    #pragma unroll
    for (int vid = tid; vid < kTotalVecs; vid += kNumThreads) {
        int m = vid / kVecsPerRow;
        int vec_idx = vid % kVecsPerRow;
        int n_local = vec_idx * kVecElems;
        int n_global = tile_n_offset + n_local;

        if (m >= batch_size || n_global + kVecElems > n_dim) continue;
        if (!row_active(g, m)) continue;

        // Read from swizzled SMEM (8 consecutive floats are physically contiguous
        // since swizzle operates on groups of 8+ elements and n_local is 8-aligned)
        const float* smem_row = &sScratch(m, n_local);
        float4 f_lo = *reinterpret_cast<const float4*>(smem_row);
        float4 f_hi = *reinterpret_cast<const float4*>(smem_row + 4);

        // Convert to bf16 and pack into uint4 (8 bf16 = 16 bytes)
        uint4 out_data;
        __nv_bfloat162* out_p = reinterpret_cast<__nv_bfloat162*>(&out_data);
        out_p[0] = __float22bfloat162_rn(make_float2(f_lo.x, f_lo.y));
        out_p[1] = __float22bfloat162_rn(make_float2(f_lo.z, f_lo.w));
        out_p[2] = __float22bfloat162_rn(make_float2(f_hi.x, f_hi.y));
        out_p[3] = __float22bfloat162_rn(make_float2(f_hi.z, f_hi.w));

        // 128-bit store to gmem
        bfloat16_t* gmem_row_ptr = gmem_out + (long)m * gmem_stride + n_global;
        *reinterpret_cast<uint4*>(gmem_row_ptr) = out_data;
    }
}

}  // namespace megakernel
