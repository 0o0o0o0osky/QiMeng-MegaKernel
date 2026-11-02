// Ampere BF16 Tensor-Core GEMM pipeline.
// Design:
//   - Multistage pipeline (BK=64, kStage=3 recommended per tc_gap_analysis)
//   - Epilogue functors transform fp32 accumulator *before* bf16 conversion
//   - All stores are 128-bit coalesced (8 bf16 per thread per transaction)
//   - Supports residual loads (StoreAccum) via functor read-modify path
// Usage (from MegaKernel fusion instruction device functions):
//   gemm_unified_tile<Cfg, EpiFunctor>(
//       A_ptr, B_ptr, C_out_ptr,
//       M, N, K, iy, ix_start, ix_end,
//       epilogue_functor, smem);

#pragma once

#include <type_traits>

#include <cute/tensor.hpp>
#include "common/execution_config.cuh"
#include "common/gemm_epilogue.cuh"

using namespace cute;

namespace mk_gemm {

namespace detail {

template <typename Epilogue, typename = void>
struct uses_fragment_transform : std::false_type {};

template <typename Epilogue>
struct uses_fragment_transform<
    Epilogue,
    std::void_t<decltype(Epilogue::kUseFragmentTransform)>>
    : std::bool_constant<Epilogue::kUseFragmentTransform> {};

template <typename Epilogue, typename = void>
struct uses_fragment_mirror : std::false_type {};

template <typename Epilogue>
struct uses_fragment_mirror<
    Epilogue,
    std::void_t<decltype(Epilogue::kUseFragmentMirror)>>
    : std::bool_constant<Epilogue::kUseFragmentMirror> {};

template <typename Epilogue, typename = void>
struct uses_fp32_terminal_store : std::false_type {};

template <typename Epilogue>
struct uses_fp32_terminal_store<
    Epilogue,
    std::void_t<decltype(Epilogue::kUseFp32TerminalStore)>>
    : std::bool_constant<Epilogue::kUseFp32TerminalStore> {};

}  // namespace detail

template <typename Config>
__device__ __forceinline__
void gemm_unified_preload_b_stages(
    typename Config::T const* __restrict__ B_ptr,
    int N, int K,
    int ix,
    void* smem,
    int ldb = -1
) {
    using T = typename Config::T;

    constexpr int BM     = Config::BM;
    constexpr int BN     = Config::BN;
    constexpr int BK     = Config::BK;
    constexpr int kStage = Config::kStage;

    using SmemLayoutA     = typename Config::SmemLayoutA;
    using SmemLayoutB     = typename Config::SmemLayoutB;
    using G2SCopyB        = typename Config::G2SCopyB;
    using G2SCopyB_ZFILL  = typename Config::G2SCopyB_ZFILL;

    T* Ashm = reinterpret_cast<T*>(smem);
    T* Bshm = Ashm + cute::cosize(SmemLayoutA{});

    if (ldb < 0) ldb = K;
    Tensor B = make_tensor(make_gmem_ptr(B_ptr),
                           make_shape(N, K), make_stride(ldb, Int<1>{}));
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(threadIdx.x);
    auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
    auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

    G2SCopyB_ZFILL g2s_tiled_copy_b_zfill;

    const int ntile = (K + BK - 1) / BK;
    const bool k_ragged = (K % BK) != 0;
    const bool n_ragged = (ix * BN + BN) > N;
    const bool edge_b = k_ragged || n_ragged;

    Tensor cB = make_identity_tensor(make_shape(Int<BN>{}, Int<BK>{}));
    auto tBcB = g2s_thr_copy_b.partition_S(cB);
    auto tBpB = make_tensor<bool>(make_shape(size<1>(tBsB_copy), size<2>(tBsB_copy)));

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (istage >= ntile) { cp_async_fence(); continue; }
        if (!edge_b) {
            cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        } else {
            int kbase = istage * BK;
            CUTE_UNROLL
            for (int n = 0; n < size<0>(tBpB); ++n)
                CUTE_UNROLL
                for (int kk = 0; kk < size<1>(tBpB); ++kk)
                    tBpB(n, kk) = ((ix * BN + get<0>(tBcB(0, n, kk))) < N) &&
                                  ((kbase + get<1>(tBcB(0, n, kk))) < K);
            cute::copy_if(g2s_tiled_copy_b_zfill, tBpB, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
        }
        cp_async_fence();
    }
}

// gemm_unified_tile — __device__ function for a single M-tile across N-tile range
// Template parameters:
//   Config: GemmConfig<BM, BN, BK, kStage>
//   Epilogue: functor with float transform(float acc, int m_global, int n_global)
//             OR terminal store functor (dst_ptr, ld set by caller)
// Args:
//   A_ptr, B_ptr: [M,K] row-major, [N,K] K-major
//   C_out_ptr: dummy for coordinate; actual store via epilogue functor
//   M, N, K: runtime dims
//   iy: M-tile index (this function computes one M-tile across [ix_start, ix_end) N-tiles)
//   ix_start, ix_end: N-tile range
//   epilogue: functor instance (by value, contains pointers)
//   smem: shared memory buffer (caller allocates)
template <typename Config, typename Epilogue>
__device__ __forceinline__
void gemm_unified_tile(
    typename Config::T const* __restrict__ A_ptr,
    typename Config::T const* __restrict__ B_ptr,
    typename Config::T*       __restrict__ C_out_ptr,  // for coord partition only
    int M, int N, int K,
    int iy, int ix_start, int ix_end,
    Epilogue epilogue,
    void* smem,
    bool b_stages_preloaded = false,
    // Optional row strides (inner/K stride stays 1). -1 → contiguous defaults
    // (lda=K, ldb=K, ldc=N).  Used by strided views (e.g. materialized attention
    // reading Q/K/V from a packed [seq, 2560] qkv tensor, or writing ctx into a
    // [seq, 2048] head-interleaved output) — no repack kernel needed.
    int lda = -1, int ldb = -1, int ldc = -1
) {
    using T    = typename Config::T;
    if (lda < 0) lda = K;
    if (ldb < 0) ldb = K;
    if (ldc < 0) ldc = N;

    constexpr int BM     = Config::BM;
    constexpr int BN     = Config::BN;
    constexpr int BK     = Config::BK;
    constexpr int kStage = Config::kStage;

    using SmemLayoutA  = typename Config::SmemLayoutA;
    using SmemLayoutB  = typename Config::SmemLayoutB;
    using SmemLayoutC  = typename Config::SmemLayoutC;
    using TiledMMA     = typename Config::TiledMMA;
    using G2SCopyA     = typename Config::G2SCopyA;
    using G2SCopyB     = typename Config::G2SCopyB;
    using G2SCopyA_ZFILL = typename Config::G2SCopyA_ZFILL;
    using G2SCopyB_ZFILL = typename Config::G2SCopyB_ZFILL;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    using R2SCopyAtomC = typename Config::R2SCopyAtomC;
    using S2GCopyC     = typename Config::S2GCopyC;
    constexpr int kS2gRounds = Config::kS2gRounds;

    constexpr bool kEarly = Config::kEarlyWait;

    T* Ashm = reinterpret_cast<T*>(smem);
    T* Bshm = Ashm + cute::cosize(SmemLayoutA{});

    int idx = threadIdx.x;

    // Global tensors
    Tensor A = make_tensor(make_gmem_ptr(A_ptr),
                           make_shape(M, K), make_stride(lda, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(B_ptr),
                           make_shape(N, K), make_stride(ldb, Int<1>{}));

    // Shared memory
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    // TiledMMA
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    // Partition A for this M-tile
    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));

    // G2S copy setup
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);

    // ZFILL G2S copies for ragged (edge) tile loads — same thread/value layout
    // as the non-ZFILL copies, so the partitioned src/dst tensors above are
    // reused; only the copy atom differs (predicated lanes → zero-fill).
    // Pass the TiledCopy object to copy_if (the ThrCopy overload is deleted).
    G2SCopyA_ZFILL g2s_tiled_copy_a_zfill;
    G2SCopyB_ZFILL g2s_tiled_copy_b_zfill;

    // S2R copy setup
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA_s2r = s2r_thr_copy_a.partition_S(sA);

    auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b   = s2r_tiled_copy_b.get_slice(idx);
    auto tBsB_s2r = s2r_thr_copy_b.partition_S(sB);

    int ntile = (K + BK - 1) / BK;  // ceil(K/BK), handles partial last K-tile
    const bool k_ragged = (K % BK) != 0;
    // M-edge: the iy M-tile may extend past M (e.g. decoder chunk=50 < BM=64).
    // local_tile does NOT clamp, so without predication the A load reads gmem
    // rows [M, iy*BM+BM) out of bounds (garbage from the allocation tail) →
    // non-deterministic + wrong results.  m_ragged is constant across the
    // N-tile loop (depends only on iy, M).
    const bool m_ragged = (iy * BM + BM) > M;

    // ── N-tile loop ────────────────────────────────────────────────────────
    for (int ix = ix_start; ix < ix_end; ++ix) {
        Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
        auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
        auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

        // N-edge: like M, the ix N-tile may extend past N (partial-N tiles).
        const bool n_ragged = (ix * BN + BN) > N;

        // K-edge predication: identity coords for the A/B G2S partitions.
        // Each cp.async op copies 8 contiguous bf16 along K; the residue is a
        // multiple of 8, so per-op predication (first element's K coord) is exact.
        // The G2S value layout is (1,8): one op = 1 row (M or N) × 8 K-cols, so
        // M/N predication at the op's first-element row coord is also exact.
        // copy_if with cp.async zero-fills masked ops → MMA gets 0 for OOB.
        Tensor cA = make_identity_tensor(make_shape(Int<BM>{}, Int<BK>{}));
        Tensor cB = make_identity_tensor(make_shape(Int<BN>{}, Int<BK>{}));
        auto tAcA = g2s_thr_copy_a.partition_S(cA);  // (CPY, CPY_M, CPY_K)
        auto tBcB = g2s_thr_copy_b.partition_S(cB);  // (CPY, CPY_N, CPY_K)

        // Predicate tensors (CPY_M/N, CPY_K), rebuilt per loaded tile.
        auto tApA = make_tensor<bool>(make_shape(size<1>(tAsA_copy), size<2>(tAsA_copy)));
        auto tBpB = make_tensor<bool>(make_shape(size<1>(tBsB_copy), size<2>(tBsB_copy)));

        const bool edge_a = k_ragged || m_ragged;
        const bool edge_b = k_ragged || n_ragged;

        auto load_ktile_a = [&](int it, int iw) {
            if (!edge_a) {
                cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, it), tAsA_copy(_, _, _, iw));
            } else {
                int kbase = it * BK;
                CUTE_UNROLL
                for (int m = 0; m < size<0>(tApA); ++m)
                    CUTE_UNROLL
                    for (int kk = 0; kk < size<1>(tApA); ++kk)
                        tApA(m, kk) = ((iy * BM + get<0>(tAcA(0, m, kk))) < M) &&
                                      ((kbase + get<1>(tAcA(0, m, kk))) < K);
                cute::copy_if(g2s_tiled_copy_a_zfill, tApA, tAgA_copy(_, _, _, it), tAsA_copy(_, _, _, iw));
            }
        };

        auto load_ktile_b = [&](int it, int iw) {
            if (!edge_b) {
                cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, it), tBsB_copy(_, _, _, iw));
            } else {
                int kbase = it * BK;
                CUTE_UNROLL
                for (int n = 0; n < size<0>(tBpB); ++n)
                    CUTE_UNROLL
                    for (int kk = 0; kk < size<1>(tBpB); ++kk)
                        tBpB(n, kk) = ((ix * BN + get<0>(tBcB(0, n, kk))) < N) &&
                                      ((kbase + get<1>(tBcB(0, n, kk))) < K);
                cute::copy_if(g2s_tiled_copy_b_zfill, tBpB, tBgB_copy(_, _, _, it), tBsB_copy(_, _, _, iw));
            }
        };

        // Load K-tile `it` into smem stage `iw` (predicated if ragged).
        auto load_ktile = [&](int it, int iw) {
            load_ktile_a(it, iw);
            load_ktile_b(it, iw);
        };

        // Fragment setup
        T* C_dummy = Ashm;
        Tensor gC_fake = make_tensor(make_gmem_ptr(C_dummy),
                                     make_shape(Int<BM>{}, Int<BN>{}),
                                     make_stride(Int<BN>{}, Int<1>{}));
        auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
        auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
        auto tCrC = thr_mma.partition_fragment_C(gC_fake);
        clear(tCrC);

        auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);
        auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

        int itile_to_read = 0, ismem_read = 0, ismem_write = 0;

        // ── Prologue: prefill kStage-1 tiles ───────────────────────────────
        const bool use_preloaded_b = b_stages_preloaded && (ix == ix_start);
        #pragma unroll
        for (int istage = 0; istage < kStage - 1; ++istage) {
            if (istage < ntile) {
                load_ktile_a(istage, istage);
                if (!use_preloaded_b) {
                    load_ktile_b(istage, istage);
                }
                ++itile_to_read; ++ismem_write;
            }
            // Commit an empty group when K has fewer than kStage-1 tiles so
            // wait_group retains the same stage accounting.
            cp_async_fence();
        }
        cp_async_wait<kStage - 2>();
        __syncthreads();

        // Preload first fragment
        cute::copy(s2r_tiled_copy_a, tAsA_s2r(_, _, 0, ismem_read), tCrA_view(_, _, 0));
        cute::copy(s2r_tiled_copy_b, tBsB_s2r(_, _, 0, ismem_read), tCrB_view(_, _, 0));

        // ── Main loop ──────────────────────────────────────────────────────
        #pragma unroll 1
        for (int itile = 0; itile < ntile; ++itile) {
            int nk = size<2>(tCrA);
            #pragma unroll
            for (int ik = 0; ik < nk; ++ik) {
                int ik_next = (ik + 1) % nk;
                // ── Boundary wait (default): drain + read-advance at ik==nk-1 ──
                if constexpr (!kEarly)
                if (ik == nk - 1) {
                    // Tail fix: once prefetch stops issuing (itile_to_read == ntile)
                    // the outstanding-group count shrinks, and wait<kStage-2>
                    // (>=kStage-2 allowed pending) can return before the next
                    // tile's K-chunk lands — the MMA then consumes it early
                    // (single-column, ~1/ntile magnitude nondeterminism).
                    // Invariant: with continuous prefetch, tile itile+1 is
                    // guaranteed landed iff itile <= ntile - kStage; drain fully
                    // on every later transition.
                    if (itile + kStage >= ntile + 1) {
                        cp_async_wait<0>();
                    } else {
                        cp_async_wait<kStage - 2>();
                    }
                    __syncthreads();
                    ismem_read = (ismem_read + 1) % kStage;
                }
                cute::copy(s2r_tiled_copy_a, tAsA_s2r(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
                cute::copy(s2r_tiled_copy_b, tBsB_s2r(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));
                if (ik == 0 && itile_to_read < ntile) {
                    load_ktile(itile_to_read, ismem_write);
                    cp_async_fence();
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }
                // For shallow K tiles, move the boundary wait one fragment
                // earlier so the following MMA can cover part of its latency.
                // Moved from ik==nk-1 so the wait is covered by the ik==nk-1 MMA.
                // The read-stage advance stays paired with this wait: the ik==nk-1
                // S2R (frag 0 of the next tile) then reads the advanced stage, and
                // the per-tile advance count is unchanged (final ismem_read same,
                // and every S2R reads the same stage as the default path).  For
                // nk==2 this lands at ik==0, *after* the G2S issue above.  The
                // tail-drain condition (itile+kStage>=ntile+1 → wait<0>) is kept
                // verbatim; only its position moves.
                if constexpr (kEarly)
                if (ik == nk - 2) {
                    if (itile + kStage >= ntile + 1) {
                        cp_async_wait<0>();
                    } else {
                        cp_async_wait<kStage - 2>();
                    }
                    __syncthreads();
                    ismem_read = (ismem_read + 1) % kStage;
                }
                cute::gemm(tiled_mma, tCrC, tCrA(_, _, ik), tCrB(_, _, ik), tCrC);
            }
        }

        // ── Epilogue ───────────────────────────────────────────────────────
        // WAR hazard guard: the epilogue reuses sA's smem as the C scratch
        // (sC = sA(_,_,ismem_read), see below).  Two distinct hazards must be
        // closed before any thread writes sC:
        //   (1) The main loop's last cp_async_wait<kStage-2> leaves up to
        //       kStage-2 cp.async groups still IN FLIGHT, landing in sA/sB
        //       stages.  __syncthreads() does NOT wait for cp.async — only
        //       cp_async_wait does.  A late-landing copy would corrupt sC
        //       between the R2S store and the S2G load → non-determinism.
        //       Drain ALL outstanding copies first.
        //   (2) The main loop's final S2R reads of sA happen after the last
        //       in-loop __syncthreads(); a fast thread could begin its R2S
        //       write into sC (== sA) while a slow thread is still reading sA.
        // cp_async_wait<0>() drains (1); __syncthreads() closes (2) and makes
        // the drained copies visible to all threads.
        cp_async_wait<0>();
        __syncthreads();

        // Step 1: apply the epilogue functor on the fp32 accumulator *in place*,
        // in MMA fragment order, using MMA-partitioned coordinates.  Doing the
        // transform here (before any retiling) guarantees value/coordinate
        // ordering match — tCrC and tCcC share the identical MMA partition.
        {
            Tensor cC   = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
            Tensor tCcC = thr_mma.partition_C(cC);  // (MMA, MMA_M, MMA_N) coords

            if constexpr (detail::uses_fragment_transform<Epilogue>::value) {
                // Opt-in fragment path for transforms that need two or more
                // accumulator values at once (for example Q/K RoPE pairs).
                // The functor sees fp32 MMA registers and matching coordinates;
                // conversion to T still happens exactly once in Step 2 below.
                epilogue.transform_fragment(
                    tCrC, tCcC, iy * BM, ix * BN, M, N);
            } else {
                // Preserve the existing scalar epilogue path byte-for-byte for
                // all functors that do not opt in to fragment transforms.
                CUTE_UNROLL
                for (int e = 0; e < size(tCrC); ++e) {
                    auto coord = tCcC(e);
                    int m_global = iy * BM + get<0>(coord);
                    int n_global = ix * BN + get<1>(coord);
                    if (m_global < M && n_global < N) {
                        tCrC(e) = epilogue.transform(
                            float(tCrC(e)), m_global, n_global);
                    }
                }
            }
        }

        if constexpr (detail::uses_fp32_terminal_store<Epilogue>::value) {
            // Opt-in materialized-score path.  Each MMA fragment owns unique
            // output coordinates, so it can preserve the FP32 accumulator
            // directly without the ordinary BF16 shared-memory landing.
            Tensor cC = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
            Tensor tCcC = thr_mma.partition_C(cC);
            CUTE_UNROLL
            for (int e = 0; e < size(tCrC); ++e) {
                auto coord = tCcC(e);
                const int m_global = iy * BM + get<0>(coord);
                const int n_global = ix * BN + get<1>(coord);
                if (m_global < M && n_global < N)
                    epilogue.store(float(tCrC(e)), m_global, n_global);
            }
        } else {
        // Step 2: vectorized R2S→S2G of the transformed accumulator.
        // Pure type-convert (fp32→bf16) R2S, then 128-bit coalesced S2G.
        // Reuse sA's smem as C scratch.
        auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

        auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
        auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(idx);
        auto tCrC_r2s         = r2s_thr_copy_c.retile_S(tCrC);   // (CPY, CPY_M, CPY_N)
        auto tCsC_r2s         = r2s_thr_copy_c.partition_D(sC);  // (CPY, _1, _1, pipe)

        S2GCopyC s2g_tiled_copy_c;
        auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
        auto tCsC_s2g       = s2g_thr_copy_c.partition_S(sC);    // (CPY, _1, _1, pipe)

        // C_out for S2G
        Tensor gC = make_tensor(make_gmem_ptr(C_out_ptr),
                                make_shape(M, N), make_stride(ldc, Int<1>{}));
        Tensor gC_tile = local_tile(gC, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));
        auto tCgC_s2g  = s2g_thr_copy_c.partition_D(gC_tile);  // (CPY, CPY_M, CPY_N)

        auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);   // (CPY_, CPY_MN)
        auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);   // (CPY_, CPY_MN)

        // S2G block-level coordinate guard for partial tiles.  N is a multiple
        // of 8 and BN/tile bases are multiples of 8 (the 128-bit store width),
        // so every 8-element store block is wholly in- or out-of-bounds.  We
        // only need each block's base (m,n); take it from a BM×BN identity
        // partition (compile-time shape — no dynamic M/N layout, cheap).
        const bool tile_full = (iy * BM + BM <= M) && (ix * BN + BN <= N);

        Tensor cTile = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
        auto tCcTile  = s2g_thr_copy_c.partition_D(cTile);     // (CPY, CPY_M, CPY_N) local coords
        auto tCcTilex = group_modes<1, 3>(tCcTile);            // (CPY_, CPY_MN)

        int step = size<3>(tCsC_r2s);  // pipeline depth (kSmemCBatch=2)

        #pragma unroll
        for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
            // ---- R2S: convert fp32→bf16, write to smem ----
            #pragma unroll
            for (int j = 0; j < step; ++j) {
                // Odd MMA-tile counts (e.g. BM160xBN96 -> 5x3=15 slices) leave a
                // 1-slice tail; the phantom j would index OOB (garbage store).
                // [2026-07-05 fix: BM160+BN96 corruption found by B2's numeric gate]
                if (i + j >= size<1>(tCrC_r2sx)) break;
                auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
                CUTE_UNROLL
                for (int e = 0; e < size(t); ++e) {
                    t(e) = T(float(tCrC_r2sx(e, i + j)));
                }
                cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
            }
            __syncthreads();

            // ---- S2G: vectorized coalesced store ----
            #pragma unroll
            for (int j = 0; j < step; ++j) {
                if (i + j >= size<1>(tCrC_r2sx)) break;  // odd-count tail guard
                if constexpr (kS2gRounds == 1) {
                    if (tile_full) {
                        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
                    } else {
                        // Partial tile: guard each vector block by its global base coord.
                        auto c0 = tCcTilex(0, i + j);  // local (bm, bn) in [0, BM)x[0, BN)
                        int m0 = iy * BM + get<0>(c0);
                        int n0 = ix * BN + get<1>(c0);
                        if (m0 < M && n0 < N) {
                            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
                        }
                    }
                    if constexpr (detail::uses_fragment_mirror<Epilogue>::value) {
                        // Mirror from the exact BF16 shared-memory vector
                        // consumed by the primary S2G store, using the same
                        // vector copy atom.  This deliberately performs neither
                        // scalar BF16 stores nor a second float->BF16 conversion.
                        auto c0 = tCcTilex(0, i + j);
                        int m0 = iy * BM + get<0>(c0);
                        int n0 = ix * BN + get<1>(c0);
                        if (m0 < M && n0 < N) {
                            T* mirror_ptr = epilogue.mirror_pointer(m0, n0);
                            if (mirror_ptr != nullptr) {
                                auto source = tCsC_s2g(_, 0, 0, j);
                                auto destination = make_tensor(
                                    make_gmem_ptr(mirror_ptr),
                                    make_layout(shape(source)));
                                cute::copy(s2g_tiled_copy_c, source, destination);
                            }
                        }
                    }
                } else {
                    // Multi-round S2G: the smem stage holds one FULL accumulator
                    // tile (kMmaPM x kMmaPN); the S2G thread layout covers a
                    // (kS2gMThreads x kMmaPN) M-slice, so round r must read smem
                    // slice r (mode-1 of the partition) and write gmem slice
                    // (i+j)*kS2gRounds + r — a tile's R slices are contiguous in
                    // the flattened gmem partition (m' = tm*R + r, column-major).
                    // [2026-07-05 fix: was smem slice 0 every round + unscaled
                    // `i` in the dest index → garbage for every rounds>1 config;
                    // regression: test_unified.cu "rounds2" cases.]
                    CUTE_UNROLL
                    for (int r = 0; r < kS2gRounds; ++r) {
                        int out_i = (i + j) * kS2gRounds + r;
                        if (out_i < size<1>(tCgC_s2gx)) {
                            if (tile_full) {
                                cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, r, 0, j), tCgC_s2gx(_, out_i));
                            } else {
                                auto c0 = tCcTilex(0, out_i);
                                int m0 = iy * BM + get<0>(c0);
                                int n0 = ix * BN + get<1>(c0);
                                if (m0 < M && n0 < N) {
                                    cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, r, 0, j), tCgC_s2gx(_, out_i));
                                }
                            }
                            if constexpr (detail::uses_fragment_mirror<Epilogue>::value) {
                                auto c0 = tCcTilex(0, out_i);
                                int m0 = iy * BM + get<0>(c0);
                                int n0 = ix * BN + get<1>(c0);
                                if (m0 < M && n0 < N) {
                                    T* mirror_ptr = epilogue.mirror_pointer(m0, n0);
                                    if (mirror_ptr != nullptr) {
                                        auto source = tCsC_s2g(_, r, 0, j);
                                        auto destination = make_tensor(
                                            make_gmem_ptr(mirror_ptr),
                                            make_layout(shape(source)));
                                        cute::copy(s2g_tiled_copy_c, source, destination);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
        }
    }  // N-tile loop

}

}  // namespace mk_gemm
