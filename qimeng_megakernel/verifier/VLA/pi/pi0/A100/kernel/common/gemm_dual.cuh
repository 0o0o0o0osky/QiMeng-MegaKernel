// gemm_dual.cuh — same-A dual-B Tensor Core GEMM for SM80.
// Computes two accumulators in one K-loop with a shared A operand. A caller-
// supplied merge functor combines them exactly once before the normal epilogue.
//   - A is loaded once and its S2R fragment feeds both MMA columns.
//   - B0/B1 share the K-loop; each has its own G2S ring + S2R fragment + accum.
// The merge collapses both accumulators into tCrC0 before the single-accumulator
// R2S-to-S2G epilogue. The common core does not prescribe the combine formula.
// A, B0, and B1 for one K-tile are issued into
// the SAME cp.async commit group (one cp_async_fence after all three), so the
// in-flight *group* count is identical to the single-B kernel — cp_async_wait
// counts groups, not copies.  The kStage-2 / tail-drain (wait<0>) invariants
// are therefore preserved verbatim.

#pragma once

#include <cute/tensor.hpp>
#include "common/execution_config.cuh"
#include "common/gemm_epilogue.cuh"

using namespace cute;

namespace mk_gemm {

// Dual-B shared-memory budget: one A ring and two B rings.
//   single-B kShmSize (Cfg) uses A+1×B; dual adds one more B ring.
template <typename Config>
struct DualSmem {
    using T = typename Config::T;
    static constexpr int cosA = cute::cosize(typename Config::SmemLayoutA{});
    static constexpr int cosB = cute::cosize(typename Config::SmemLayoutB{});
    static constexpr int cosC = cute::cosize(typename Config::SmemLayoutC{});
    // AB region must also be large enough to alias sC (epilogue reuses sA).
    static constexpr int abBytes = (cosA + 2 * cosB) * (int)sizeof(T);
    static constexpr int cBytes  = cosC * (int)sizeof(T);
    static constexpr int kShmSizeDual = (abBytes > cBytes) ? abBytes : cBytes;
};

// gemm_dual_preload_b_stages — dual-B analogue of gemm_unified_preload_b_stages:
// issue the first kStage-1 K-tiles of BOTH B rings (B0 then B1, ONE cp.async
// commit group per stage) into the dual smem layout (A ring | B0 ring | B1 ring)
// during the dispatch loop's barrier wait.  The consuming kernel must then be
// called with b_stages_preloaded=true so its first-N-tile prologue issues only A
// per stage — group accounting mirrors the single-B invariant (kStage-1
// preload groups + kStage-1 A-only groups; cp_async_wait<kStage-2> completes all
// B groups + the stage-0 A group before the first fragment read).
template <typename Config>
__device__ __forceinline__
void gemm_dual_preload_b_stages(
    typename Config::T const* __restrict__ B0_ptr,
    typename Config::T const* __restrict__ B1_ptr,
    int N, int K,
    int ix,
    void* smem
) {
    using T = typename Config::T;

    constexpr int BN     = Config::BN;
    constexpr int BK     = Config::BK;
    constexpr int kStage = Config::kStage;

    using SmemLayoutA    = typename Config::SmemLayoutA;
    using SmemLayoutB    = typename Config::SmemLayoutB;
    using G2SCopyB       = typename Config::G2SCopyB;
    using G2SCopyB_ZFILL = typename Config::G2SCopyB_ZFILL;

    T* Ashm  = reinterpret_cast<T*>(smem);
    T* Bshm  = Ashm + cute::cosize(SmemLayoutA{});
    T* B1shm = Bshm + cute::cosize(SmemLayoutB{});

    Tensor B0 = make_tensor(make_gmem_ptr(B0_ptr),
                            make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor B1 = make_tensor(make_gmem_ptr(B1_ptr),
                            make_shape(N, K), make_stride(K, Int<1>{}));
    auto sB0 = make_tensor(make_smem_ptr(Bshm),  SmemLayoutB{});
    auto sB1 = make_tensor(make_smem_ptr(B1shm), SmemLayoutB{});

    Tensor gB0 = local_tile(B0, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    Tensor gB1 = local_tile(B1, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(threadIdx.x);
    auto tB0gB_copy = g2s_thr_copy_b.partition_S(gB0);
    auto tB0sB_copy = g2s_thr_copy_b.partition_D(sB0);
    auto tB1gB_copy = g2s_thr_copy_b.partition_S(gB1);
    auto tB1sB_copy = g2s_thr_copy_b.partition_D(sB1);

    G2SCopyB_ZFILL g2s_tiled_copy_b_zfill;

    const int ntile = (K + BK - 1) / BK;
    const bool k_ragged = (K % BK) != 0;
    const bool n_ragged = (ix * BN + BN) > N;
    const bool edge_b = k_ragged || n_ragged;

    Tensor cB = make_identity_tensor(make_shape(Int<BN>{}, Int<BK>{}));
    auto tBcB = g2s_thr_copy_b.partition_S(cB);
    auto tBpB = make_tensor<bool>(make_shape(size<1>(tB0sB_copy), size<2>(tB0sB_copy)));

    #pragma unroll
    for (int istage = 0; istage < kStage - 1; ++istage) {
        if (istage >= ntile) { cp_async_fence(); continue; }
        if (!edge_b) {
            cute::copy(g2s_tiled_copy_b, tB0gB_copy(_, _, _, istage), tB0sB_copy(_, _, _, istage));
            cute::copy(g2s_tiled_copy_b, tB1gB_copy(_, _, _, istage), tB1sB_copy(_, _, _, istage));
        } else {
            int kbase = istage * BK;
            CUTE_UNROLL
            for (int n = 0; n < size<0>(tBpB); ++n)
                CUTE_UNROLL
                for (int kk = 0; kk < size<1>(tBpB); ++kk)
                    tBpB(n, kk) = ((ix * BN + get<0>(tBcB(0, n, kk))) < N) &&
                                  ((kbase + get<1>(tBcB(0, n, kk))) < K);
            cute::copy_if(g2s_tiled_copy_b_zfill, tBpB, tB0gB_copy(_, _, _, istage), tB0sB_copy(_, _, _, istage));
            cute::copy_if(g2s_tiled_copy_b_zfill, tBpB, tB1gB_copy(_, _, _, istage), tB1sB_copy(_, _, _, istage));
        }
        cp_async_fence();
    }
}

// gemm_unified_dual_tile — single M-tile across [ix_start, ix_end) N-tiles,
// dual-B (gate/up) with fused gelu-mul merge then stock epilogue.
//   A_ptr  : [M,K] row-major   (X / normed)

//   B1_ptr : [N,K] row-major   (up_w)
//   C_out_ptr : [M,N] row-major (merged output)
//   epilogue : functor applied on the MERGED fp32 accumulator (use Identity)
template <typename Config, typename Merge, typename Epilogue>
__device__ __forceinline__
void gemm_unified_dual_tile(
    typename Config::T const* __restrict__ A_ptr,
    typename Config::T const* __restrict__ B0_ptr,
    typename Config::T const* __restrict__ B1_ptr,
    typename Config::T*       __restrict__ C_out_ptr,
    int M, int N, int K,
    int iy, int ix_start, int ix_end,
    Merge merge,
    Epilogue epilogue,
    void* smem,
    // When true, the first N tile's kStage-1 B0/B1 stages were
    // already issued by gemm_dual_preload_b_stages during the barrier wait, so
    // the prologue loads only A for that tile (group accounting preserved —
    // see the B-preload routine's comment).
    bool b_stages_preloaded = false
) {
    using T    = typename Config::T;

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

    // Shared-memory layout: A ring | B0 ring | B1 ring.
    T* Ashm  = reinterpret_cast<T*>(smem);
    T* Bshm  = Ashm + cute::cosize(SmemLayoutA{});
    T* B1shm = Bshm + cute::cosize(SmemLayoutB{});   // +1 B ring

    // CTA residency is validated by the concrete launcher.  The common core
    // only enforces the SM80 per-block opt-in ceiling.
    static_assert(DualSmem<Config>::kShmSizeDual
                      <= kAmpereMaxDynamicSmemPerBlock,
        "dual-B shared-memory footprint exceeds the SM80 per-block ceiling");

    int idx = threadIdx.x;

    Tensor A  = make_tensor(make_gmem_ptr(A_ptr),
                            make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B0 = make_tensor(make_gmem_ptr(B0_ptr),
                            make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor B1 = make_tensor(make_gmem_ptr(B1_ptr),
                            make_shape(N, K), make_stride(K, Int<1>{}));

    auto sA  = make_tensor(make_smem_ptr(Ashm),  SmemLayoutA{});
    auto sB0 = make_tensor(make_smem_ptr(Bshm),  SmemLayoutB{});
    auto sB1 = make_tensor(make_smem_ptr(B1shm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));

    // G2S copy setup (A + both B share the same thread/value layout)
    G2SCopyA g2s_tiled_copy_a;
    auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
    auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
    auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

    G2SCopyB g2s_tiled_copy_b;
    auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);

    G2SCopyA_ZFILL g2s_tiled_copy_a_zfill;
    G2SCopyB_ZFILL g2s_tiled_copy_b_zfill;

    // S2R copy setup: one A source and two B sources.
    auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_copy_a   = s2r_tiled_copy_a.get_slice(idx);
    auto tAsA_s2r = s2r_thr_copy_a.partition_S(sA);

    auto s2r_tiled_copy_b  = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_copy_b    = s2r_tiled_copy_b.get_slice(idx);
    auto tB0sB_s2r = s2r_thr_copy_b.partition_S(sB0);
    auto tB1sB_s2r = s2r_thr_copy_b.partition_S(sB1);

    int ntile = (K + BK - 1) / BK;
    const bool k_ragged = (K % BK) != 0;
    const bool m_ragged = (iy * BM + BM) > M;

    // ── N-tile loop ────────────────────────────────────────────────────────
    for (int ix = ix_start; ix < ix_end; ++ix) {
        Tensor gB0 = local_tile(B0, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
        Tensor gB1 = local_tile(B1, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
        auto tB0gB_copy = g2s_thr_copy_b.partition_S(gB0);
        auto tB0sB_copy = g2s_thr_copy_b.partition_D(sB0);
        auto tB1gB_copy = g2s_thr_copy_b.partition_S(gB1);
        auto tB1sB_copy = g2s_thr_copy_b.partition_D(sB1);

        const bool n_ragged = (ix * BN + BN) > N;

        Tensor cA = make_identity_tensor(make_shape(Int<BM>{}, Int<BK>{}));
        Tensor cB = make_identity_tensor(make_shape(Int<BN>{}, Int<BK>{}));
        auto tAcA = g2s_thr_copy_a.partition_S(cA);
        auto tBcB = g2s_thr_copy_b.partition_S(cB);

        auto tApA = make_tensor<bool>(make_shape(size<1>(tAsA_copy), size<2>(tAsA_copy)));
        auto tBpB = make_tensor<bool>(make_shape(size<1>(tB0sB_copy), size<2>(tB0sB_copy)));

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

        // Shared B predicate build (B0 and B1 have identical N/K coords) then
        // two copies into the two rings, in the same cp.async group as A.
        auto load_ktile_b01 = [&](int it, int iw) {
            if (!edge_b) {
                cute::copy(g2s_tiled_copy_b, tB0gB_copy(_, _, _, it), tB0sB_copy(_, _, _, iw));
                cute::copy(g2s_tiled_copy_b, tB1gB_copy(_, _, _, it), tB1sB_copy(_, _, _, iw));
            } else {
                int kbase = it * BK;
                CUTE_UNROLL
                for (int n = 0; n < size<0>(tBpB); ++n)
                    CUTE_UNROLL
                    for (int kk = 0; kk < size<1>(tBpB); ++kk)
                        tBpB(n, kk) = ((ix * BN + get<0>(tBcB(0, n, kk))) < N) &&
                                      ((kbase + get<1>(tBcB(0, n, kk))) < K);
                cute::copy_if(g2s_tiled_copy_b_zfill, tBpB, tB0gB_copy(_, _, _, it), tB0sB_copy(_, _, _, iw));
                cute::copy_if(g2s_tiled_copy_b_zfill, tBpB, tB1gB_copy(_, _, _, it), tB1sB_copy(_, _, _, iw));
            }
        };

        // load one K-tile `it` into stage `iw` — A + B0 + B1, one fence group.
        auto load_ktile = [&](int it, int iw) {
            load_ktile_a(it, iw);
            load_ktile_b01(it, iw);
        };

        // Fragments: shared A, two B operands, and two accumulators.
        T* C_dummy = Ashm;
        Tensor gC_fake = make_tensor(make_gmem_ptr(C_dummy),
                                     make_shape(Int<BM>{}, Int<BN>{}),
                                     make_stride(Int<BN>{}, Int<1>{}));
        auto tCrA  = thr_mma.partition_fragment_A(gA(_, _, 0));
        auto tCrB0 = thr_mma.partition_fragment_B(gB0(_, _, 0));
        auto tCrB1 = thr_mma.partition_fragment_B(gB1(_, _, 0));  // 2nd B frag
        auto tCrC0 = thr_mma.partition_fragment_C(gC_fake);       // gate accum
        auto tCrC1 = thr_mma.partition_fragment_C(gC_fake);       // up accum
        clear(tCrC0);
        clear(tCrC1);

        auto tCrA_view  = s2r_thr_copy_a.retile_D(tCrA);
        auto tCrB0_view = s2r_thr_copy_b.retile_D(tCrB0);
        auto tCrB1_view = s2r_thr_copy_b.retile_D(tCrB1);

        int itile_to_read = 0, ismem_read = 0, ismem_write = 0;

        // ── Prologue: prefill kStage-1 tiles (A + B0 + B1 per group) ──────────
        // If the dispatch loop preloaded this (first) N-tile's B rings during the
        // barrier wait, issue only A per stage (B groups already in flight).
        const bool use_preloaded_b = b_stages_preloaded && (ix == ix_start);
        #pragma unroll
        for (int istage = 0; istage < kStage - 1; ++istage) {
            if (istage < ntile) {
                load_ktile_a(istage, istage);
                if (!use_preloaded_b) {
                    load_ktile_b01(istage, istage);
                }
                ++itile_to_read; ++ismem_write;
            }
            // Commit an empty group when K has fewer than kStage-1 tiles so
            // wait_group retains the same stage accounting.
            cp_async_fence();
        }
        cp_async_wait<kStage - 2>();
        __syncthreads();

        // Preload first fragment (A + both B)
        cute::copy(s2r_tiled_copy_a, tAsA_s2r(_, _, 0, ismem_read),  tCrA_view(_, _, 0));
        cute::copy(s2r_tiled_copy_b, tB0sB_s2r(_, _, 0, ismem_read), tCrB0_view(_, _, 0));
        cute::copy(s2r_tiled_copy_b, tB1sB_s2r(_, _, 0, ismem_read), tCrB1_view(_, _, 0));

        // ── Main loop ────────────────────────────────────────────────────────
        #pragma unroll 1
        for (int itile = 0; itile < ntile; ++itile) {
            int nk = size<2>(tCrA);
            #pragma unroll
            for (int ik = 0; ik < nk; ++ik) {
                int ik_next = (ik + 1) % nk;

                // Boundary wait (BK64 path): drain + read-advance at ik==nk-1.
                if constexpr (!kEarly)
                if (ik == nk - 1) {
                    if (itile + kStage >= ntile + 1) {
                        cp_async_wait<0>();
                    } else {
                        cp_async_wait<kStage - 2>();
                    }
                    __syncthreads();
                    ismem_read = (ismem_read + 1) % kStage;
                }

                // Load the next A, B0, and B1 K fragments.
                cute::copy(s2r_tiled_copy_a, tAsA_s2r(_, _, ik_next, ismem_read),  tCrA_view(_, _, ik_next));
                cute::copy(s2r_tiled_copy_b, tB0sB_s2r(_, _, ik_next, ismem_read), tCrB0_view(_, _, ik_next));
                cute::copy(s2r_tiled_copy_b, tB1sB_s2r(_, _, ik_next, ismem_read), tCrB1_view(_, _, ik_next));

                // G2S issue next tile (one fence group for A+B0+B1)
                if (ik == 0 && itile_to_read < ntile) {
                    load_ktile(itile_to_read, ismem_write);
                    cp_async_fence();
                    ++itile_to_read;
                    ismem_write = (ismem_write + 1) % kStage;
                }

                // Shallow-K early boundary wait: one fragment ahead.
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

                // ── Dual MMA: shared A frag, two B frags, two accums ──────────
                cute::gemm(tiled_mma, tCrC0, tCrA(_, _, ik), tCrB0(_, _, ik), tCrC0);
                cute::gemm(tiled_mma, tCrC1, tCrA(_, _, ik), tCrB1(_, _, ik), tCrC1);
            }
        }

        // Drain any still-in-flight copies before reusing sA as C scratch.
        cp_async_wait<0>();
        __syncthreads();

        Tensor cCm   = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
        Tensor tCcCm = thr_mma.partition_C(cCm);
        CUTE_UNROLL
        for (int e = 0; e < size(tCrC0); ++e) {
            auto coord = tCcCm(e);
            const int m = iy * BM + get<0>(coord);
            tCrC0(e) = merge.transform(
                float(tCrC0(e)), float(tCrC1(e)), m);
        }
        auto& tCrC = tCrC0;

        // ── Scalar epilogue slot ──────────────────────────────────────────────
        {
            Tensor cC   = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
            Tensor tCcC = thr_mma.partition_C(cC);
            CUTE_UNROLL
            for (int e = 0; e < size(tCrC); ++e) {
                auto coord = tCcC(e);
                int m_global = iy * BM + get<0>(coord);
                int n_global = ix * BN + get<1>(coord);
                if (m_global < M && n_global < N) {
                    tCrC(e) = epilogue.transform(float(tCrC(e)), m_global, n_global);
                }
            }
        }

        // Vectorized R2S-to-S2G; sC aliases the drained A stage.
        auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

        auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
        auto r2s_thr_copy_c   = r2s_tiled_copy_c.get_slice(idx);
        auto tCrC_r2s         = r2s_thr_copy_c.retile_S(tCrC);
        auto tCsC_r2s         = r2s_thr_copy_c.partition_D(sC);

        S2GCopyC s2g_tiled_copy_c;
        auto s2g_thr_copy_c = s2g_tiled_copy_c.get_thread_slice(idx);
        auto tCsC_s2g       = s2g_thr_copy_c.partition_S(sC);

        Tensor gC = make_tensor(make_gmem_ptr(C_out_ptr),
                                make_shape(M, N), make_stride(N, Int<1>{}));
        Tensor gC_tile = local_tile(gC, make_tile(Int<BM>{}, Int<BN>{}), make_coord(iy, ix));
        auto tCgC_s2g  = s2g_thr_copy_c.partition_D(gC_tile);

        auto tCrC_r2sx = group_modes<1, 3>(tCrC_r2s);
        auto tCgC_s2gx = group_modes<1, 3>(tCgC_s2g);

        const bool tile_full = (iy * BM + BM <= M) && (ix * BN + BN <= N);

        Tensor cTile = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
        auto tCcTile  = s2g_thr_copy_c.partition_D(cTile);
        auto tCcTilex = group_modes<1, 3>(tCcTile);

        int step = size<3>(tCsC_r2s);

        #pragma unroll
        for (int i = 0; i < size<1>(tCrC_r2sx); i += step) {
            #pragma unroll
            for (int j = 0; j < step; ++j) {
                if (i + j >= size<1>(tCrC_r2sx)) break;
                auto t = make_tensor_like<T>(tCrC_r2sx(_, i + j));
                CUTE_UNROLL
                for (int e = 0; e < size(t); ++e) {
                    t(e) = T(float(tCrC_r2sx(e, i + j)));
                }
                cute::copy(r2s_tiled_copy_c, t, tCsC_r2s(_, 0, 0, j));
            }
            __syncthreads();

            #pragma unroll
            for (int j = 0; j < step; ++j) {
                if (i + j >= size<1>(tCrC_r2sx)) break;
                if constexpr (kS2gRounds == 1) {
                    if (tile_full) {
                        cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
                    } else {
                        auto c0 = tCcTilex(0, i + j);
                        int m0 = iy * BM + get<0>(c0);
                        int n0 = ix * BN + get<1>(c0);
                        if (m0 < M && n0 < N) {
                            cute::copy(s2g_tiled_copy_c, tCsC_s2g(_, 0, 0, j), tCgC_s2gx(_, i + j));
                        }
                    }
                } else {
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
                        }
                    }
                }
            }
            __syncthreads();
        }
    }  // N-tile loop
}

}  // namespace mk_gemm
