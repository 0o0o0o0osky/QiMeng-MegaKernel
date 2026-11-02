// gemm_splitk.cuh — split-K Tensor Core GEMM for SM80.
// Each partial instruction owns one K slice and writes an fp32 workspace plane.
// A separate reducer sums the planes and invokes a caller-selected epilogue.
// The common implementation does not prescribe the final transform or fusion
// placement. Ragged M, N, and K edges are predicated with cp.async zero fill.

#pragma once

#include <cute/tensor.hpp>
#include "common/execution_config.cuh"

using namespace cute;

namespace mk_gemm {

// K-slice of split k_split_idx out of k_splits, in units of BK-tiles.
// Even ceil-split; the (possibly ragged) final tile lands in the last split.
// Returns {k_start_tile, k_tile_count} (count may be 0 for degenerate splits).
struct SplitKSlice { int k_start_tile; int k_tile_count; };

__host__ __device__ __forceinline__
SplitKSlice splitk_k_slice(int K, int BK, int k_splits, int k_split_idx) {
    int k_tiles = (K + BK - 1) / BK;
    int k_per   = (k_tiles + k_splits - 1) / k_splits;
    int k0      = k_split_idx * k_per;
    int kc      = (k0 + k_per > k_tiles) ? (k_tiles - k0) : k_per;
    if (kc < 0) kc = 0;
    return {k0, kc};
}

template <typename Config>
__device__ __forceinline__
void gemm_splitk_preload_b_stages(
    typename Config::T const* __restrict__ B_ptr,
    int N, int K,
    int ix,
    int k_start_tile, int k_tile_count,
    void* smem)
{
    using T = typename Config::T;
    constexpr int BM=Config::BM, BN=Config::BN, BK=Config::BK, kStage=Config::kStage;
    using SmemLayoutA = typename Config::SmemLayoutA;
    using SmemLayoutB = typename Config::SmemLayoutB;
    using G2SCopyB    = typename Config::G2SCopyB;

    T* Ashm = reinterpret_cast<T*>(smem);
    T* Bshm = Ashm + cute::cosize(SmemLayoutA{});

    Tensor B = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(K, Int<1>{}));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    G2SCopyB g2s_b;
    auto g2s_thr_b = g2s_b.get_slice(threadIdx.x);
    auto tBgB = g2s_thr_b.partition_S(gB);
    auto tBsB = g2s_thr_b.partition_D(sB);

    const int k_end = k_start_tile + k_tile_count;
    #pragma unroll
    for (int s=0; s<kStage-1; ++s) {
        int it = k_start_tile + s;
        if (it >= k_end) { cp_async_fence(); continue; }
        cute::copy(g2s_b, tBgB(_, _, _, it), tBsB(_, _, _, s));
        cp_async_fence();
    }
}

// Partial GEMM over a K-slice. Writes fp32 C-tile to scratch_ks[m_global*N + n_global].
// One CTA computes one (M-tile, N-tile) over its K-slice. Grid: (num_n, num_m, k_splits).
template <typename Config>
__device__ __forceinline__
void gemm_splitk_partial(
    typename Config::T const* __restrict__ A_ptr,
    typename Config::T const* __restrict__ B_ptr,
    float* __restrict__ scratch_ks,   // points at scratch base for THIS k-split: [M, N] fp32
    int M, int N, int K,
    int iy, int ix,
    int k_start_tile, int k_tile_count,
    void* smem,
    bool b_stages_preloaded = false,
    bool atomic_accumulate = false)
{
    using T = typename Config::T;
    constexpr int BM=Config::BM, BN=Config::BN, BK=Config::BK, kStage=Config::kStage;
    using SmemLayoutA  = typename Config::SmemLayoutA;
    using SmemLayoutB  = typename Config::SmemLayoutB;
    using TiledMMA     = typename Config::TiledMMA;
    using G2SCopyA     = typename Config::G2SCopyA;
    using G2SCopyB     = typename Config::G2SCopyB;
    using S2RCopyAtomA = typename Config::S2RCopyAtomA;
    using S2RCopyAtomB = typename Config::S2RCopyAtomB;

    T* Ashm = reinterpret_cast<T*>(smem);
    T* Bshm = Ashm + cute::cosize(SmemLayoutA{});
    int idx = threadIdx.x;

    Tensor A = make_tensor(make_gmem_ptr(A_ptr), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(B_ptr), make_shape(N, K), make_stride(K, Int<1>{}));
    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(idx);

    Tensor gA = local_tile(A, make_tile(Int<BM>{}, Int<BK>{}), make_coord(iy, _));
    Tensor gB = local_tile(B, make_tile(Int<BN>{}, Int<BK>{}), make_coord(ix, _));

    G2SCopyA g2s_a; auto g2s_thr_a = g2s_a.get_slice(idx);
    auto tAgA = g2s_thr_a.partition_S(gA);
    auto tAsA = g2s_thr_a.partition_D(sA);
    G2SCopyB g2s_b; auto g2s_thr_b = g2s_b.get_slice(idx);
    auto tBgB = g2s_thr_b.partition_S(gB);
    auto tBsB = g2s_thr_b.partition_D(sB);

    auto s2r_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
    auto s2r_thr_a = s2r_a.get_slice(idx);
    auto tAsA_s2r = s2r_thr_a.partition_S(sA);
    auto s2r_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
    auto s2r_thr_b = s2r_b.get_slice(idx);
    auto tBsB_s2r = s2r_thr_b.partition_S(sB);

    auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
    T* C_dummy = Ashm;
    Tensor gC_fake = make_tensor(make_gmem_ptr(C_dummy),
                                 make_shape(Int<BM>{}, Int<BN>{}), make_stride(Int<BN>{}, Int<1>{}));
    auto tCrC = thr_mma.partition_fragment_C(gC_fake);
    clear(tCrC);
    auto tCrA_view = s2r_thr_a.retile_D(tCrA);
    auto tCrB_view = s2r_thr_b.retile_D(tCrB);

    // Ragged-K predication tensors (mirror gemm_unified_tile::load_ktile).
    const bool k_ragged = (K % BK) != 0;
    // M/N-edge: the (iy,ix) tile may extend past M/N (e.g. decoder chunk M=50 <
    // BM=64).  local_tile does NOT clamp, so an unpredicated A load would read
    // gmem rows [M, iy*BM+BM) out of bounds → non-deterministic partials / faults.
    const bool m_ragged = (iy * BM + BM) > M;
    const bool n_ragged = (ix * BN + BN) > N;
    // ZFILL G2S copies for the M/N-edge path — cp.async zero-fills predicated-off
    // lanes, so it is the sole writer of each smem address (same thread/value
    // layout as the non-ZFILL copies; mirrors gemm_unified_tile).
    typename Config::G2SCopyA_ZFILL g2s_a_zfill;
    typename Config::G2SCopyB_ZFILL g2s_b_zfill;
    Tensor cA = make_identity_tensor(make_shape(Int<BM>{}, Int<BK>{}));
    Tensor cB = make_identity_tensor(make_shape(Int<BN>{}, Int<BK>{}));
    auto tAcA = g2s_thr_a.partition_S(cA);
    auto tBcB = g2s_thr_b.partition_S(cB);
    auto tApA = make_tensor<bool>(make_shape(size<1>(tAsA), size<2>(tAsA)));
    auto tBpB = make_tensor<bool>(make_shape(size<1>(tBsB), size<2>(tBsB)));

    auto load_ktile_a = [&](int it, int iw) {
        if (m_ragged) {
            // M-edge: predicate rows AND K, ZFILL (all K-tiles of this M-tile).
            int kbase = it * BK;
            CUTE_UNROLL
            for (int m = 0; m < size<0>(tApA); ++m)
                CUTE_UNROLL
                for (int kk = 0; kk < size<1>(tApA); ++kk)
                    tApA(m, kk) = ((iy * BM + get<0>(tAcA(0, m, kk))) < M) &&
                                  ((kbase + get<1>(tAcA(0, m, kk))) < K);
            cute::copy_if(g2s_a_zfill, tApA, tAgA(_, _, _, it), tAsA(_, _, _, iw));
        } else if (!k_ragged || (it + 1) * BK <= K) {
            cute::copy(g2s_a, tAgA(_, _, _, it), tAsA(_, _, _, iw));
        } else {
            int kbase = it * BK;
            CUTE_UNROLL
            for (int m = 0; m < size<0>(tApA); ++m)
                CUTE_UNROLL
                for (int kk = 0; kk < size<1>(tApA); ++kk)
                    tApA(m, kk) = (kbase + get<1>(tAcA(0, m, kk))) < K;
            clear(tAsA(_, _, _, iw));
            cute::copy_if(g2s_a, tApA, tAgA(_, _, _, it), tAsA(_, _, _, iw));
        }
    };

    auto load_ktile_b = [&](int it, int iw) {
        if (n_ragged) {
            // N-edge: predicate cols AND K, ZFILL (all K-tiles of this N-tile).
            int kbase = it * BK;
            CUTE_UNROLL
            for (int n = 0; n < size<0>(tBpB); ++n)
                CUTE_UNROLL
                for (int kk = 0; kk < size<1>(tBpB); ++kk)
                    tBpB(n, kk) = ((ix * BN + get<0>(tBcB(0, n, kk))) < N) &&
                                  ((kbase + get<1>(tBcB(0, n, kk))) < K);
            cute::copy_if(g2s_b_zfill, tBpB, tBgB(_, _, _, it), tBsB(_, _, _, iw));
        } else if (!k_ragged || (it + 1) * BK <= K) {
            cute::copy(g2s_b, tBgB(_, _, _, it), tBsB(_, _, _, iw));
        } else {
            int kbase = it * BK;
            CUTE_UNROLL
            for (int n = 0; n < size<0>(tBpB); ++n)
                CUTE_UNROLL
                for (int kk = 0; kk < size<1>(tBpB); ++kk)
                    tBpB(n, kk) = (kbase + get<1>(tBcB(0, n, kk))) < K;
            clear(tBsB(_, _, _, iw));
            cute::copy_if(g2s_b, tBpB, tBgB(_, _, _, it), tBsB(_, _, _, iw));
        }
    };

    const int k_end = k_start_tile + k_tile_count;  // exclusive (in K-tiles)
    int itile = k_start_tile, ismem_read=0, ismem_write=0;

    // Prologue: prefill kStage-1 tiles from the K-slice.
    #pragma unroll
    for (int s=0; s<kStage-1; ++s) {
        if (itile < k_end) {
            load_ktile_a(itile, ismem_write);
            if (!b_stages_preloaded) {
                load_ktile_b(itile, ismem_write);
            }
            cp_async_fence();
            ++itile; ++ismem_write;
        } else {
            cp_async_fence();
        }
    }
    cp_async_wait<kStage-2>();
    __syncthreads();
    cute::copy(s2r_a, tAsA_s2r(_,_,0,ismem_read), tCrA_view(_,_,0));
    cute::copy(s2r_b, tBsB_s2r(_,_,0,ismem_read), tCrB_view(_,_,0));

    #pragma unroll 1
    for (int kt=0; kt<k_tile_count; ++kt) {
        int nk = size<2>(tCrA);
        #pragma unroll
        for (int ik=0; ik<nk; ++ik) {
            int ik_next=(ik+1)%nk;
            if (ik==nk-1) {
                // Tail drain fix (mirrors gemm_unified_tile): once prefetch stops
                // issuing, the outstanding-group count shrinks and wait<kStage-2>
                // can return before the next tile's K-chunk lands — the MMA then
                // consumes it early (rare, timing-dependent nondeterminism).
                // Tile kt+1 is guaranteed landed iff kt <= k_tile_count - kStage;
                // drain fully on every later transition.
                if (kt + kStage >= k_tile_count + 1) {
                    cp_async_wait<0>();
                } else {
                    cp_async_wait<kStage-2>();
                }
                __syncthreads();
                ismem_read=(ismem_read+1)%kStage;
            }
            cute::copy(s2r_a, tAsA_s2r(_,_,ik_next,ismem_read), tCrA_view(_,_,ik_next));
            cute::copy(s2r_b, tBsB_s2r(_,_,ik_next,ismem_read), tCrB_view(_,_,ik_next));
            if (ik==0 && itile < k_end) {
                load_ktile_a(itile, ismem_write);
                load_ktile_b(itile, ismem_write);
                cp_async_fence();
                ++itile; ismem_write=(ismem_write+1)%kStage;
            }
            cute::gemm(tiled_mma, tCrC, tCrA(_,_,ik), tCrB(_,_,ik), tCrC);
        }
    }

    // Scatter fp32 accumulator directly to scratch[m_global*N + n_global] (MMA-fragment order).
    Tensor cC = make_identity_tensor(make_shape(Int<BM>{}, Int<BN>{}));
    Tensor tCcC = thr_mma.partition_C(cC);
    CUTE_UNROLL
    for (int e=0; e<size(tCrC); ++e) {
        auto coord = tCcC(e);
        int m_global = iy*BM + get<0>(coord);
        int n_global = ix*BN + get<1>(coord);
        if (m_global < M && n_global < N) {
            float* dst = scratch_ks + (long)m_global * N + n_global;
            if (atomic_accumulate) {
                atomicAdd(dst, float(tCrC(e)));
            } else {
                *dst = float(tCrC(e));
            }
        }
    }
}

// ── Tile-level API: k_splits as a runtime parameter ─────────────────────────
// One CTA computes one (M-tile, N-tile, k_split) work unit.  The K-slice and
// the fp32 scratch plane offset are derived internally.
// scratch layout: [k_splits, M, N] fp32 (plane stride = M*N).

template <typename Config>
__device__ __forceinline__
void gemm_splitk_preload_b_stages_tile(
    typename Config::T const* __restrict__ B_ptr,
    int N, int K,
    int ix,
    int k_split_idx, int k_splits,
    void* smem)
{
    SplitKSlice s = splitk_k_slice(K, Config::BK, k_splits, k_split_idx);
    if (s.k_tile_count <= 0) return;
    gemm_splitk_preload_b_stages<Config>(B_ptr, N, K, ix, s.k_start_tile, s.k_tile_count, smem);
}

template <typename Config>
__device__ __forceinline__
void gemm_splitk_partial_tile(
    typename Config::T const* __restrict__ A_ptr,
    typename Config::T const* __restrict__ B_ptr,
    float* __restrict__ scratch,      // base of [k_splits, M, N] fp32
    int M, int N, int K,
    int iy, int ix,
    int k_split_idx, int k_splits,
    void* smem,
    bool b_stages_preloaded = false,
    bool atomic_accumulate = false)
{
    SplitKSlice s = splitk_k_slice(K, Config::BK, k_splits, k_split_idx);
    if (s.k_tile_count <= 0) return;
    float* scratch_ks = scratch + (long)k_split_idx * M * N;
    gemm_splitk_partial<Config>(A_ptr, B_ptr, scratch_ks, M, N, K, iy, ix,
                                s.k_start_tile, s.k_tile_count, smem,
                                b_stages_preloaded, atomic_accumulate);
}

// Reduce with a pluggable epilogue functor (gemm_epilogue.cuh interface:
// float transform(float acc, int m_global, int n_global)).  Scratch loads and
// the final store stay 8-wide vectorized; the epilogue is applied per element.
// N must be a multiple of 8 so an 8-group never crosses a row boundary.
template <typename T, typename Epilogue>
__device__ __forceinline__
void splitk_reduce_tile(
    const float* __restrict__ scratch,   // [k_splits, M, N] fp32
    T* __restrict__ C,                   // [M, N]
    int M, int N, int k_splits,
    int chunk_idx, int num_chunks,
    Epilogue const& epilogue)
{
    const long total_vec = ((long)M * N) / 8;
    const long per       = (total_vec + num_chunks - 1) / num_chunks;
    const long g0        = (long)chunk_idx * per;
    long g1              = g0 + per;
    if (g1 > total_vec) g1 = total_vec;
    const long plane     = (long)M * N;

    for (long gi = g0 + threadIdx.x; gi < g1; gi += blockDim.x) {
        const long i = gi * 8;
        float acc[8];
        #pragma unroll
        for (int e=0;e<8;++e) acc[e]=0.f;
        #pragma unroll 1
        for (int ks=0; ks<k_splits; ++ks) {
            const float* s = scratch + (long)ks*plane + i;
            float4 lo = *reinterpret_cast<const float4*>(s);
            float4 hi = *reinterpret_cast<const float4*>(s+4);
            acc[0]+=lo.x;acc[1]+=lo.y;acc[2]+=lo.z;acc[3]+=lo.w;
            acc[4]+=hi.x;acc[5]+=hi.y;acc[6]+=hi.z;acc[7]+=hi.w;
        }
        const int m = (int)(i / N);
        const int n = (int)(i % N);                  // multiple of 8
        T ob[8];
        #pragma unroll
        for (int e=0;e<8;++e) {
            ob[e] = T(epilogue.transform(acc[e], m, n + e));
        }
        *reinterpret_cast<uint4*>(reinterpret_cast<T*>(C)+i) =
            *reinterpret_cast<const uint4*>(ob);
    }
}

}  // namespace mk_gemm
