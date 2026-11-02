// QKV GEMM (M16N16 tensor-core pipeline) + RoPE for multi-batch decode.
//
// Instruction word layout (QKVProj_GEMM_RoPE.serialize()):
//   word[0] = opcode = 2
//   word[1] = layer_idx
//   word[2] = batch_size
//   word[3] = tile_M_idx        (always 0)
//   word[4] = tile_N_start_idx
//   word[5] = tile_N_end_idx
//   word[6] = tile_M_granularity (16)
//   word[7] = tile_N_granularity (16)

#include "globals.cuh"
#include "llama_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"
#include "matmul_pipeline.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

#include <cuda_bf16.h>

namespace megakernel {

using namespace cute;

// =============================================================================
// compute_gemm_tile<GemmCfg>
//
// Runs the 3-stage B G2S pipeline + MMA over all K-tiles.
// After GEMM, writes tCrD to smem_C using CuTe coordinate predicate
// (batch_size < kTileM rows are skipped), then all threads scatter to
// global Q/K/V with RoPE applied.
// =============================================================================

// Inner phases (align with ampere_skeleton): Prologue → Main loop (issue/wait/compute)
// → Epilogue (R2S-style smem C + post-fusion global scatter).
template <typename GemmCfg>
__device__ __forceinline__
void compute_gemm_tile(
    int              tid,
    int              batch_size,
    uint8_t* __restrict__ smem_base,
    const bfloat16_t* __restrict__ gmem_A_ptr,  // [batch_size, HIDDEN_DIM]
    const bfloat16_t* __restrict__ gmem_B_ptr,  // [kTileN_, HIDDEN_DIM]
    int              tile_n_global,              // first output column in QKV space
    int              layer_idx,
    LlamaGlobals&    g,
    int              sm_id,
    int              inst_idx
) {
    using T          = typename GemmCfg::T;
    using TiledMMA   = typename GemmCfg::MMA;
    using SmemLB     = typename GemmCfg::SmemLayoutB;
    using SmemLCRaw = typename GemmCfg::SmemLayoutCRaw;
    using S2RA       = typename GemmCfg::S2RCopyAtomA;
    using S2RB       = typename GemmCfg::S2RCopyAtomB;
    using G2SCB      = typename GemmCfg::G2SCopyB;

    constexpr int kTileM_   = KernelConfig::kTileM;
    constexpr int kTileN_   = GemmCfg::kTileN_;
    constexpr int kTileK    = GemmCfg::kTileK;
    constexpr int kStage    = GemmCfg::kStage;
    constexpr int HIDDEN    = LlamaConfig::HIDDEN_DIM;
    constexpr int ntile     = HIDDEN / kTileK;  // 32
    constexpr int kATileCount = kQkvStagedA ? kStage : ntile;
    using SmemLA = GemmSharedBase::SmemLayoutATileRing<kATileCount>;

    // ── GEMM / GEMV Prologue ───────────────────────────────────────────────
    // Preload (kStage - 1) B tiles via G2S before the main K-loop.

    // ------------------------------------------------------------------
    // Smem tensors
    // ------------------------------------------------------------------
    T* Ashm = smem_ptr_at<T>(smem_base, QkvGemmSmemLayout::kOffsetA);
    T* Bshm = smem_ptr_at<T>(smem_base, QkvGemmSmemLayout::kOffsetStagedB);
    T* Cshm_raw = smem_ptr_at<T>(smem_base, QkvGemmSmemLayout::kOffsetOutputScratch);

    // A: [kTileM, HIDDEN_DIM] swizzled (row 0 from standalone RMS or fused rms_norm_batched).
    // Tile A along K: [kTileM, kTileK, ntile]
    auto sA_tiles = make_tensor(make_smem_ptr(Ashm), SmemLA{});

    // B: [kTileN_, kTileK, kStage] swizzled
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLB{});

    // C: Raw tile for MMA partition_C stores; Step2 reads same plane vectorized.
    auto sC_raw = make_tensor(make_smem_ptr(Cshm_raw), SmemLCRaw{});

    // ------------------------------------------------------------------
    // Global B: [kTileN_, HIDDEN_DIM] → tiled along K
    // ------------------------------------------------------------------
    auto gB = make_tensor(
        make_gmem_ptr(gmem_B_ptr),
        make_shape(Int<kTileN_>{}, Int<HIDDEN>{}),
        make_stride(Int<HIDDEN>{}, Int<1>{})
    );
    auto gB_tiles = local_tile(
        gB,
        make_tile(Int<kTileN_>{}, Int<kTileK>{}),
        make_coord(0, _)
    );  // [kTileN_, kTileK, ntile]

    // Dummy C for partition_fragment_C: same Raw smem base as MMA C operand.
    auto gC_dummy = make_tensor(
        make_smem_ptr(Cshm_raw),
        make_shape(Int<kTileM_>{}, Int<kTileN_>{}),
        LayoutRight{}
    );

    // ------------------------------------------------------------------
    // MMA + S2R setup
    // ------------------------------------------------------------------
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(sA_tiles(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gC_dummy);
    clear(tCrD);

    auto s2r_copy_a    = make_tiled_copy_A(S2RA{}, tiled_mma);
    auto s2r_thr_cpy_a = s2r_copy_a.get_slice(tid);
    auto tAsA          = s2r_thr_cpy_a.partition_S(sA_tiles);  // (CPY, M, K, ntile)
    auto tCrA_view     = s2r_thr_cpy_a.retile_D(tCrA);         // (CPY, M, K)

    auto s2r_copy_b    = make_tiled_copy_B(S2RB{}, tiled_mma);
    auto s2r_thr_cpy_b = s2r_copy_b.get_slice(tid);
    auto tBsB          = s2r_thr_cpy_b.partition_S(sB);          // (CPY, N, K, kStage)
    auto tCrB_view     = s2r_thr_cpy_b.retile_D(tCrB);           // (CPY, N, K)

    // ------------------------------------------------------------------
    // G2S setup for B
    // ------------------------------------------------------------------
    G2SCB g2s_copy_b;
    auto g2s_thr_cpy_b = g2s_copy_b.get_slice(tid);
    auto tBgB_copy     = g2s_thr_cpy_b.partition_S(gB_tiles);   // (CPY, N, K, ntile)
    auto tBsB_copy     = g2s_thr_cpy_b.partition_D(sB);          // (CPY, N, K, kStage)

    // ── (Prologue cont.) Issue initial G2S copies ──────────────────────────
    // ------------------------------------------------------------------
    // 3-stage G2S prologue: preload first (kStage-1) B tiles
    // ------------------------------------------------------------------
    CpAsyncStageRing<kStage> pipeline;

#ifdef ENABLE_PROFILING
    uint64_t prof_stage_begin = 0;
#endif

    #pragma unroll
    for (int istage = 0; istage < kStage - 1 && pipeline.has_next(ntile); ++istage) {
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            prof_stage_begin = prof_clock_now();
        }
#endif
        if constexpr (kQkvStagedA) {
            stage_activation_tile_async<HIDDEN>(
                gmem_A_ptr,
                Ashm,
                batch_size,
                pipeline.next_tile,
                pipeline.write_stage);
        }
        issue_gemm_g2s_128(
            g2s_copy_b,
            tBgB_copy(_, _, _, pipeline.next_tile),
            tBsB_copy(_, _, _, pipeline.write_stage));
        pipeline.commit_issue_step();
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_STAGING, prof_clock_now() - prof_stage_begin);
        }
#endif
        pipeline.advance_write();
    }

#ifdef ENABLE_PROFILING
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    pipeline.wait_for_read_stage();
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_SYNC, prof_clock_now() - prof_stage_begin);
    }
#endif

    // ── GEMM Main Loop ─────────────────────────────────────────────────────
    // For each K-tile: Phase 1 — issue next G2S; Phase 2 — wait cp.async;
    // Phase 3 — S2R + MMA on the current stage. (Mirrors ampere_skeleton phases.)
    // ------------------------------------------------------------------
    // Main K-loop (3-stage software pipeline for B; A reads directly from smem)
    //
    // The shared stage-ring specialization commits every logical issue step,
    // including empty tail groups, so wait-group distance stays valid.
    // ------------------------------------------------------------------
    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);  // kTileK / kMmaPK = 64/16 = 4

        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
#ifdef ENABLE_PROFILING
            if (tid == 0) {
                prof_stage_begin = prof_clock_now();
            }
#endif
            constexpr int mma_size = decltype(cute::size(TiledMMA{}))::value;
            const int a_read_tile = kQkvStagedA ? pipeline.read_stage : itile;
            if (tid < mma_size) {
                cute::copy(s2r_copy_a, tAsA(_, _, ik, a_read_tile), tCrA_view(_, _, ik));
                cute::copy(s2r_copy_b, tBsB(_, _, ik, pipeline.read_stage), tCrB_view(_, _, ik));
            }

            // Issue G2S for the next B tile at the first inner step.
            if (ik == 0) {
                if (pipeline.has_next(ntile)) {
                    if constexpr (kQkvStagedA) {
                        stage_activation_tile_async<HIDDEN>(
                            gmem_A_ptr,
                            Ashm,
                            batch_size,
                            pipeline.next_tile,
                            pipeline.write_stage);
                    }
                    issue_gemm_g2s_128(
                        g2s_copy_b,
                        tBgB_copy(_, _, _, pipeline.next_tile),
                        tBsB_copy(_, _, _, pipeline.write_stage));
                    pipeline.advance_write();
                }
                pipeline.commit_issue_step();
            }
#ifdef ENABLE_PROFILING
            if (tid == 0) {
                PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_STAGING, prof_clock_now() - prof_stage_begin);
            }
#endif

            // Advance smem read slot after the last inner step.
            if (ik == nk - 1) {
#ifdef ENABLE_PROFILING
                if (tid == 0) {
                    prof_stage_begin = prof_clock_now();
                }
#endif
                pipeline.wait_for_read_stage();
                pipeline.advance_read();
#ifdef ENABLE_PROFILING
                if (tid == 0) {
                    PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_SYNC, prof_clock_now() - prof_stage_begin);
                }
#endif
            }

#ifdef ENABLE_PROFILING
            if (tid == 0) {
                prof_stage_begin = prof_clock_now();
            }
#endif
            if (tid < mma_size) {
                cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
            }
#ifdef ENABLE_PROFILING
            if (tid == 0) {
                PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_MMA, prof_clock_now() - prof_stage_begin);
            }
#endif
        }
    }

#if DEBUG_PRINT
    if (tid == 0 && blockIdx.x == 0) {
        printf("[gemm tile_n=%d] tCrD[0..3] = %.4f %.4f %.4f %.4f\n",
               tile_n_global,
               float(tCrD(0)), float(tCrD(1)),
               float(tCrD(2)), float(tCrD(3)));
    }
#endif

    // ── Epilogue + Post-Fusion Stage ───────────────────────────────────────
    // Step 1: tCrD (f32) → sC (bf16) with CuTe coord predicate (batch rows).
    // ------------------------------------------------------------------
    // Epilogue — Step 1: tCrD (float32 regs) → sC (bf16 smem, row-major)
    //
    // Uses a CuTe identity-coordinate tensor to predicate writes: only
    // elements whose M coordinate < batch_size and row_active are stored.
    // Handles batch_size < kTileM=16 and inactive-masked rows (PyVM skip writes).
    //
    // Non-MMA threads (tid >= size(TiledMMA{})) must not write at all:
    // CuTe maps them modularly to MMA-thread positions and would alias.
    // ------------------------------------------------------------------
 #ifdef ENABLE_PROFILING
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    {
        constexpr int mma_size = decltype(cute::size(TiledMMA{}))::value;
        if (tid < mma_size) {
            // CuTe identity tensor: shape (kMmaPM, kMmaPN), value = coord tuple.
            auto cC    = make_identity_tensor(shape(sC_raw));
            auto tCcC  = thr_mma.partition_C(cC);   // thread's coordinate view
            auto tCsC  = thr_mma.partition_C(sC_raw);   // thread's smem view (Raw)
            CUTE_UNROLL
            for (int n = 0; n < (int)size(tCrD); ++n) {
                // get<0>: M dimension index within the kMmaPM×kMmaPN tile.
                int mm = get<0>(tCcC(n));
                if (mm < batch_size && row_active(g, mm)) {
                    tCsC(n) = T(float(tCrD(n)));
                }
            }
        }
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_EPILOGUE_SC_CYCLES(g, sm_id, inst_idx, prof_clock_now() - prof_stage_begin);
        prof_stage_begin = prof_clock_now();
    }
#endif
    __syncthreads();
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_SYNC, prof_clock_now() - prof_stage_begin);
        prof_stage_begin = prof_clock_now();
    }
#endif
    // Step 2: smem C (Raw plane) → global (RoPE for Q/K, direct for V).
    // ------------------------------------------------------------------
    // Vectorized CuTe load uses RowMajor Raw (SmemLayoutCRaw).
    // Neighbor column: scalar Raw read (n_partner = n_local ^ 1).
    //
    // All NUM_THREADS scatter kTileM_ × kTileN_; thread t owns flat
    // [t * kPerThread, (t+1) * kPerThread).
    // ------------------------------------------------------------------
    constexpr int HeadDim = LlamaConfig::HEAD_DIM;
    constexpr int Q_End   = LlamaConfig::HIDDEN_DIM;
    constexpr int K_End   = Q_End + LlamaConfig::NUM_KV_HEADS * HeadDim;
    constexpr int V_End   = K_End + LlamaConfig::NUM_KV_HEADS * HeadDim;

    constexpr int kOutElems  = kTileM_ * kTileN_;
    constexpr int kPerThread = kOutElems / KernelConfig::NUM_THREADS;

    auto t_q_out      = g.get_q_post_rope();                // [B, HIDDEN_DIM]
    auto t_k_cache    = g.get_k_cache()(layer_idx, _, _, _, _);  // [B, Seq, KVH, D]
    auto t_v_cache    = g.get_v_cache()(layer_idx, _, _, _, _);
    const float* rope_cos_ldg = g.rope_cos_ptr;
    const float* rope_sin_ldg = g.rope_sin_ptr;

    float local_c[kPerThread];
    float local_nei[kPerThread];

    // Vectorized loads from RowMajor Raw plane.
    // Linear index (m,n) -> m*kTileN_+n matches LayoutRight SmemLayoutCRaw; avoid
    // CuTe tiled scatter here (partition order vs flat_idx mismatched in practice).
    static_assert(
        kPerThread == 1 || kPerThread == 2 || kPerThread == 4 || kPerThread == 8,
        "kTileM*kTileN/NUM_THREADS must be 1, 2, 4, or 8 for epilogue loads");
    const T* raw_line = Cshm_raw;
    const int flat0   = tid * kPerThread;
    if constexpr (kPerThread == 1) {
        __nv_bfloat16 const h = *reinterpret_cast<__nv_bfloat16 const*>(raw_line + flat0);
        float const           f = __bfloat162float(h);
        int const             m = flat0 / kTileN_;
        local_c[0]              = (m < batch_size && row_active(g, m)) ? f : 0.f;
    } else if constexpr (kPerThread == 8) {
        uint4 const v = *reinterpret_cast<uint4 const*>(raw_line + flat0);
        __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&v);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            float f0 = __bfloat162float(__low2bfloat16(pairs[e]));
            float f1 = __bfloat162float(__high2bfloat16(pairs[e]));
            int fl0  = flat0 + 2 * e;
            int fl1  = flat0 + 2 * e + 1;
            int m0   = fl0 / kTileN_;
            int m1   = fl1 / kTileN_;
            local_c[2 * e] =
                (m0 < batch_size && row_active(g, m0)) ? f0 : 0.f;
            local_c[2 * e + 1] =
                (m1 < batch_size && row_active(g, m1)) ? f1 : 0.f;
        }
    } else if constexpr (kPerThread == 4) {
        uint64_t const w =
            *reinterpret_cast<uint64_t const*>(raw_line + flat0);
        __nv_bfloat162 const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&w);
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            float f0 = __bfloat162float(__low2bfloat16(pairs[e]));
            float f1 = __bfloat162float(__high2bfloat16(pairs[e]));
            int fl0  = flat0 + 2 * e;
            int fl1  = flat0 + 2 * e + 1;
            int m0   = fl0 / kTileN_;
            int m1   = fl1 / kTileN_;
            local_c[2 * e] =
                (m0 < batch_size && row_active(g, m0)) ? f0 : 0.f;
            local_c[2 * e + 1] =
                (m1 < batch_size && row_active(g, m1)) ? f1 : 0.f;
        }
    } else {
        // kPerThread == 2
        uint32_t const w = *reinterpret_cast<uint32_t const*>(raw_line + flat0);
        __nv_bfloat162 const h2 = *reinterpret_cast<__nv_bfloat162 const*>(&w);
        float f0 = __bfloat162float(__low2bfloat16(h2));
        float f1 = __bfloat162float(__high2bfloat16(h2));
        int m0   = flat0 / kTileN_;
        int m1   = (flat0 + 1) / kTileN_;
        local_c[0] =
            (m0 < batch_size && row_active(g, m0)) ? f0 : 0.f;
        local_c[1] =
            (m1 < batch_size && row_active(g, m1)) ? f1 : 0.f;
    }
#pragma unroll
    for (int e = 0; e < kPerThread; ++e) {
        int flat_idx = tid * kPerThread + e;
        int m        = flat_idx / kTileN_;
        int n_local  = flat_idx % kTileN_;
        if (m >= batch_size || !row_active(g, m)) {
            local_nei[e] = 0.f;
        } else {
            int n_partner = n_local ^ 1;
            local_nei[e] =
                (n_partner < kTileN_) ? float(sC_raw(m, n_partner)) : 0.f;
        }
    }

    #pragma unroll
    for (int e = 0; e < kPerThread; ++e) {
        int flat_idx = tid * kPerThread + e;
        int m        = flat_idx / kTileN_;
        int n_local  = flat_idx % kTileN_;
        int n_global = tile_n_global + n_local;

        if (m >= batch_size) continue;
        if (!row_active(g, m)) continue;

        int32_t pos_m = row_position(g, m);

        float val          = local_c[e];
        int   head_dim_idx = n_global % HeadDim;

        if (n_global < Q_End) {
            long rope_off = (long)pos_m * HeadDim + head_dim_idx;
            float c = __ldg(rope_cos_ldg + rope_off);
            float s = __ldg(rope_sin_ldg + rope_off);
            int   n_partner = n_local ^ 1;
            float neighbor =
                (n_partner < kTileN_) ? local_nei[e] : 0.0f;
            float sign = (n_global % 2 == 0) ? -1.0f : 1.0f;
            t_q_out(m, n_global) = T(val * c + sign * neighbor * s);

        } else if (n_global < K_End) {
            long rope_off = (long)pos_m * HeadDim + head_dim_idx;
            float c = __ldg(rope_cos_ldg + rope_off);
            float s = __ldg(rope_sin_ldg + rope_off);
            int   n_partner = n_local ^ 1;
            float neighbor =
                (n_partner < kTileN_) ? local_nei[e] : 0.0f;
            float sign = (n_global % 2 == 0) ? -1.0f : 1.0f;
            int   kh   = (n_global - Q_End) / HeadDim;
            t_k_cache(m, pos_m, kh, head_dim_idx) = T(val * c + sign * neighbor * s);

        } else if (n_global < V_End) {
            int vh = (n_global - K_End) / HeadDim;
            t_v_cache(m, pos_m, vh, head_dim_idx) = T(val);
        }
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_EPILOGUE_ROPE_CYCLES(g, sm_id, inst_idx, prof_clock_now() - prof_stage_begin);
        prof_stage_begin = prof_clock_now();
    }
#endif
    __syncthreads();
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_SYNC, prof_clock_now() - prof_stage_begin);
    }
#endif

#if DEBUG_PRINT
    if (tid == 0 && blockIdx.x == 0) {
        constexpr int Q_EndDbg = LlamaConfig::HIDDEN_DIM;
        printf("[rope tile_n=%d] q_out row0[n=%d..%d] = %.5f %.5f %.5f %.5f\n",
               tile_n_global, tile_n_global, tile_n_global + 3,
               (tile_n_global + 0 < Q_EndDbg)
                   ? float(t_q_out(0, tile_n_global + 0)) : 0.f,
               (tile_n_global + 1 < Q_EndDbg)
                   ? float(t_q_out(0, tile_n_global + 1)) : 0.f,
               (tile_n_global + 2 < Q_EndDbg)
                   ? float(t_q_out(0, tile_n_global + 2)) : 0.f,
               (tile_n_global + 3 < Q_EndDbg)
                   ? float(t_q_out(0, tile_n_global + 3)) : 0.f);
    }
#endif
}

// =============================================================================
// qkv_gemm_rope (device function, dispatched from mk_kernel)
// =============================================================================

__device__ __forceinline__ void qkv_gemm_rope(
    LlamaGlobals& g,
    SmemLayout&   smem,
    int           inst_idx
) {
    // ── Instruction Parsing ────────────────────────────────────────────────
    // -----------------------------------------------------------------------
    // Parse instruction
    // -----------------------------------------------------------------------
    const int32_t* inst = smem.instruction;
    int layer_idx       = inst[1];
    int batch_size      = inst[2];
    // inst[3] = tile_M_idx (always 0)
    int start_block_idx = inst[4];
    int end_block_idx   = inst[5];

    int      tid   = threadIdx.x;
    int      sm_id = blockIdx.x;
    uint8_t* smem_base   = smem.main_workspace;

    // ── Barrier Waiting ────────────────────────────────────────────────────
    // -----------------------------------------------------------------------
    // Barrier wait (layer > 0: wait for DownProj of previous layer)
    // -----------------------------------------------------------------------
    auto dependency_barriers = g.get_barrier();
    volatile int32_t* previous_layer_ready = layer_idx > 0
        ? &dependency_barriers(layer_idx - 1, OPCODE_DOWN_PROJ_RESIDUAL - 1, 0)
        : nullptr;
    if (layer_idx > 0) {
        int expected = LlamaConfig::HIDDEN_DIM / KernelConfig::kTileN;
        if (tid == 0) {
            barrier_wait(previous_layer_ready, expected, sm_id,
                                  OPCODE_QKV_GEMM_ROPE, 0, layer_idx, &g, inst_idx);
        } else {
            const bfloat16_t* qw_pref = g.qkv_weight_ptr
                + (long)layer_idx * LlamaConfig::QKV_OUT_DIM * LlamaConfig::HIDDEN_DIM;
            const long long nbytes =
                (long long)LlamaConfig::QKV_OUT_DIM * (long long)LlamaConfig::HIDDEN_DIM
                * (long long)sizeof(bfloat16_t);
            long long nvec_ll = nbytes / 16;
            if (nvec_ll > 4096) {
                nvec_ll = 4096;
            }
            const int n_vec = (nvec_ll > 0) ? static_cast<int>(nvec_ll) : 0;
            for (int j = tid; j < n_vec; j += KernelConfig::NUM_THREADS) {
                (void)__ldg(reinterpret_cast<const uint4*>(qw_pref) + j);
            }
        }
    }
    __syncthreads();

    // Wait for standalone attention RMS (barrier row 7 / opcode 8): one publish per batch row.
    // Lane 0 spins on the barrier; other lanes use the idle window to prefetch the current-layer
    // QKV weight tile through L2 (mirrors the DownProj prefetch above). This also warms layer 0,
    // which the DownProj-gated prefetch skips (it only runs when layer_idx > 0).
    volatile int32_t* rms_ready =
        &dependency_barriers(layer_idx, OPCODE_ATTN_RMS - 1, 0);
    if (tid == 0) {
        int expected_rms = batch_size;
        barrier_wait(
            rms_ready,
            expected_rms,
            sm_id,
            OPCODE_QKV_GEMM_ROPE,
            0,
            layer_idx,
            &g,
            inst_idx);
    } else {
        const bfloat16_t* qw_pref = g.qkv_weight_ptr
            + (long)layer_idx * LlamaConfig::QKV_OUT_DIM * LlamaConfig::HIDDEN_DIM;
        const long long nbytes =
            (long long)LlamaConfig::QKV_OUT_DIM * (long long)LlamaConfig::HIDDEN_DIM
            * (long long)sizeof(bfloat16_t);
        long long nvec_ll = nbytes / 16;
        if (nvec_ll > 4096) {
            nvec_ll = 4096;
        }
        const int n_vec = (nvec_ll > 0) ? static_cast<int>(nvec_ll) : 0;
        for (int j = tid; j < n_vec; j += KernelConfig::NUM_THREADS) {
            (void)__ldg(reinterpret_cast<const uint4*>(qw_pref) + j);
        }
    }
    __syncthreads();

    // ── Pre-Operation Stage (Case 1: produce resident A) ───────────────────
    // Load normalized activations from attn_rms_out (written by decode_attn_rms).
    // -----------------------------------------------------------------------
#ifdef ENABLE_PROFILING
    uint64_t prof_stage_begin = 0;
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    if constexpr (!kQkvStagedA) {
        load_batched_to_smem_a<
            KernelConfig::NUM_THREADS,
            LlamaConfig::HIDDEN_DIM,
            GemmConfig1Tile::SmemLayoutA   // same layout for both 1-tile and 2-tile
        >(
            g.attn_rms_out_ptr,
            smem_ptr_at<bfloat16_t>(smem_base, QkvGemmSmemLayout::kOffsetResidentA),
            batch_size
        );
        cp_async_fence();
        cp_async_wait<0>();
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_STAGING, prof_clock_now() - prof_stage_begin);
        prof_stage_begin = prof_clock_now();
    }
#endif
    __syncthreads();
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_SYNC, prof_clock_now() - prof_stage_begin);
    }
#endif

#if DEBUG_PRINT
    if constexpr (!kQkvStagedA) {
    if (tid == 0 && sm_id == 0) {
        auto sA_dbg = make_tensor(
            make_smem_ptr(smem_ptr_at<bfloat16_t>(smem_base, QkvGemmSmemLayout::kOffsetResidentA)),
            GemmConfig1Tile::SmemLayoutA{});
        printf("[rms layer=%d sm=%d] smem_A row0[0..3] = %.5f %.5f %.5f %.5f\n",
               layer_idx, sm_id,
               float(sA_dbg(0,0)), float(sA_dbg(0,1)),
               float(sA_dbg(0,2)), float(sA_dbg(0,3)));
        printf("[rms layer=%d sm=%d] smem_A row1[0..3] = %.5f %.5f %.5f %.5f\n",
               layer_idx, sm_id,
               float(sA_dbg(1,0)), float(sA_dbg(1,1)),
               float(sA_dbg(1,2)), float(sA_dbg(1,3)));
    }
    }
#endif

    // ── GEMM (compute_gemm_tile: prologue / main loop / epilogue) ──────────
    // -----------------------------------------------------------------------
    // N-tile GEMM dispatch
    // QKV weight layout: [QKV_OUT_DIM, HIDDEN_DIM] for this layer.
    // -----------------------------------------------------------------------
    const bfloat16_t* qkv_w = g.qkv_weight_ptr
        + (long)layer_idx * LlamaConfig::QKV_OUT_DIM * LlamaConfig::HIDDEN_DIM;

    auto t_barriers = g.get_barrier();
    constexpr int TilesPerHead = LlamaConfig::HEAD_DIM / KernelConfig::kTileN;

    // Process adjacent tiles in 4/2/1 groups without changing the generic
    // staging/epilogue structure.
    int i = start_block_idx;
    for (; i < end_block_idx; ) {
        bool use_eight_tiles =
            KernelConfig::NUM_WARPS == 8 && (i + 7) < end_block_idx;
        bool use_four_tiles = !use_eight_tiles && (i + 3) < end_block_idx;
        bool use_two_tiles = !use_eight_tiles && !use_four_tiles && (i + 1) < end_block_idx;
        int n_blocks = use_eight_tiles ? 8 : (use_four_tiles ? 4 : (use_two_tiles ? 2 : 1));
        const bfloat16_t* gmem_B =
            qkv_w + (long)i * KernelConfig::kTileN * LlamaConfig::HIDDEN_DIM;

        if (use_eight_tiles) {
            compute_gemm_tile<WideGemmConfig>(
                tid, batch_size, smem_base,
                g.attn_rms_out_ptr,
                gmem_B,
                i * KernelConfig::kTileN,
                layer_idx, g, sm_id, inst_idx
            );
        } else if (use_four_tiles) {
            compute_gemm_tile<GemmConfig4Tile>(
                tid, batch_size, smem_base,
                g.attn_rms_out_ptr,
                gmem_B,
                i * KernelConfig::kTileN,
                layer_idx, g, sm_id, inst_idx
            );
        } else if (use_two_tiles) {
            compute_gemm_tile<GemmConfig2Tile>(
                tid, batch_size, smem_base,
                g.attn_rms_out_ptr,
                gmem_B,
                i * KernelConfig::kTileN,
                layer_idx, g, sm_id, inst_idx
            );
        } else {
            compute_gemm_tile<GemmConfig1Tile>(
                tid, batch_size, smem_base,
                g.attn_rms_out_ptr,
                gmem_B,
                i * KernelConfig::kTileN,
                layer_idx, g, sm_id, inst_idx
            );
        }

        // Update barrier: one increment per logical tile, coalesced by head when
        // a grouped dispatch stays within the same head bucket.
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            prof_stage_begin = prof_clock_now();
        }
#endif
        __syncthreads();
        if (tid == 0) {
            int tile_offset = 0;
            while (tile_offset < n_blocks) {
                int head_idx = (i + tile_offset) / TilesPerHead;
                int tiles_in_head = 1;
                while ((tile_offset + tiles_in_head) < n_blocks &&
                       ((i + tile_offset + tiles_in_head) / TilesPerHead) == head_idx) {
                    ++tiles_in_head;
                }
                barrier_publish(
                    &t_barriers(layer_idx, OPCODE_QKV_GEMM_ROPE - 1, head_idx),
                    tiles_in_head);
                tile_offset += tiles_in_head;
            }
        }
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_PUBLISH, prof_clock_now() - prof_stage_begin);
        }
#endif
        i += n_blocks;
    }

    // ── Barrier Update ─────────────────────────────────────────────────────
    // (Per-tile atomics above publish head progress for downstream ops.)
}

}  // namespace megakernel
