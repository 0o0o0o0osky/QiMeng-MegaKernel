#include "globals.cuh"
#include "llama_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"
#include "matmul_pipeline.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>
#include <cute/arch/mma_sm80.hpp>

namespace megakernel {

using namespace cute;

template <typename SmemLayoutA>
__device__ __forceinline__ void load_attn_out_to_smem_a_async(
    LlamaGlobals& g,
    int batch_size,
    uint8_t* smem_base)
{
    auto sA = make_tensor(
        make_smem_ptr(smem_ptr_at<bfloat16_t>(smem_base, OprojGemmSmemLayout::kOffsetResidentA)),
        SmemLayoutA{});

    constexpr int kHidden = LlamaConfig::HIDDEN_DIM;
    constexpr int kVecElems = 8;  // 16B bf16 vector per cp.async
    constexpr int kVecsPerRow = kHidden / kVecElems;
    constexpr int kTotalVecs = KernelConfig::kTileM * kVecsPerRow;
    const bfloat16_t* gA_ptr = g.attn_out_ptr;

    for (int vec_idx = threadIdx.x; vec_idx < kTotalVecs; vec_idx += KernelConfig::NUM_THREADS) {
        int m = vec_idx / kVecsPerRow;
        int k_vec = vec_idx % kVecsPerRow;
        int k0 = k_vec * kVecElems;
        if (m < batch_size) {
            cp_async_cg<16>(
                &sA(m, k0),
                gA_ptr + (long)m * kHidden + k0);
        } else {
            // kTileM may exceed batch_size; never feed uninitialized smem into the MMA.
            // cp_async_cg<0> does not clear the destination — zero this bf16 vector explicitly.
            CUTE_UNROLL
            for (int kk = 0; kk < kVecElems; ++kk) {
                sA(m, k0 + kk) = bfloat16_t(0.f);
            }
        }
    }
}

template <typename GemmCfg>
__device__ __forceinline__ void compute_oproj_tile(
    int tid,
    int batch_size,
    uint8_t* __restrict__ smem_base,
    const bfloat16_t* __restrict__ gmem_a_ptr,
    const bfloat16_t* __restrict__ gmem_b_ptr,
    int tile_n_global,
    LlamaGlobals& g,
    int sm_id,
    int inst_idx)
{
    using T = typename GemmCfg::T;
    using TiledMMA = typename GemmCfg::MMA;
    using SmemLB = typename GemmCfg::SmemLayoutB;
    using SmemLCRaw = typename GemmCfg::SmemLayoutCRaw;
    using S2RA = typename GemmCfg::S2RCopyAtomA;
    using S2RB = typename GemmCfg::S2RCopyAtomB;
    using G2SCB = typename GemmCfg::G2SCopyB;

    constexpr int kTileM_ = KernelConfig::kTileM;
    constexpr int kTileN_ = GemmCfg::kTileN_;
    constexpr int kTileK = GemmCfg::kTileK;
    constexpr int kStage = GemmCfg::kStage;
    constexpr int kHidden = LlamaConfig::HIDDEN_DIM;
    constexpr int ntile = kHidden / kTileK;
    constexpr int kATileCount = kOprojStagedA ? kStage : ntile;
    using SmemLA = GemmSharedBase::SmemLayoutATileRing<kATileCount>;

    T* Ashm = smem_ptr_at<T>(smem_base, OprojGemmSmemLayout::kOffsetA);
    T* Bshm = smem_ptr_at<T>(smem_base, OprojGemmSmemLayout::kOffsetStagedB);
    T* Cshm = smem_ptr_at<T>(smem_base, OprojGemmSmemLayout::kOffsetOutputScratch);

    auto sA_tiles = make_tensor(make_smem_ptr(Ashm), SmemLA{});

    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLB{});
    // MMA partition_C must target Raw row-major C (same as qkv_gemm_rope); linear epilogue
    // uses m*kTileN_+n on the same buffer — Swizzled SmemLayoutC would permute addresses.
    auto sC_raw = make_tensor(make_smem_ptr(Cshm), SmemLCRaw{});

    auto gB = make_tensor(
        make_gmem_ptr(gmem_b_ptr),
        make_shape(Int<kTileN_>{}, Int<kHidden>{}),
        make_stride(Int<kHidden>{}, Int<1>{}));
    auto gB_tiles = local_tile(
        gB,
        make_tile(Int<kTileN_>{}, Int<kTileK>{}),
        make_coord(0, _));

    auto gC_dummy = make_tensor(
        make_smem_ptr(Cshm),
        make_shape(Int<kTileM_>{}, Int<kTileN_>{}),
        LayoutRight{});

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(sA_tiles(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(gC_dummy);
    clear(tCrD);

    auto s2r_copy_a = make_tiled_copy_A(S2RA{}, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_slice(tid);
    auto tAsA = s2r_thr_a.partition_S(sA_tiles);
    auto tCrA_view = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b = make_tiled_copy_B(S2RB{}, tiled_mma);
    auto s2r_thr_b = s2r_copy_b.get_slice(tid);
    auto tBsB = s2r_thr_b.partition_S(sB);
    auto tCrB_view = s2r_thr_b.retile_D(tCrB);

    G2SCB g2s_copy_b;
    auto g2s_thr_b = g2s_copy_b.get_slice(tid);
    auto tBgB_copy = g2s_thr_b.partition_S(gB_tiles);
    auto tBsB_copy = g2s_thr_b.partition_D(sB);

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
        if constexpr (kOprojStagedA) {
            stage_activation_tile_async<kHidden>(
                gmem_a_ptr, Ashm, batch_size,
                pipeline.next_tile, pipeline.write_stage);
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

    #pragma unroll 1
    for (int itile = 0; itile < ntile; ++itile) {
        int nk = size<2>(tCrA);
        #pragma unroll
        for (int ik = 0; ik < nk; ++ik) {
#ifdef ENABLE_PROFILING
            if (tid == 0) {
                prof_stage_begin = prof_clock_now();
            }
#endif
            constexpr int mma_size = decltype(cute::size(TiledMMA{}))::value;
            const int a_read_tile = kOprojStagedA ? pipeline.read_stage : itile;
            if (tid < mma_size) {
                cute::copy(s2r_copy_a, tAsA(_, _, ik, a_read_tile), tCrA_view(_, _, ik));
                cute::copy(s2r_copy_b, tBsB(_, _, ik, pipeline.read_stage), tCrB_view(_, _, ik));
            }

            if (ik == 0) {
                if (pipeline.has_next(ntile)) {
                    if constexpr (kOprojStagedA) {
                        stage_activation_tile_async<kHidden>(
                            gmem_a_ptr, Ashm, batch_size,
                            pipeline.next_tile, pipeline.write_stage);
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

#ifdef ENABLE_PROFILING
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    {
        constexpr int mma_size = decltype(cute::size(TiledMMA{}))::value;
        if (tid < mma_size) {
            auto cC = make_identity_tensor(shape(sC_raw));
            auto tCcC = thr_mma.partition_C(cC);
            auto tCsC = thr_mma.partition_C(sC_raw);
            CUTE_UNROLL
            for (int n = 0; n < (int)size(tCrD); ++n) {
                int mm = get<0>(tCcC(n));
                int nn = get<1>(tCcC(n));
                if (mm < batch_size && nn < kTileN_) {
                    tCsC(n) = T(float(tCrD(n)));
                }
            }
        }
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_EPILOGUE, prof_clock_now() - prof_stage_begin);
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

    const T* c_tile_ptr = Cshm;

#if defined(DEBUG_OPROJ_TILE_N_GLOBAL)
    if (tid == 0 && tile_n_global == DEBUG_OPROJ_TILE_N_GLOBAL) {
        printf(
            "[DEBUG_OPROJ] sm_id=%d inst_idx=%d tile_n_global=%d batch_size=%d c00=%f\n",
            sm_id,
            inst_idx,
            tile_n_global,
            batch_size,
            float(c_tile_ptr[0]));
    }
#endif

    vec_epilogue_residual_add_bf16<kTileM_, kTileN_, KernelConfig::NUM_THREADS>(
        c_tile_ptr,
        g.hidden_state_ptr,
        LlamaConfig::HIDDEN_DIM,
        tile_n_global,
        batch_size,
        LlamaConfig::HIDDEN_DIM,
        g,
        tid);
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_EPILOGUE, prof_clock_now() - prof_stage_begin);
        prof_stage_begin = prof_clock_now();
    }
#endif
    __syncthreads();
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_SYNC, prof_clock_now() - prof_stage_begin);
    }
#endif
}

__device__ __forceinline__ void decode_oproj_gemm_residual(
    LlamaGlobals& g,
    SmemLayout& smem,
    int inst_idx)
{
    // Instruction Parse
    const int32_t* inst = smem.instruction;
    int layer_idx = inst[1];
    int batch_size = inst[2];
    int tile_m_idx = inst[3];
    int start_block_idx = inst[4];
    int end_block_idx = inst[5];
    int reduction_block_idx = inst[6];

    int tid = threadIdx.x;
    int sm_id = blockIdx.x;
    uint8_t* smem_base = smem.main_workspace;

    if (tile_m_idx != 0 || reduction_block_idx != 0) {
#if DEBUG_PRINT
        if (tid == 0) {
            printf("oproj unsupported tile_m_idx=%d reduction_block_idx=%d\n",
                   tile_m_idx, reduction_block_idx);
        }
#endif
        __trap();
    }

    // Barrier Wait
    auto barriers = g.get_barrier();
    if (tid == 0) {
        for (int b = 0; b < batch_size; ++b) {
            volatile int32_t* bar = &barriers(layer_idx, OPCODE_ATTENTION_REDUCTION - 1, b);
            barrier_wait(
                bar,
                LlamaConfig::NUM_Q_HEADS,
                sm_id,
                OPCODE_O_PROJ_RESIDUAL,
                b,
                layer_idx,
                &g,
                inst_idx);
        }
    }
    __syncthreads();

    // Resident-A Prologue
#ifdef ENABLE_PROFILING
    uint64_t prof_stage_begin = 0;
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    if constexpr (!kOprojStagedA) {
        load_attn_out_to_smem_a_async<GemmConfig1Tile::SmemLayoutA>(
            g, batch_size, smem_base);
        cp_async_fence();
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_STAGING, prof_clock_now() - prof_stage_begin);
        prof_stage_begin = prof_clock_now();
    }
#endif
    if constexpr (!kOprojStagedA) {
        cp_async_wait<0>();
    }
    __syncthreads();
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_SYNC, prof_clock_now() - prof_stage_begin);
    }
#endif

    const bfloat16_t* oproj_w = g.o_weight_ptr
        + (long)layer_idx * LlamaConfig::HIDDEN_DIM * LlamaConfig::HIDDEN_DIM;
    // GEMM Tile Dispatch
    int i = start_block_idx;
    if constexpr (KernelConfig::NUM_WARPS == 8) {
        for (; i + 7 < end_block_idx; i += 8) {
            const bfloat16_t* gmem_b =
                oproj_w + (long)i * KernelConfig::kTileN * LlamaConfig::HIDDEN_DIM;
            compute_oproj_tile<WideGemmConfig>(
                tid, batch_size, smem_base, g.attn_out_ptr, gmem_b,
                i * KernelConfig::kTileN, g, sm_id, inst_idx);
            __syncthreads();
            if (tid == 0) {
                barrier_publish(
                    &barriers(layer_idx, OPCODE_O_PROJ_RESIDUAL - 1, 0), 8);
            }
        }
    }
    for (; i + 3 < end_block_idx; i += 4) {
        const bfloat16_t* gmem_b =
            oproj_w + (long)i * KernelConfig::kTileN * LlamaConfig::HIDDEN_DIM;
        compute_oproj_tile<GemmConfig4Tile>(
            tid, batch_size, smem_base, g.attn_out_ptr, gmem_b,
            i * KernelConfig::kTileN, g, sm_id, inst_idx);
        __syncthreads();
        if (tid == 0) {
            barrier_publish(
                &barriers(layer_idx, OPCODE_O_PROJ_RESIDUAL - 1, 0), 4);
        }
    }
    for (; i + 1 < end_block_idx; i += 2) {
        const bfloat16_t* gmem_b =
            oproj_w + (long)i * KernelConfig::kTileN * LlamaConfig::HIDDEN_DIM;
        compute_oproj_tile<GemmConfig2Tile>(
            tid, batch_size, smem_base, g.attn_out_ptr, gmem_b,
            i * KernelConfig::kTileN, g, sm_id, inst_idx);

#ifdef ENABLE_PROFILING
        if (tid == 0) {
            prof_stage_begin = prof_clock_now();
        }
#endif
        __syncthreads();
        if (tid == 0) {
            barrier_publish(&barriers(layer_idx, OPCODE_O_PROJ_RESIDUAL - 1, 0), 2);
        }
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_PUBLISH, prof_clock_now() - prof_stage_begin);
        }
#endif
    }

    // Barrier Publication
    if (i < end_block_idx) {
        const bfloat16_t* gmem_b =
            oproj_w + (long)i * KernelConfig::kTileN * LlamaConfig::HIDDEN_DIM;
        compute_oproj_tile<GemmConfig1Tile>(
            tid, batch_size, smem_base, g.attn_out_ptr, gmem_b,
            i * KernelConfig::kTileN, g, sm_id, inst_idx);

#ifdef ENABLE_PROFILING
        if (tid == 0) {
            prof_stage_begin = prof_clock_now();
        }
#endif
        __syncthreads();
        if (tid == 0) {
            barrier_publish(&barriers(layer_idx, OPCODE_O_PROJ_RESIDUAL - 1, 0), 1);
        }
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_PUBLISH, prof_clock_now() - prof_stage_begin);
        }
#endif
    }
}

}  // namespace megakernel
