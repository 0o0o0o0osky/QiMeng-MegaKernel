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

template <int TileNBlocks>
struct DownProjGemmConfig : GemmSharedBase {
    static constexpr int kTileN_ = KernelConfig::kTileN * TileNBlocks;
    static constexpr int kMmaEURepeatN = TileNBlocks;

    using mma_op = SM80_16x8x16_F32BF16BF16F32_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    using mma_atom_shape = mma_traits::Shape_MNK;

    static constexpr int N_REG_REPEAT = 2;
    static constexpr int kMmaPM = 1 * kMmaEURepeatM * get<0>(mma_atom_shape{});
    static constexpr int kMmaPN = N_REG_REPEAT * kMmaEURepeatN * get<1>(mma_atom_shape{});
    static constexpr int kMmaPK = 1 * kMmaEURepeatK * get<2>(mma_atom_shape{});

    using MMA_EU_RepeatT = decltype(make_layout(make_shape(
        Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})));
    using MMA_P_T = Tile<Int<kMmaPM>, Int<kMmaPN>, Int<kMmaPK>>;
    using MMA = decltype(make_tiled_mma(mma_atom{}, MMA_EU_RepeatT{}, MMA_P_T{}));

    using SmemLayoutA = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<KernelConfig::kTileM>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using SmemLayoutB = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(Int<kTileN_>{}, Int<kTileK>{}, Int<kStage>{})
    ));
    using S2RCopyAtomA = s2r_copy_atom;
    using S2RCopyAtomB = s2r_copy_atom;
    using G2SCopyA = G2SCopyB;
};

using DownProjGemmConfig1Tile = DownProjGemmConfig<1>;
using DownProjGemmConfig2Tile = DownProjGemmConfig<2>;
using DownProjGemmConfig4Tile = DownProjGemmConfig<4>;
using DownProjGemmConfig8Tile = DownProjGemmConfig<8>;

template <typename GemmCfg>
__device__ __forceinline__ void load_downproj_a_stage(
    int tid,
    int batch_size,
    const bfloat16_t* __restrict__ gmem_a_ptr,
    bfloat16_t* smem_a_ptr,
    int k_tile_idx,
    int stage_idx)
{
    constexpr int kTileK = GemmCfg::kTileK;
    auto sA = make_tensor(make_smem_ptr(smem_a_ptr), typename GemmCfg::SmemLayoutA{});
    if (tid < 128) {
        int vec_idx = tid;
        int m = vec_idx / 8;
        int k_vec = vec_idx % 8;
        int k0 = k_tile_idx * kTileK + k_vec * 8;
        if (m < batch_size) {
            cp_async_cg<16>(&sA(m, k_vec * 8, stage_idx), gmem_a_ptr + (long)m * LlamaConfig::INTERM_DIM + k0);
        } else {
            CUTE_UNROLL
            for (int kk = 0; kk < 8; ++kk) {
                sA(m, k_vec * 8 + kk, stage_idx) = bfloat16_t(0.f);
            }
        }
    }
}

template <typename GemmCfg>
__device__ __forceinline__ void compute_downproj_tile_to_float_smem(
    int tid,
    int batch_size,
    uint8_t* __restrict__ smem_base,
    const bfloat16_t* __restrict__ gmem_a_ptr,
    const bfloat16_t* __restrict__ gmem_b_ptr,
    float* out_smem,
    LlamaGlobals& g,
    int sm_id,
    int inst_idx)
{
    using T = typename GemmCfg::T;
    using TiledMMA = typename GemmCfg::MMA;
    using SmemLA = typename GemmCfg::SmemLayoutA;
    using SmemLB = typename GemmCfg::SmemLayoutB;
    using S2RA = typename GemmCfg::S2RCopyAtomA;
    using S2RB = typename GemmCfg::S2RCopyAtomB;
    using G2SCB = typename GemmCfg::G2SCopyB;

    constexpr int kTileM_ = KernelConfig::kTileM;
    constexpr int kTileN_ = GemmCfg::kTileN_;
    constexpr int kTileK = GemmCfg::kTileK;
    constexpr int kStage = GemmCfg::kStage;
    constexpr int kInputDim = LlamaConfig::INTERM_DIM;
    constexpr int ntile = kInputDim / kTileK;

    T* Ashm = smem_ptr_at<T>(smem_base, DownprojGemmSmemLayout::kOffsetStagedA);
    T* Bshm = smem_ptr_at<T>(smem_base, DownprojGemmSmemLayout::kOffsetStagedB);

    auto sA = make_tensor(make_smem_ptr(Ashm), SmemLA{});
    auto sA_tiles = local_tile(
        sA,
        make_tile(Int<kTileM_>{}, Int<kTileK>{}, Int<1>{}),
        make_coord(0, 0, _),
        Step<_1, _1, X>{});
    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLB{});
    auto sOut = make_tensor(
        make_smem_ptr(out_smem),
        make_shape(Int<kTileM_>{}, Int<kTileN_>{}),
        LayoutRight{});

    auto gB = make_tensor(
        make_gmem_ptr(gmem_b_ptr),
        make_shape(Int<kTileN_>{}, Int<kInputDim>{}),
        make_stride(Int<kInputDim>{}, Int<1>{}));
    auto gB_tiles = local_tile(
        gB,
        make_tile(Int<kTileN_>{}, Int<kTileK>{}),
        make_coord(0, _));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(sA_tiles(_, _, 0));
    auto tCrB = thr_mma.partition_fragment_B(sB(_, _, 0));
    auto tCrD = thr_mma.partition_fragment_C(sOut);
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
        load_downproj_a_stage<GemmCfg>(
            tid, batch_size, gmem_a_ptr, Ashm, pipeline.next_tile, pipeline.write_stage);
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
            if (tid < mma_size) {
                cute::copy(s2r_copy_a, tAsA(_, _, ik, pipeline.read_stage), tCrA_view(_, _, ik));
                cute::copy(s2r_copy_b, tBsB(_, _, ik, pipeline.read_stage), tCrB_view(_, _, ik));
            }

            if (ik == 0) {
                if (pipeline.has_next(ntile)) {
                    load_downproj_a_stage<GemmCfg>(
                        tid,
                        batch_size,
                        gmem_a_ptr,
                        Ashm,
                        pipeline.next_tile,
                        pipeline.write_stage);
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
            auto cOut = make_identity_tensor(shape(sOut));
            auto tCcOut = thr_mma.partition_C(cOut);
            auto tCsOut = thr_mma.partition_C(sOut);
            CUTE_UNROLL
            for (int n = 0; n < (int)size(tCrD); ++n) {
                int mm = get<0>(tCcOut(n));
                int nn = get<1>(tCcOut(n));
                tCsOut(n) = (mm < batch_size && nn < kTileN_) ? float(tCrD(n)) : 0.f;
            }
        }
    }
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_EPILOGUE, prof_clock_now() - prof_stage_begin);
    }
#endif
    __syncthreads();
}

__device__ __forceinline__ void decode_downproj_gemm_residual(
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

    int tid = threadIdx.x;
    int sm_id = blockIdx.x;
    uint8_t* smem_base = smem.main_workspace;

    if (tile_m_idx != 0) {
#if DEBUG_PRINT
        if (tid == 0) {
            printf("downproj unsupported tile_m_idx=%d\n", tile_m_idx);
        }
#endif
        __trap();
    }
    if (start_block_idx >= end_block_idx) {
        return;
    }

    // Barrier Wait
    auto barriers = g.get_barrier();
    volatile int32_t* input_ready =
        &barriers(layer_idx, OPCODE_UPGATE_SILU - 1, 0);
    if (tid == 0) {
        barrier_wait(
            input_ready,
            LlamaConfig::INTERM_DIM / KernelConfig::kTileN,
            sm_id,
            OPCODE_DOWN_PROJ_RESIDUAL,
            0,
            layer_idx,
            &g,
            inst_idx);
    }
    __syncthreads();

    const bfloat16_t* silu_ptr = g.silu_out_ptr;
    const bfloat16_t* down_w = g.down_weight_ptr
        + (long)layer_idx * LlamaConfig::HIDDEN_DIM * LlamaConfig::INTERM_DIM;
    auto hidden = g.get_hidden_state();
    // GEMM Tile Dispatch
    int i = start_block_idx;
    while (i < end_block_idx) {
        bool use_eight_tiles =
            KernelConfig::NUM_WARPS == 8 && (i + 7 < end_block_idx);
        bool use_four_tiles = !use_eight_tiles &&
            (i + 3 < end_block_idx) &&
            (end_block_idx - i >= 4);
        bool use_two_tiles = !use_eight_tiles && !use_four_tiles && (i + 1 < end_block_idx);
        int n_blocks = use_eight_tiles ? 8 : (use_four_tiles ? 4 : (use_two_tiles ? 2 : 1));
        int tile_n_global = i * KernelConfig::kTileN;
        float* out_tile = smem_ptr_at<float>(smem_base, DownprojGemmSmemLayout::kOffsetOutputScratch);
        const bfloat16_t* down_ptr = down_w + (long)tile_n_global * LlamaConfig::INTERM_DIM;

        if (use_eight_tiles) {
            compute_downproj_tile_to_float_smem<DownProjGemmConfig8Tile>(
                tid, batch_size, smem_base, silu_ptr, down_ptr, out_tile, g, sm_id, inst_idx);
        } else if (use_four_tiles) {
            compute_downproj_tile_to_float_smem<DownProjGemmConfig4Tile>(
                tid, batch_size, smem_base, silu_ptr, down_ptr, out_tile, g, sm_id, inst_idx);
        } else if (use_two_tiles) {
            compute_downproj_tile_to_float_smem<DownProjGemmConfig2Tile>(
                tid, batch_size, smem_base, silu_ptr, down_ptr, out_tile, g, sm_id, inst_idx);
        } else {
            compute_downproj_tile_to_float_smem<DownProjGemmConfig1Tile>(
                tid, batch_size, smem_base, silu_ptr, down_ptr, out_tile, g, sm_id, inst_idx);
        }

#ifdef ENABLE_PROFILING
        uint64_t prof_stage_begin = 0;
        if (tid == 0) {
            prof_stage_begin = prof_clock_now();
        }
#endif
        if (use_eight_tiles) {
            vec_epilogue_residual_add_float_smem<KernelConfig::kTileM, DownProjGemmConfig8Tile::kTileN_, KernelConfig::NUM_THREADS>(
                out_tile, g.hidden_state_ptr, LlamaConfig::HIDDEN_DIM,
                tile_n_global, batch_size, LlamaConfig::HIDDEN_DIM, g, tid);
        } else if (use_four_tiles) {
            vec_epilogue_residual_add_float_smem<KernelConfig::kTileM, DownProjGemmConfig4Tile::kTileN_, KernelConfig::NUM_THREADS>(
                out_tile, g.hidden_state_ptr, LlamaConfig::HIDDEN_DIM,
                tile_n_global, batch_size, LlamaConfig::HIDDEN_DIM, g, tid);
        } else if (use_two_tiles) {
            vec_epilogue_residual_add_float_smem<KernelConfig::kTileM, DownProjGemmConfig2Tile::kTileN_, KernelConfig::NUM_THREADS>(
                out_tile, g.hidden_state_ptr, LlamaConfig::HIDDEN_DIM,
                tile_n_global, batch_size, LlamaConfig::HIDDEN_DIM, g, tid);
        } else {
            vec_epilogue_residual_add_float_smem<KernelConfig::kTileM, DownProjGemmConfig1Tile::kTileN_, KernelConfig::NUM_THREADS>(
                out_tile, g.hidden_state_ptr, LlamaConfig::HIDDEN_DIM,
                tile_n_global, batch_size, LlamaConfig::HIDDEN_DIM, g, tid);
        }
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_EPILOGUE, prof_clock_now() - prof_stage_begin);
            prof_stage_begin = prof_clock_now();
        }
#endif

        // Barrier Publication
        __syncthreads();
        if (tid == 0) {
            barrier_publish(
                &barriers(layer_idx, OPCODE_DOWN_PROJ_RESIDUAL - 1, 0),
                n_blocks);
        }
#ifdef ENABLE_PROFILING
        if (tid == 0) {
            PROF_ADD_STAGE_CYCLES(g, sm_id, inst_idx, PROF_FIELD_PUBLISH, prof_clock_now() - prof_stage_begin);
        }
#endif
        i += n_blocks;
    }
}

}  // namespace megakernel
