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

template <typename GemmCfg>
__device__ __forceinline__ void compute_lmhead_tile_to_float_smem(
    int tid,
    int batch_size,
    uint8_t* __restrict__ smem_base,
    const bfloat16_t* __restrict__ gmem_a_ptr,
    const bfloat16_t* __restrict__ gmem_b_ptr,
    float* out_smem,
    int tile_n_global,
    LlamaGlobals& g,
    int sm_id,
    int inst_idx)
{
    using T = typename GemmCfg::T;
    using TiledMMA = typename GemmCfg::MMA;
    using SmemLB = typename GemmCfg::SmemLayoutB;
    using SmemLOut = SharedOutputScratchLayoutT<KernelConfig::kTileM, GemmCfg::kTileN_>;
    using S2RA = typename GemmCfg::S2RCopyAtomA;
    using S2RB = typename GemmCfg::S2RCopyAtomB;
    using G2SCB = typename GemmCfg::G2SCopyB;

    constexpr int kTileN_ = GemmCfg::kTileN_;
    constexpr int kTileK = GemmCfg::kTileK;
    constexpr int kStage = GemmCfg::kStage;
    constexpr int kHidden = LlamaConfig::HIDDEN_DIM;
    constexpr int ntile = kHidden / kTileK;
    constexpr int kATileCount = kLmheadStagedA ? kStage : ntile;
    using SmemLA = GemmSharedBase::SmemLayoutATileRing<kATileCount>;

    T* Ashm = smem_ptr_at<T>(smem_base, LmheadGemmSmemLayout::kOffsetA);
    T* Bshm = smem_ptr_at<T>(smem_base, LmheadGemmSmemLayout::kOffsetStagedB);

    auto sA_tiles = make_tensor(make_smem_ptr(Ashm), SmemLA{});

    auto sB = make_tensor(make_smem_ptr(Bshm), SmemLB{});
    auto sOut = make_tensor(make_smem_ptr(out_smem), SmemLOut{});

    auto gB = make_tensor(
        make_gmem_ptr(gmem_b_ptr),
        make_shape(Int<kTileN_>{}, Int<kHidden>{}),
        make_stride(Int<kHidden>{}, Int<1>{}));
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
        if constexpr (kLmheadStagedA) {
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
            const int a_read_tile = kLmheadStagedA ? pipeline.read_stage : itile;
            if (tid < mma_size) {
                cute::copy(s2r_copy_a, tAsA(_, _, ik, a_read_tile), tCrA_view(_, _, ik));
                cute::copy(s2r_copy_b, tBsB(_, _, ik, pipeline.read_stage), tCrB_view(_, _, ik));
            }

            if (ik == 0) {
                if (pipeline.has_next(ntile)) {
                    if constexpr (kLmheadStagedA) {
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
            auto cOut = make_identity_tensor(shape(sOut));
            auto tCcOut = thr_mma.partition_C(cOut);
            auto logits = g.get_logits();
            CUTE_UNROLL
            for (int n = 0; n < (int)size(tCrD); ++n) {
                int mm = get<0>(tCcOut(n));
                int nn = get<1>(tCcOut(n));
                int n_global = tile_n_global + nn;
                if (mm < batch_size && nn < kTileN_ && n_global < LlamaConfig::VOCAB_SIZE) {
                    logits(mm, n_global) = row_active(g, mm)
                        ? bfloat16_t(float(tCrD(n)))
                        : bfloat16_t(-1.0e9f);
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
    }
#endif
}

__device__ __forceinline__ void decode_lmhead_gemm(
    LlamaGlobals& g,
    SmemLayout& smem,
    int inst_idx)
{
    const int32_t* inst = smem.instruction;
    int batch_size = inst[1];
    int start_block_idx = inst[2];
    int end_block_idx = inst[3];

    int tid = threadIdx.x;
    int sm_id = blockIdx.x;
    uint8_t* smem_base = smem.main_workspace;

#ifdef ENABLE_PROFILING
    uint64_t prof_stage_begin = 0;
#endif

    if (start_block_idx >= end_block_idx) {
        return;
    }

    auto barriers = g.get_barrier();
    volatile int32_t* input_ready = &barriers(
        LlamaConfig::NUM_LAYERS - 1, OPCODE_LMHEAD_RMS - 1, 0);
    if (tid == 0) {
        barrier_wait(
            input_ready,
            batch_size,
            sm_id,
            OPCODE_LMHEAD_GEMM,
            0,
            LlamaConfig::NUM_LAYERS - 1,
            &g,
            inst_idx);
    }
    __syncthreads();

#ifdef ENABLE_PROFILING
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    if constexpr (!kLmheadStagedA) {
        load_batched_to_smem_a<
            KernelConfig::NUM_THREADS,
            LlamaConfig::HIDDEN_DIM,
            GemmConfig1Tile::SmemLayoutA
        >(
            g.lm_head_rms_out_ptr,
            smem_ptr_at<bfloat16_t>(smem_base, LmheadGemmSmemLayout::kOffsetResidentA),
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

    const bfloat16_t* lm_head_w = g.lm_head_weight_ptr;
    auto logits = g.get_logits();

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
        float* out_tile = smem_ptr_at<float>(smem_base, LmheadGemmSmemLayout::kOffsetOutputScratch);
        const bfloat16_t* lm_ptr = lm_head_w + (long)tile_n_global * LlamaConfig::HIDDEN_DIM;

        if (use_eight_tiles) {
            compute_lmhead_tile_to_float_smem<WideGemmConfig>(
                tid, batch_size, smem_base, g.lm_head_rms_out_ptr,
                lm_ptr, out_tile, tile_n_global, g, sm_id, inst_idx);
        } else if (use_four_tiles) {
            compute_lmhead_tile_to_float_smem<GemmConfig4Tile>(
                tid, batch_size, smem_base, g.lm_head_rms_out_ptr,
                lm_ptr, out_tile, tile_n_global, g, sm_id, inst_idx);
        } else if (use_two_tiles) {
            compute_lmhead_tile_to_float_smem<GemmConfig2Tile>(
                tid, batch_size, smem_base, g.lm_head_rms_out_ptr,
                lm_ptr, out_tile, tile_n_global, g, sm_id, inst_idx);
        } else {
            compute_lmhead_tile_to_float_smem<GemmConfig1Tile>(
                tid, batch_size, smem_base, g.lm_head_rms_out_ptr,
                lm_ptr, out_tile, tile_n_global, g, sm_id, inst_idx);
        }
        i += n_blocks;
    }
}

}  // namespace megakernel
