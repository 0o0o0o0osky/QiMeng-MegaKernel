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
__device__ __forceinline__ void compute_upgate_tile_pair_to_float_smem(
    int tid,
    int batch_size,
    uint8_t* __restrict__ smem_base,
    const bfloat16_t* __restrict__ gmem_a_ptr,
    const bfloat16_t* __restrict__ gmem_up_ptr,
    const bfloat16_t* __restrict__ gmem_gate_ptr,
    float* up_smem,
    float* gate_smem,
    LlamaGlobals& g,
    int sm_id,
    int inst_idx)
{
    using T = typename GemmCfg::T;
    using TiledMMA = typename GemmCfg::MMA;
    using SmemLB = typename GemmCfg::SmemLayoutB;
    using S2RA = typename GemmCfg::S2RCopyAtomA;
    using S2RB = typename GemmCfg::S2RCopyAtomB;
    using G2SCB = typename GemmCfg::G2SCopyB;

    constexpr int kTileM_ = KernelConfig::kTileM;
    constexpr int kTileN_ = GemmCfg::kTileN_;
    constexpr int kTileK = GemmCfg::kTileK;
    constexpr int kStage = GemmCfg::kStage;
    constexpr int kHidden = LlamaConfig::HIDDEN_DIM;
    constexpr int ntile = kHidden / kTileK;
    constexpr int kATileCount = kUpgateStagedA ? kStage : ntile;
    using SmemLA = GemmSharedBase::SmemLayoutATileRing<kATileCount>;

    T* Ashm = smem_ptr_at<T>(smem_base, UpgateGemmSmemLayout::kOffsetA);
    T* BUpShm = smem_ptr_at<T>(smem_base, UpgateGemmSmemLayout::kOffsetStagedBUp);
    T* BGateShm = smem_ptr_at<T>(smem_base, UpgateGemmSmemLayout::kOffsetStagedBGate);

    auto sA_tiles = make_tensor(make_smem_ptr(Ashm), SmemLA{});

    auto sBUp = make_tensor(make_smem_ptr(BUpShm), SmemLB{});
    auto sBGate = make_tensor(make_smem_ptr(BGateShm), SmemLB{});
    auto sUpOut = make_tensor(
        make_smem_ptr(up_smem),
        SharedOutputScratchLayoutT<kTileM_, kTileN_>{});
    auto sGateOut = make_tensor(
        make_smem_ptr(gate_smem),
        SharedOutputScratchLayoutT<kTileM_, kTileN_>{});

    auto gBUp = make_tensor(
        make_gmem_ptr(gmem_up_ptr),
        make_shape(Int<kTileN_>{}, Int<kHidden>{}),
        make_stride(Int<kHidden>{}, Int<1>{}));
    auto gBUpTiles = local_tile(
        gBUp,
        make_tile(Int<kTileN_>{}, Int<kTileK>{}),
        make_coord(0, _));
    auto gBGate = make_tensor(
        make_gmem_ptr(gmem_gate_ptr),
        make_shape(Int<kTileN_>{}, Int<kHidden>{}),
        make_stride(Int<kHidden>{}, Int<1>{}));
    auto gBGateTiles = local_tile(
        gBGate,
        make_tile(Int<kTileN_>{}, Int<kTileK>{}),
        make_coord(0, _));

    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_slice(tid);

    auto tCrA = thr_mma.partition_fragment_A(sA_tiles(_, _, 0));
    auto tCrBUp = thr_mma.partition_fragment_B(sBUp(_, _, 0));
    auto tCrBGate = thr_mma.partition_fragment_B(sBGate(_, _, 0));
    auto tCrDUp = thr_mma.partition_fragment_C(sUpOut);
    auto tCrDGate = thr_mma.partition_fragment_C(sGateOut);
    clear(tCrDUp);
    clear(tCrDGate);

    auto s2r_copy_a = make_tiled_copy_A(S2RA{}, tiled_mma);
    auto s2r_thr_a = s2r_copy_a.get_slice(tid);
    auto tAsA = s2r_thr_a.partition_S(sA_tiles);
    auto tCrA_view = s2r_thr_a.retile_D(tCrA);

    auto s2r_copy_b_up = make_tiled_copy_B(S2RB{}, tiled_mma);
    auto s2r_thr_b_up = s2r_copy_b_up.get_slice(tid);
    auto tBsBUp = s2r_thr_b_up.partition_S(sBUp);
    auto tCrBUpView = s2r_thr_b_up.retile_D(tCrBUp);

    auto s2r_copy_b_gate = make_tiled_copy_B(S2RB{}, tiled_mma);
    auto s2r_thr_b_gate = s2r_copy_b_gate.get_slice(tid);
    auto tBsBGate = s2r_thr_b_gate.partition_S(sBGate);
    auto tCrBGateView = s2r_thr_b_gate.retile_D(tCrBGate);

    G2SCB g2s_copy_b_up;
    auto g2s_thr_b_up = g2s_copy_b_up.get_slice(tid);
    auto tBgBUpCopy = g2s_thr_b_up.partition_S(gBUpTiles);
    auto tBsBUpCopy = g2s_thr_b_up.partition_D(sBUp);

    G2SCB g2s_copy_b_gate;
    auto g2s_thr_b_gate = g2s_copy_b_gate.get_slice(tid);
    auto tBgBGateCopy = g2s_thr_b_gate.partition_S(gBGateTiles);
    auto tBsBGateCopy = g2s_thr_b_gate.partition_D(sBGate);

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
        if constexpr (kUpgateStagedA) {
            stage_activation_tile_async<kHidden>(
                gmem_a_ptr, Ashm, batch_size,
                pipeline.next_tile, pipeline.write_stage);
        }
        issue_gemm_g2s_128(
            g2s_copy_b_up,
            tBgBUpCopy(_, _, _, pipeline.next_tile),
            tBsBUpCopy(_, _, _, pipeline.write_stage));
        issue_gemm_g2s_128(
            g2s_copy_b_gate,
            tBgBGateCopy(_, _, _, pipeline.next_tile),
            tBsBGateCopy(_, _, _, pipeline.write_stage));
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
            const int a_read_tile = kUpgateStagedA ? pipeline.read_stage : itile;
            if (tid < mma_size) {
                cute::copy(s2r_copy_a, tAsA(_, _, ik, a_read_tile), tCrA_view(_, _, ik));
                cute::copy(s2r_copy_b_up, tBsBUp(_, _, ik, pipeline.read_stage), tCrBUpView(_, _, ik));
                cute::copy(s2r_copy_b_gate, tBsBGate(_, _, ik, pipeline.read_stage), tCrBGateView(_, _, ik));
            }

            if (ik == 0) {
                if (pipeline.has_next(ntile)) {
                    if constexpr (kUpgateStagedA) {
                        stage_activation_tile_async<kHidden>(
                            gmem_a_ptr, Ashm, batch_size,
                            pipeline.next_tile, pipeline.write_stage);
                    }
                    issue_gemm_g2s_128(
                        g2s_copy_b_up,
                        tBgBUpCopy(_, _, _, pipeline.next_tile),
                        tBsBUpCopy(_, _, _, pipeline.write_stage));
                    issue_gemm_g2s_128(
                        g2s_copy_b_gate,
                        tBgBGateCopy(_, _, _, pipeline.next_tile),
                        tBsBGateCopy(_, _, _, pipeline.write_stage));
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
                cute::gemm(tiled_mma, tCrDUp, tCrA(_, _, ik), tCrBUp(_, _, ik), tCrDUp);
                cute::gemm(tiled_mma, tCrDGate, tCrA(_, _, ik), tCrBGate(_, _, ik), tCrDGate);
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
            auto cOut = make_identity_tensor(shape(sUpOut));
            auto tCcOut = thr_mma.partition_C(cOut);
            auto tCsUpOut = thr_mma.partition_C(sUpOut);
            CUTE_UNROLL
            for (int n = 0; n < (int)size(tCrDUp); ++n) {
                int mm = get<0>(tCcOut(n));
                int nn = get<1>(tCcOut(n));
                if (mm < batch_size && nn < kTileN_) {
                    float up_val = float(tCrDUp(n));
                    float gate_val = float(tCrDGate(n));
                    float silu = gate_val / (1.0f + __expf(-gate_val));
                    tCsUpOut(n) = silu * up_val;
                } else {
                    tCsUpOut(n) = 0.f;
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

__device__ __forceinline__ void decode_upgate_gemm_silu(
    LlamaGlobals& g,
    SmemLayout& smem,
    int inst_idx)
{
    // Instruction Parse
    const int32_t* inst = smem.instruction;
    int layer_idx = inst[1];
    int batch_size = inst[2];
    int tile_m_idx = inst[3];
    int num_blocks = inst[4];
    const int32_t* block_idxs = inst + 5;

    int tid = threadIdx.x;
    int sm_id = blockIdx.x;
    uint8_t* smem_base = smem.main_workspace;

#ifdef ENABLE_PROFILING
    uint64_t prof_stage_begin = 0;
#endif

    if (tile_m_idx != 0) {
#if DEBUG_PRINT
        if (tid == 0) {
            printf("upgate unsupported tile_m_idx=%d\n", tile_m_idx);
        }
#endif
        __trap();
    }
    if (num_blocks <= 0) {
        return;
    }

    // Barrier Wait: standalone MLP RMS (row 8 / opcode 9): one publish per batch row.
    auto barriers = g.get_barrier();
    volatile int32_t* input_ready =
        &barriers(layer_idx, OPCODE_MLP_RMS - 1, 0);
    if (tid == 0) {
        barrier_wait(
            input_ready,
            batch_size,
            sm_id,
            OPCODE_UPGATE_SILU,
            0,
            layer_idx,
            &g,
            inst_idx);
    }
    __syncthreads();

    // Resident-A Prologue: load from mlp_rms_out (written by decode_mlp_rms).
#ifdef ENABLE_PROFILING
    if (tid == 0) {
        prof_stage_begin = prof_clock_now();
    }
#endif
    if constexpr (!kUpgateStagedA) {
        load_batched_to_smem_a<
            KernelConfig::NUM_THREADS,
            LlamaConfig::HIDDEN_DIM,
            GemmConfig1Tile::SmemLayoutA
        >(
            g.mlp_rms_out_ptr,
            smem_ptr_at<bfloat16_t>(smem_base, UpgateGemmSmemLayout::kOffsetResidentA),
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

    const bfloat16_t* up_w = g.up_weight_ptr
        + (long)layer_idx * LlamaConfig::INTERM_DIM * LlamaConfig::HIDDEN_DIM;
    const bfloat16_t* gate_w = g.gate_weight_ptr
        + (long)layer_idx * LlamaConfig::INTERM_DIM * LlamaConfig::HIDDEN_DIM;
    // GEMM Tile Dispatch
    int i = 0;
    while (i < num_blocks) {
        int block_idx = block_idxs[i];
        bool use_two_tiles = (i + 1 < num_blocks) && (block_idxs[i + 1] == block_idx + 1);
        int n_blocks = use_two_tiles ? 2 : 1;
        int tile_n_global = block_idx * KernelConfig::kTileN;

        float* up_tile = smem_ptr_at<float>(smem_base, UpgateGemmSmemLayout::kOffsetUpScratch);
        float* gate_tile = smem_ptr_at<float>(smem_base, UpgateGemmSmemLayout::kOffsetGateScratch);
        const bfloat16_t* up_ptr = up_w + (long)tile_n_global * LlamaConfig::HIDDEN_DIM;
        const bfloat16_t* gate_ptr = gate_w + (long)tile_n_global * LlamaConfig::HIDDEN_DIM;

        if (use_two_tiles) {
            compute_upgate_tile_pair_to_float_smem<GemmConfig2Tile>(
                tid, batch_size, smem_base, g.mlp_rms_out_ptr,
                up_ptr, gate_ptr, up_tile, gate_tile, g, sm_id, inst_idx);
        } else {
            compute_upgate_tile_pair_to_float_smem<GemmConfig1Tile>(
                tid, batch_size, smem_base, g.mlp_rms_out_ptr,
                up_ptr, gate_ptr, up_tile, gate_tile, g, sm_id, inst_idx);
        }

 #ifdef ENABLE_PROFILING
        if (tid == 0) {
            prof_stage_begin = prof_clock_now();
        }
#endif
        if (use_two_tiles) {
            vec_epilogue_store_float_to_bf16_gmem<KernelConfig::kTileM, GemmConfig2Tile::kTileN_, KernelConfig::NUM_THREADS>(
                up_tile, g.silu_out_ptr, LlamaConfig::INTERM_DIM,
                tile_n_global, batch_size, LlamaConfig::INTERM_DIM, g, tid);
        } else {
            vec_epilogue_store_float_to_bf16_gmem<KernelConfig::kTileM, GemmConfig1Tile::kTileN_, KernelConfig::NUM_THREADS>(
                up_tile, g.silu_out_ptr, LlamaConfig::INTERM_DIM,
                tile_n_global, batch_size, LlamaConfig::INTERM_DIM, g, tid);
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
                &barriers(layer_idx, OPCODE_UPGATE_SILU - 1, 0),
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
