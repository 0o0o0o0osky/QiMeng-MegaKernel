#include "globals.cuh"
#include "qwen_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

namespace megakernel {

using namespace cute;

constexpr int DownProjTileM     = KernelConfig::NUM_WARPS;
constexpr int TileKFull         =
    DecodeScheduleConfig::DOWN_PROJ_GEMV_K_SEGMENT;
constexpr int InnerKSlices      =
    DecodeScheduleConfig::DOWN_PROJ_SPLIT_K_GRANULARITY / TileKFull;
constexpr int DownProjHiddenDim = QwenConfig::HIDDEN_DIM;
constexpr int DownProjIntermDim = QwenConfig::INTERM_DIM;
using DownProjUpGateBuckets = SplitKBucketContract<
    DownProjIntermDim,
    DownProjTileM,
    DecodeScheduleConfig::DOWN_PROJ_SPLIT_K_GRANULARITY>;

static_assert(TileKFull % 8 == 0, "TileKFull must be divisible by 8 for uint128_t loads");
static_assert(DecodeScheduleConfig::DOWN_PROJ_SPLIT_K_GRANULARITY % TileKFull == 0);

struct DownProjSmem {
    alignas(128) bfloat16_t activation[TileKFull];                    // one K-slice at a time
    alignas(128) bfloat16_t weight_stage_0[DownProjTileM * TileKFull];
    alignas(128) bfloat16_t weight_stage_1[DownProjTileM * TileKFull];
    alignas(128) float      gemv_output[DownProjTileM];
    alignas(128) float      reduction_workspace[KernelConfig::NUM_WARPS];
};

static_assert(sizeof(DownProjSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "DownProj workspace exceeds the selected Ampere CTA budget");

__device__ __forceinline__ void atomic_add_bf16(bfloat16_t* address, float val) {
    unsigned short* address_us = reinterpret_cast<unsigned short*>(address);
    unsigned short old = *address_us;
    unsigned short assumed;
    do {
        assumed = old;
        bfloat16_t current_bf;
        *reinterpret_cast<unsigned short*>(&current_bf) = assumed;
        float new_val = float(current_bf) + val;
        bfloat16_t new_bf = bfloat16_t(new_val);
        unsigned short new_us = *reinterpret_cast<unsigned short*>(&new_bf);
        old = atomicCAS(address_us, assumed, new_us);
    } while (assumed != old);
}

struct ParsedInstruction {
    int layer_idx;
    int tile_M_idx;
    int tile_N_start_idx;
    int tile_N_end_idx;
    int split_K_idx;
    int tile_M_granularity;
    int tile_N_granularity;
    int split_K_granularity;
    int num_blocks;

    __device__ __forceinline__ ParsedInstruction(const int32_t* smem_inst) {
        layer_idx         = smem_inst[1];
        tile_M_idx        = smem_inst[2];
        tile_N_start_idx  = smem_inst[3];
        tile_N_end_idx    = smem_inst[4];
        split_K_idx       = smem_inst[5];
        tile_M_granularity  = smem_inst[6];
        tile_N_granularity  = smem_inst[7];
        split_K_granularity = smem_inst[8];
        num_blocks = tile_N_end_idx - tile_N_start_idx;
    }
};

__device__ __forceinline__ void down_proj_gemv_residual(
    QwenGlobals& g,
    SmemLayout&  smem
) {
    constexpr int NumWarps = KernelConfig::NUM_WARPS;

    DownProjSmem& s = *reinterpret_cast<DownProjSmem*>(smem.main_workspace);
    ParsedInstruction inst(smem.instruction);

    if (inst.num_blocks == 0) return;

    int tid = threadIdx.x;
    bfloat16_t* s_input          = s.activation;
    float*      s_gemv_out       = s.gemv_output;
    float*      s_reduce_workspace = s.reduction_workspace;

    auto t_g_silu_out  = g.get_silu_out();
    auto t_g_hidden    = g.get_hidden_state();
    auto g_down_weight = g.get_down_weight();

    // With 4-split-K, split_K_granularity=2752 always = InnerKSlices * TileKFull.
    long col_start = static_cast<long>(inst.split_K_idx) * inst.split_K_granularity;

    if (tid == 0) {
        int wait_opcode_idx = BARRIER_ROW(OPCODE_RMS_UPGATE_SILU);
        auto barriers = g.get_barrier();
        int bucket =
            DownProjUpGateBuckets::consumer_bucket(inst.split_K_idx);
        int32_t* bar = &barriers(inst.layer_idx, wait_opcode_idx, bucket);
        barrier_wait(
            bar, DownProjUpGateBuckets::expected_arrivals(bucket));
    }
    __syncthreads();

    auto g_layer_weight = g_down_weight(inst.layer_idx, _, _);
    int total_blocks    = inst.num_blocks;

    for (int i = 0; i < total_blocks; ++i) {
        int  global_block_idx = inst.tile_N_start_idx + i;
        long row_start        = static_cast<long>(global_block_idx) * inst.tile_N_granularity;

        float thread_sum = 0.f;  // accumulate over inner K-slices in float

        // ----- Prologue: prefetch weight for K-slice 0 → stage_0 -----
        for (int row = 0; row < DownProjTileM; ++row) {
            if (row_start + row < DownProjHiddenDim) {
                auto t_g_row = make_tensor(
                    make_gmem_ptr(reinterpret_cast<const uint128_t*>(
                        &g_layer_weight(row_start + row, col_start))),
                    make_shape(Int<TileKFull / 8>{}),
                    LayoutRight{}
                );
                auto t_s_row = make_tensor(
                    make_smem_ptr(reinterpret_cast<uint128_t*>(
                        &s.weight_stage_0[row * TileKFull])),
                    make_shape(Int<TileKFull / 8>{}),
                    LayoutRight{}
                );
                load_tile_async(t_g_row, t_s_row);
            }
        }
        cp_async_fence();  // Group A: weight K-slice 0

        // ----- Inner K-slice loop with double-buffer on weight -----
        for (int ks = 0; ks < InnerKSlices; ++ks) {
            int  stage_curr     = ks % 2;
            long col_k          = col_start + static_cast<long>(ks) * TileKFull;
            bfloat16_t* s_weight_curr = (stage_curr == 0) ? s.weight_stage_0
                                                           : s.weight_stage_1;

            // Prefetch weight for next K-slice (only on ks=0; ks=1 has no next)
            if (ks + 1 < InnerKSlices) {
                int  stage_next  = (ks + 1) % 2;
                long col_k_next  = col_k + TileKFull;
                bfloat16_t* s_dst = (stage_next == 0) ? s.weight_stage_0
                                                       : s.weight_stage_1;
                for (int row = 0; row < DownProjTileM; ++row) {
                    if (row_start + row < DownProjHiddenDim) {
                        auto t_g_row = make_tensor(
                            make_gmem_ptr(reinterpret_cast<const uint128_t*>(
                                &g_layer_weight(row_start + row, col_k_next))),
                            make_shape(Int<TileKFull / 8>{}),
                            LayoutRight{}
                        );
                        auto t_s_row = make_tensor(
                            make_smem_ptr(reinterpret_cast<uint128_t*>(
                                &s_dst[row * TileKFull])),
                            make_shape(Int<TileKFull / 8>{}),
                            LayoutRight{}
                        );
                        load_tile_async(t_g_row, t_s_row);
                    }
                }
                cp_async_fence();  // Group B: weight K-slice 1
            }

            // Wait for current K-slice weight; leave next-slice group in flight
            if (ks + 1 < InnerKSlices) {
                cp_async_wait<1>();  // Group A done; Group B may still be in flight
            } else {
                cp_async_wait<0>();  // Group B done (was already done after activation wait)
            }
            __syncthreads();

            // Load activation for this K-slice synchronously
            {
                const bfloat16_t* g_input_ptr = t_g_silu_out.data().get() + col_k;
                auto t_g = make_tensor(
                    make_gmem_ptr(reinterpret_cast<const uint128_t*>(g_input_ptr)),
                    make_shape(Int<TileKFull / 8>{}),
                    LayoutRight{}
                );
                auto t_s = make_tensor(
                    make_smem_ptr(reinterpret_cast<uint128_t*>(s_input)),
                    make_shape(Int<TileKFull / 8>{}),
                    LayoutRight{}
                );
                load_tile_async(t_g, t_s);
                cp_async_fence();
                cp_async_wait<0>();  // also drains any remaining weight group (Group B at ks=0)
                __syncthreads();
            }

            // Compute GEMV for this K-slice
            auto t_s_input  = make_tensor(make_smem_ptr(s_input),
                                          make_shape(Int<TileKFull>{}), LayoutRight{});
            auto t_s_weight = make_tensor(make_smem_ptr(s_weight_curr),
                                          make_shape(Int<DownProjTileM>{}, Int<TileKFull>{}),
                                          LayoutRight{});
            auto t_s_reduce = make_tensor(make_smem_ptr(s_reduce_workspace),
                                          make_shape(Int<NumWarps>{}), LayoutRight{});
            auto t_s_output = make_tensor(make_smem_ptr(s_gemv_out),
                                          make_shape(Int<DownProjTileM>{}), LayoutRight{});

            compute_gemv_tile<NumWarps, DownProjTileM, TileKFull>(
                t_s_input, t_s_weight, t_s_reduce, t_s_output);
            __syncthreads();

            // Accumulate K-slice contribution into float thread register
            if (tid < DownProjTileM) {
                thread_sum += s_gemv_out[tid];
            }
            __syncthreads();
        }

        // Single atomic write per N-tile (sum of both K-slices already in thread_sum)
        if (tid < DownProjTileM) {
            int  row_in_tile  = tid;
            int  global_row   = static_cast<int>(row_start) + row_in_tile;
            if (global_row < DownProjHiddenDim) {
                atomic_add_bf16(&t_g_hidden(global_row), thread_sum);
            }
        }

        __syncthreads();

        if (tid == 0) {
            auto barriers = g.get_barrier();
            barrier_publish(
                &barriers(
                    inst.layer_idx,
                    BARRIER_ROW(OPCODE_DOWN_PROJ_RESIDUAL),
                    0),
                1);
        }
    }
}

}  // namespace megakernel
