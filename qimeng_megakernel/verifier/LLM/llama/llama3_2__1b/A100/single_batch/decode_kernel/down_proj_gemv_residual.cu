#include "globals.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

namespace megakernel {

using namespace cute;

constexpr int kDownProjOutputTileN = KernelConfig::NUM_WARPS;
using DownProjUpGateBuckets = SplitKBucketContract<
    LlamaConfig::INTERM_DIM,
    kDownProjOutputTileN,
    DecodeScheduleConfig::DOWN_PROJ_SPLIT_K_GRANULARITY>;
constexpr int kDownProjInputTileK =
    DownProjUpGateBuckets::CONSUMER_SPLIT_K;
constexpr int kDownProjInputVecs =
    kDownProjInputTileK * sizeof(bfloat16_t) / sizeof(uint128_t);

// ========================================================================
// Shared Memory Layout for Down Projection
// ========================================================================
struct DownProjSmem {
    alignas(128) bfloat16_t activation[kDownProjInputTileK];
    alignas(128) bfloat16_t
        weight_stage_0[kDownProjOutputTileN * kDownProjInputTileK];
    alignas(128) bfloat16_t
        weight_stage_1[kDownProjOutputTileN * kDownProjInputTileK];
    alignas(128) float gemv_output[kDownProjOutputTileN];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

static_assert(sizeof(DownProjSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "DownProj workspace exceeds the selected Ampere CTA budget");

// ========================================================================
// Helper: BFloat16 Atomic Add
// ========================================================================
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

// ========================================================================
// Instruction Parsing
// ========================================================================
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
        layer_idx = smem_inst[1];
        tile_M_idx = smem_inst[2];
        tile_N_start_idx = smem_inst[3];
        tile_N_end_idx = smem_inst[4];
        split_K_idx = smem_inst[5];
        tile_M_granularity = smem_inst[6];   // 16
        tile_N_granularity = smem_inst[7];
        split_K_granularity = smem_inst[8];
        num_blocks = tile_N_end_idx - tile_N_start_idx;
    }
};

// ========================================================================
// Kernel Function
// ========================================================================
__device__ __forceinline__ void down_proj_gemv_residual(
    LlamaGlobals& g,
    SmemLayout&   smem
) {
    constexpr int TileM = kDownProjOutputTileN;
    constexpr int TileK = kDownProjInputTileK;
    constexpr int HiddenDim = LlamaConfig::HIDDEN_DIM;
    constexpr int NumWarps = KernelConfig::NUM_WARPS;

    // Setup shared memory layout
    DownProjSmem& s = *reinterpret_cast<DownProjSmem*>(smem.main_workspace);

    ParsedInstruction inst(smem.instruction);

    if (inst.num_blocks == 0) return;

    int tid = threadIdx.x;

    bfloat16_t* s_input = s.activation;
    float* s_gemv_out = s.gemv_output;
    float* s_reduce_workspace = s.reduction_workspace;

    // Global tensors
    auto t_g_silu_out = g.get_silu_out();
    auto t_g_hidden = g.get_hidden_state();
    auto g_down_weight = g.get_down_weight();

    // Weight layer base: down_weight[layer, HiddenDim, IntermDim]
    auto g_layer_weight = g_down_weight(inst.layer_idx, _, _);

    // Column offset based on split_K_idx
    long col_start = inst.split_K_idx * inst.split_K_granularity;

    int total_blocks = inst.num_blocks;

    // Weight prefetch before barrier (overlap with UpGate spin).
    if (total_blocks > 0) {
        int global_block_idx = inst.tile_N_start_idx;
        long row_start = global_block_idx * inst.tile_N_granularity;
        for (int row = 0; row < TileM; ++row) {
            if (row_start + row < HiddenDim) {
                auto t_g_row = make_tensor(
                    make_gmem_ptr(reinterpret_cast<const uint128_t*>(&g_layer_weight(row_start + row, col_start))),
                    make_shape(Int<kDownProjInputVecs>{}),
                    LayoutRight{}
                );
                auto t_s_row = make_tensor(
                    make_smem_ptr(reinterpret_cast<uint128_t*>(&s.weight_stage_0[row * TileK])),
                    make_shape(Int<kDownProjInputVecs>{}),
                    LayoutRight{}
                );
                load_tile_async(t_g_row, t_s_row);
            }
        }
        cp_async_fence();
    }
    __syncthreads();

    // Wait only for the UpGate slice consumed by this split-K instruction.
    if (tid == 0) {
        int wait_opcode =
            BARRIER_ROW(OPCODE_RMS_UPGATE_SILU);
        auto barriers = g.get_barrier();
        int bucket =
            DownProjUpGateBuckets::consumer_bucket(inst.split_K_idx);
        int32_t* bar = &barriers(inst.layer_idx, wait_opcode, bucket);
        int expected_upgate_blocks =
            DownProjUpGateBuckets::expected_arrivals(bucket);
        barrier_wait(bar, expected_upgate_blocks);
    }
    __syncthreads();

    // Load input tile from silu_out based on split_K_idx.
    {
        long input_offset = inst.split_K_idx * inst.split_K_granularity;
        const bfloat16_t* g_input_ptr = t_g_silu_out.data().get() + input_offset;
        auto t_g = make_tensor(
            make_gmem_ptr(reinterpret_cast<const uint128_t*>(g_input_ptr)),
            make_shape(Int<kDownProjInputVecs>{}),
            LayoutRight{}
        );
        auto t_s = make_tensor(
            make_smem_ptr(reinterpret_cast<uint128_t*>(s_input)),
            make_shape(Int<kDownProjInputVecs>{}),
            LayoutRight{}
        );
        load_tile_async(t_g, t_s);
    }
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    // Main loop with double buffering.
    for (int i = 0; i < total_blocks; ++i) {
        int stage_curr = i % 2;
        int stage_next = (i + 1) % 2;
        int global_block_idx = inst.tile_N_start_idx + i;
        long row_start = global_block_idx * inst.tile_N_granularity;

        // Prefetch next tile
        if (i + 1 < total_blocks) {
            int next_global_block_idx = inst.tile_N_start_idx + (i + 1);
            long next_row_start = next_global_block_idx * inst.tile_N_granularity;
            bfloat16_t* s_dst = (stage_next == 0) ? s.weight_stage_0 : s.weight_stage_1;
            for (int row = 0; row < TileM; ++row) {
                if (next_row_start + row < HiddenDim) {
                    auto t_g_row = make_tensor(
                        make_gmem_ptr(reinterpret_cast<const uint128_t*>(&g_layer_weight(next_row_start + row, col_start))),
                        make_shape(Int<kDownProjInputVecs>{}),
                        LayoutRight{}
                    );
                    auto t_s_row = make_tensor(
                        make_smem_ptr(reinterpret_cast<uint128_t*>(&s_dst[row * TileK])),
                        make_shape(Int<kDownProjInputVecs>{}),
                        LayoutRight{}
                    );
                    load_tile_async(t_g_row, t_s_row);
                }
            }
            cp_async_fence();
        }

        // Wait for current stage
        if (i + 1 < total_blocks) {
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();

        // Compute GEMV
        bfloat16_t* s_weight_curr = (stage_curr == 0) ? s.weight_stage_0 : s.weight_stage_1;

        auto t_s_input = make_tensor(make_smem_ptr(s_input), make_shape(Int<TileK>{}), LayoutRight{});
        auto t_s_weight = make_tensor(make_smem_ptr(s_weight_curr), make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        auto t_s_reduce = make_tensor(make_smem_ptr(s_reduce_workspace), make_shape(Int<NumWarps>{}), LayoutRight{});
        auto t_s_output = make_tensor(make_smem_ptr(s_gemv_out), make_shape(Int<TileM>{}), LayoutRight{});

        compute_gemv_tile<NumWarps, TileM, TileK>(t_s_input, t_s_weight, t_s_reduce, t_s_output);
        __syncthreads();

        // Atomic add to residual (hidden_states).
        if (tid < TileM) {
            int row_in_tile = tid;
            int global_row = row_start + row_in_tile;

            if (global_row < HiddenDim) {
                float gemv_val = s_gemv_out[row_in_tile];
                atomic_add_bf16(&t_g_hidden(global_row), gemv_val);
            }
        }

        __syncthreads();

        // Update barrier
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

} // namespace megakernel
