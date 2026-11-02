// Shared-expert paired GEMV + SiLU (opcode 10).
#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

struct SharedUpGateSiLUSmem {
    alignas(128) bfloat16_t activation[QwenMoeConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t weight_stage_0[16 * QwenMoeConfig::HIDDEN_DIM];
    alignas(128) bfloat16_t weight_stage_1[16 * QwenMoeConfig::HIDDEN_DIM];
    alignas(128) float up_output[16];
    alignas(128) float gate_output[16];
    alignas(128) float reduction_workspace[KernelConfig::NUM_WARPS];
};

__device__ __forceinline__ void shared_expert_upgate_silu(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    constexpr int TileM    = 16;
    constexpr int TileK    = QwenMoeConfig::HIDDEN_DIM;   // 2048
    constexpr int HiddenDim = QwenMoeConfig::HIDDEN_DIM;  // 2048
    constexpr int IntermDim = QwenMoeConfig::INTERM_DIM;  // 5632
    constexpr int NumWarps  = KernelConfig::NUM_WARPS;    // 16

    SharedUpGateSiLUSmem& s = *reinterpret_cast<SharedUpGateSiLUSmem*>(smem.main_workspace);

    struct ParsedInstruction {
        int layer_idx;
        int tile_M_idx;
        int tile_N_start_idx;
        int tile_N_end_idx;
        int init_shared_out;
        int tile_M_granularity;
        int tile_N_granularity;
        int num_blocks;

        __device__ __forceinline__ ParsedInstruction(const int32_t* smem_inst) {
            // [opcode, layer, tile_M, tile_N_start, tile_N_end, init, tile_M_gran, tile_N_gran]
            layer_idx          = smem_inst[1];
            tile_M_idx         = smem_inst[2];
            tile_N_start_idx   = smem_inst[3];
            tile_N_end_idx     = smem_inst[4];
            init_shared_out    = smem_inst[5];
            tile_M_granularity = smem_inst[6];
            tile_N_granularity = smem_inst[7];
            num_blocks         = tile_N_end_idx - tile_N_start_idx;
        }
    };

    ParsedInstruction inst(smem.instruction);

    int tid = threadIdx.x;

    // Independent pre-slot: exactly one instruction initializes shared_out.
    if (inst.init_shared_out) {
        bfloat16_t* base = g.get_shared_out().data().get();
        for (int idx = tid; idx < HiddenDim; idx += blockDim.x) {
            base[idx] = bfloat16_t(0.0f);
        }
    }
    __syncthreads();

    if (inst.num_blocks == 0) return;

    // Acquire opcode 8 slot 1: normalized hidden is ready.
    {
        constexpr int wait_opcode_row = OPCODE_ROUTER_GEMV - 1;
        auto t_barriers = g.get_barrier();
        const int32_t* bar = &t_barriers(inst.layer_idx, wait_opcode_row, 1);
        barrier_wait(bar, 1);
    }
    __syncthreads();

    // Dependent input staging: opcode 8 owns RMSNorm and publication.
    auto t_g_input = g.get_normed_hidden_state();
    auto t_s_norm  = make_tensor(
        make_smem_ptr(s.activation),
        make_shape(Int<HiddenDim>{}),
        LayoutRight{}
    );
    auto t_reduce_workspace = make_tensor(
        make_smem_ptr(s.reduction_workspace),
        make_shape(Int<NumWarps>{}),
        LayoutRight{}
    );

    for (int idx = tid; idx < HiddenDim; idx += blockDim.x) {
        t_s_norm(idx) = t_g_input(idx);
    }
    __syncthreads();

    // Up/Gate weight base pointers for this layer
    // t_g_up/gate are 3-D tensors [layers, IntermDim, HiddenDim]; .data().get() is
    // the global base pointer.  We add the layer offset manually so the arithmetic
    // stays identical to the Llama production kernel.
    const bfloat16_t* up_base   = g.get_shared_expert_up_proj_weight().data().get()
                                  + static_cast<long>(inst.layer_idx) * IntermDim * TileK;
    const bfloat16_t* gate_base = g.get_shared_expert_gate_proj_weight().data().get()
                                  + static_cast<long>(inst.layer_idx) * IntermDim * TileK;

    int total_iters = inst.num_blocks * 2;

    // Prologue: load first tile (block 0 Up Proj) into stage_0
    if (total_iters > 0) {
        int block_idx = inst.tile_N_start_idx;
        const bfloat16_t* d_tile_ptr = up_base + static_cast<long>(block_idx) * TileM * TileK;
        auto t_g_tile = make_tensor(make_gmem_ptr(d_tile_ptr),
                                    make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        auto t_s_tile = make_tensor(make_smem_ptr(s.weight_stage_0),
                                    make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
        load_tile_async(t_g_tile, t_s_tile);
        cp_async_fence();
    }

    for (int i = 0; i < total_iters; ++i) {
        // Even i: Up step (stage_0).  Odd i: Gate step (stage_1).
        bool is_gate_step  = (i % 2 != 0);
        int  curr_pair_idx = i / 2;
        int  curr_block_idx = inst.tile_N_start_idx + curr_pair_idx;

        // Prefetch next tile
        if (i + 1 < total_iters) {
            int  next_iter     = i + 1;
            bool next_is_gate  = (next_iter % 2 != 0);
            int  next_pair_idx = next_iter / 2;
            int  next_block_idx = inst.tile_N_start_idx + next_pair_idx;

            if (next_is_gate) {
                const bfloat16_t* d_next = gate_base + static_cast<long>(next_block_idx) * TileM * TileK;
                auto t_g_next = make_tensor(make_gmem_ptr(d_next),
                                            make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
                auto t_s_next = make_tensor(make_smem_ptr(s.weight_stage_1),
                                            make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
                load_tile_async(t_g_next, t_s_next);
            } else {
                const bfloat16_t* d_next = up_base + static_cast<long>(next_block_idx) * TileM * TileK;
                auto t_g_next = make_tensor(make_gmem_ptr(d_next),
                                            make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
                auto t_s_next = make_tensor(make_smem_ptr(s.weight_stage_0),
                                            make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
                load_tile_async(t_g_next, t_s_next);
            }
            cp_async_fence();
        }

        if (i + 1 < total_iters) { cp_async_wait<1>(); } else { cp_async_wait<0>(); }
        __syncthreads();

        // Compute GEMV
        if (!is_gate_step) {
            auto t_s_weight = make_tensor(make_smem_ptr(s.weight_stage_0),
                                          make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            auto t_s_output = make_tensor(make_smem_ptr(s.up_output),
                                          make_shape(Int<TileM>{}), LayoutRight{});
            compute_gemv_tile<NumWarps, TileM, TileK>(
                t_s_norm, t_s_weight, t_reduce_workspace, t_s_output);
        } else {
            auto t_s_weight = make_tensor(make_smem_ptr(s.weight_stage_1),
                                          make_shape(Int<TileM>{}, Int<TileK>{}), LayoutRight{});
            auto t_s_output = make_tensor(make_smem_ptr(s.gate_output),
                                          make_shape(Int<TileM>{}), LayoutRight{});
            compute_gemv_tile<NumWarps, TileM, TileK>(
                t_s_norm, t_s_weight, t_reduce_workspace, t_s_output);
        }
        __syncthreads();

        // Epilogue: SiLU + write back (gate step only)
        if (is_gate_step) {
            if (tid < TileM) {
                float up_val   = s.up_output[tid];
                float gate_val = s.gate_output[tid];
                float silu_gate = gate_val / (1.0f + __expf(-gate_val));
                float final_val = silu_gate * up_val;
                int global_idx  = curr_block_idx * TileM + tid;
                if (global_idx < IntermDim) {
                    g.get_shared_expert_silu_out()(global_idx) = bfloat16_t(final_val);
                }
            }
            __syncthreads();
            if (tid == 0) {
                auto t_barriers = g.get_barrier();
                barrier_publish(
                    &t_barriers(inst.layer_idx, OPCODE_SHARED_UPGATE_SILU - 1, 0), 1
                );
            }
        }
    }
}

}  // namespace megakernel
