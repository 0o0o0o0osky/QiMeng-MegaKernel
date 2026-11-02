#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int HeadDim = QwenMoeConfig::HEAD_DIM;
constexpr int GqaGroup = QwenMoeConfig::ATTN_REDUCTION_SIZE;
constexpr int WarpSize = 32;
constexpr int MaxPartials = 128;

struct AttnReductionSmem {
    alignas(128) float lse_workspace[GqaGroup * MaxPartials];
    alignas(128) float data_workspace[2 * GqaGroup * HeadDim];
};

__device__ __forceinline__ void decode_attention_reduction(
    QwenGlobals& g,
    SmemLayout&  smem
) {
    AttnReductionSmem& s = *reinterpret_cast<AttnReductionSmem*>(smem.main_workspace);

    struct ParsedInstruction {
        int layer_idx;
        int q_head_start_idx;
        int num_partials;
        int reduction_list_length;
        int32_t* reduction_list;
        int is_terminal;

        __device__ __forceinline__ ParsedInstruction(int32_t* smem_inst) {
            layer_idx             = smem_inst[1];
            q_head_start_idx      = smem_inst[2];
            num_partials          = smem_inst[3];
            reduction_list_length = smem_inst[4];
            reduction_list        = &smem_inst[5];
            is_terminal           = smem_inst[5 + reduction_list_length];
        }
    };

    __syncthreads();
    ParsedInstruction inst(smem.instruction);
    __syncthreads();

    int tid = threadIdx.x;
    int warp_id = tid / WarpSize;
    int lane_id = tid % WarpSize;

    bool is_active = (warp_id < GqaGroup);
    int my_q_head_abs = inst.q_head_start_idx + warp_id;

    auto t_lse = make_tensor(
        make_smem_ptr(s.lse_workspace),
        make_shape(Int<GqaGroup>{}, Int<MaxPartials>{}),
        LayoutRight{}
    );
    auto t_data = make_tensor(
        make_smem_ptr(s.data_workspace),
        make_shape(Int<2>{}, Int<GqaGroup>{}, Int<HeadDim>{}),
        LayoutRight{}
    );

    {
        int wait_opcode_idx = 1;
        auto barriers = g.get_barrier();

        for (int i = 0; i < GqaGroup; ++i) {
            const int32_t* bar = &barriers(inst.layer_idx, wait_opcode_idx, inst.q_head_start_idx + i);
            barrier_wait(bar, inst.num_partials);
        }
    }
    __syncthreads();

    if (is_active) {
        auto attn_lse_inter = g.get_attn_lse_intermediates();

        for (int i = lane_id; i < inst.num_partials; i += WarpSize) {
            int real_partial_idx = inst.reduction_list[i];
            if (real_partial_idx < 0 || real_partial_idx >= g.max_splits) {
                __trap();
            }
            t_lse(warp_id, i) = attn_lse_inter(my_q_head_abs, real_partial_idx);
        }
        __syncwarp();
    }

    if (is_active) {
        float acc_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float acc_lse = -INFINITY;
        auto attn_out_inter = g.get_attn_out_intermediates();

        auto load_stage_async = [&](int step, int stage_idx) {
            int real_partial_idx = inst.reduction_list[step];
            if (real_partial_idx < 0 || real_partial_idx >= g.max_splits) {
                __trap();
            }
            if (lane_id < 32) {
                int vec_offset = lane_id * 4;
                cp_async_cg<16>(&t_data(stage_idx, warp_id, vec_offset),
                               &attn_out_inter(my_q_head_abs, real_partial_idx, vec_offset));
            }
        };

        if (inst.num_partials > 0) {
            load_stage_async(0, 0);
            cp_async_fence();
        }

        int stage = 0;
        for (int step = 0; step < inst.num_partials; ++step) {
            if (step + 1 < inst.num_partials) {
                load_stage_async(step + 1, stage ^ 1);
                cp_async_fence();
            }

            if (step + 1 < inst.num_partials) {
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
            __syncwarp();

            float curr_lse = t_lse(warp_id, step);
            float curr_o[4];
            curr_o[0] = t_data(stage, warp_id, lane_id);
            curr_o[1] = t_data(stage, warp_id, lane_id + WarpSize);
            curr_o[2] = t_data(stage, warp_id, lane_id + 64);
            curr_o[3] = t_data(stage, warp_id, lane_id + 96);

            float max_lse = fmaxf(acc_lse, curr_lse);
            float acc_exp = exp2f(acc_lse - max_lse);
            float curr_exp = exp2f(curr_lse - max_lse);

            float new_denom = acc_exp + curr_exp;
            float acc_scale = acc_exp / new_denom;
            float curr_scale = curr_exp / new_denom;

            acc_o[0] = acc_o[0] * acc_scale + curr_o[0] * curr_scale;
            acc_o[1] = acc_o[1] * acc_scale + curr_o[1] * curr_scale;
            acc_o[2] = acc_o[2] * acc_scale + curr_o[2] * curr_scale;
            acc_o[3] = acc_o[3] * acc_scale + curr_o[3] * curr_scale;
            acc_lse = max_lse + log2f(new_denom);

            stage ^= 1;
        }

        auto attn_out = make_tensor(
            make_gmem_ptr(g.attn_out_ptr),
            make_shape(Int<QwenMoeConfig::NUM_Q_HEADS>{}, Int<QwenMoeConfig::HEAD_DIM>{}),
            LayoutRight{}
        );
        attn_out(my_q_head_abs, lane_id)        = static_cast<bfloat16_t>(acc_o[0]);
        attn_out(my_q_head_abs, lane_id + 32)   = static_cast<bfloat16_t>(acc_o[1]);
        attn_out(my_q_head_abs, lane_id + 64)   = static_cast<bfloat16_t>(acc_o[2]);
        attn_out(my_q_head_abs, lane_id + 96)   = static_cast<bfloat16_t>(acc_o[3]);
    }

    __syncthreads();

    if (tid == 0) {
        int my_opcode_idx = 2;
        auto barriers = g.get_barrier();
        barrier_publish(&barriers(inst.layer_idx, my_opcode_idx, 0), GqaGroup);
    }
}

}  // namespace megakernel
