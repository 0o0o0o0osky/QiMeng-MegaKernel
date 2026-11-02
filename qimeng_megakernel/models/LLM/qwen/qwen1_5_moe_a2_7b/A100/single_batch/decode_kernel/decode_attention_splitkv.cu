// Qwen1.5-MoE A100: splitkv attention (opcode 2 only).

#include "globals.cuh"
#include "qwen_moe_config.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int kBlockSize = 16;
constexpr int kHeadDim = QwenMoeConfig::HEAD_DIM;
constexpr int kBlocksPerHead = kHeadDim / kBlockSize;
constexpr int GQAGroup = QwenMoeConfig::GQA_GROUP;

struct AttnSplitKVSmem {
    alignas(128) bfloat16_t k_tiles[2][kBlockSize * kHeadDim];
    alignas(128) bfloat16_t v_tiles[2][kBlockSize * kHeadDim];
    alignas(128) float q_vectors[GQAGroup * kHeadDim];
    alignas(128) float reduce_workspace[GQAGroup * 4 * (kHeadDim + 2)];
};

__device__ __forceinline__ void decode_attention_splitkv(
    QwenGlobals& g,
    SmemLayout&  smem
) {
    AttnSplitKVSmem& s = *reinterpret_cast<AttnSplitKVSmem*>(smem.main_workspace);

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;

    int my_head_idx = warp_id;

    struct ParsedInstruction {
        int layer_idx, kv_head_idx, num_partials, partial_idx;
        int kv_start, kv_len;

        __device__ __forceinline__ ParsedInstruction(const int32_t* inst_ptr, uint32_t pos_id) {
            layer_idx    = inst_ptr[1];
            kv_head_idx  = inst_ptr[2];
            num_partials = inst_ptr[3];
            partial_idx  = inst_ptr[4];

            int total_len = pos_id + 1;
            int total_blocks = (total_len + kBlockSize - 1) / kBlockSize;
            int blocks_per_partial = (total_blocks + num_partials - 1) / num_partials;
            int start_block = partial_idx * blocks_per_partial;
            int end_block   = min(start_block + blocks_per_partial, total_blocks);
            kv_start = start_block * kBlockSize;
            int end_token = min(end_block * kBlockSize, total_len);
            kv_len = max(0, end_token - kv_start);
        }
    };

    ParsedInstruction inst(smem.instruction, g.pos_id);

    constexpr float Log2E = 1.44269504f;
    int q_head_start = inst.kv_head_idx * GQAGroup;

    auto k_cache_layer = g.get_k_cache()(inst.layer_idx, 0, _, _, _);
    auto v_cache_layer = g.get_v_cache()(inst.layer_idx, 0, _, _, _);

    {
        int wait_opcode_idx = 0;
        auto barriers = g.get_barrier();
        for (int i = 0; i < GQAGroup; ++i) {
            int q_head_idx = q_head_start + i;
            const int32_t* bar = &barriers(inst.layer_idx, wait_opcode_idx, q_head_idx);
            barrier_wait(bar, kBlocksPerHead);
        }
    }
    __syncthreads();

    constexpr int kQLoadElems = GQAGroup * kHeadDim;
    for (int i = tid; i < kQLoadElems; i += blockDim.x) {
        int h = i / kHeadDim;
        int d = i % kHeadDim;
        s.q_vectors[i] = static_cast<float>(g.get_q_post_rope()((q_head_start + h) * kHeadDim + d));
    }
    __syncthreads();

    float r_q[4];
    if (my_head_idx < GQAGroup) {
        r_q[0] = s.q_vectors[my_head_idx * kHeadDim + lane_id];
        r_q[1] = s.q_vectors[my_head_idx * kHeadDim + lane_id + 32];
        r_q[2] = s.q_vectors[my_head_idx * kHeadDim + lane_id + 64];
        r_q[3] = s.q_vectors[my_head_idx * kHeadDim + lane_id + 96];
    }

    float r_m = -INFINITY;
    float r_l = 0.0f;
    float r_o[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    int num_tiles = (inst.kv_len + kBlockSize - 1) / kBlockSize;
    int stage = 0;

    auto get_k_tile = [&](int s_idx) {
        return make_tensor(
            make_smem_ptr(s.k_tiles[s_idx]),
            make_shape(Int<kBlockSize>{}, Int<kHeadDim>{}),
            LayoutRight{}
        );
    };
    auto get_v_tile = [&](int s_idx) {
        return make_tensor(
            make_smem_ptr(s.v_tiles[s_idx]),
            make_shape(Int<kBlockSize>{}, Int<kHeadDim>{}),
            LayoutRight{}
        );
    };

    auto load_tile_manual = [&](int tile_idx, int s_idx) {
        if (tile_idx == 0) {
            int wait_opcode_idx = 0;
            auto barriers = g.get_barrier();
            int slot_k = QwenMoeConfig::NUM_Q_HEADS + inst.kv_head_idx;
            int slot_v = QwenMoeConfig::NUM_Q_HEADS + QwenMoeConfig::NUM_KV_HEADS + inst.kv_head_idx;
            const int32_t* bar_k = &barriers(inst.layer_idx, wait_opcode_idx, slot_k);
            const int32_t* bar_v = &barriers(inst.layer_idx, wait_opcode_idx, slot_v);
            barrier_wait(bar_k, kBlocksPerHead);
            barrier_wait(bar_v, kBlocksPerHead);
        }
        __syncthreads();

        int t_start = inst.kv_start + tile_idx * kBlockSize;
        int t_count = min(kBlockSize, inst.kv_len - tile_idx * kBlockSize);
        auto t_k_dst = get_k_tile(s_idx);
        auto t_v_dst = get_v_tile(s_idx);

        int num_vecs = (t_count * kHeadDim) / 8;
        for (int i = tid; i < num_vecs; i += blockDim.x) {
            int token_in_tile = i / (kHeadDim / 8);
            int vec_in_token  = i % (kHeadDim / 8);
            int dim_offset    = vec_in_token * 8;
            int abs_token_idx = t_start + token_in_tile;
            cp_async_cg<16>(&t_k_dst(token_in_tile, dim_offset),
                            &k_cache_layer(abs_token_idx, inst.kv_head_idx, dim_offset));
            cp_async_cg<16>(&t_v_dst(token_in_tile, dim_offset),
                            &v_cache_layer(abs_token_idx, inst.kv_head_idx, dim_offset));
        }
        cp_async_fence();
    };

    if (num_tiles > 0) load_tile_manual(0, 0);

    for (int tile = 0; tile < num_tiles; ++tile) {
        if (tile + 1 < num_tiles) load_tile_manual(tile + 1, stage ^ 1);
        if (tile + 1 < num_tiles) cp_async_wait<1>();
        else cp_async_wait<0>();
        __syncthreads();

        if (my_head_idx < GQAGroup) {
            int t_count = min(kBlockSize, inst.kv_len - tile * kBlockSize);
            int my_t_start = 0;
            int my_t_end   = t_count;

            auto t_k_curr = get_k_tile(stage);
            auto t_v_curr = get_v_tile(stage);

            for (int t = my_t_start; t < my_t_end; ++t) {
                float k0 = static_cast<float>(t_k_curr(t, lane_id));
                float k1 = static_cast<float>(t_k_curr(t, lane_id + 32));
                float k2 = static_cast<float>(t_k_curr(t, lane_id + 64));
                float k3 = static_cast<float>(t_k_curr(t, lane_id + 96));
                float v0 = static_cast<float>(t_v_curr(t, lane_id));
                float v1 = static_cast<float>(t_v_curr(t, lane_id + 32));
                float v2 = static_cast<float>(t_v_curr(t, lane_id + 64));
                float v3 = static_cast<float>(t_v_curr(t, lane_id + 96));

                float score = r_q[0] * k0 + r_q[1] * k1 + r_q[2] * k2 + r_q[3] * k3;
                score = warp_reduce_sum(score);
                score *= (g.attn_scale * Log2E);

                float m_prev = r_m;
                float m_new = max(m_prev, score);
                float exp_score = exp2f(score - m_new);
                float correction = exp2f(m_prev - m_new);

                r_l = r_l * correction + exp_score;
                r_o[0] = r_o[0] * correction + exp_score * v0;
                r_o[1] = r_o[1] * correction + exp_score * v1;
                r_o[2] = r_o[2] * correction + exp_score * v2;
                r_o[3] = r_o[3] * correction + exp_score * v3;
                r_m = m_new;
            }
        }
        stage ^= 1;
        __syncthreads();
    }

    constexpr int kReduceDim = kHeadDim + 2;
    auto t_s_reduce = make_tensor(
        make_smem_ptr(s.reduce_workspace),
        make_shape(Int<GQAGroup>{}, Int<4>{}, Int<kReduceDim>{}),
        make_stride(Int<GQAGroup * kReduceDim>{}, Int<kReduceDim>{}, _1{})
    );
    constexpr int kM = kHeadDim;
    constexpr int kL = kHeadDim + 1;

    constexpr int WarpsPerGroup = 1;
    if (my_head_idx < GQAGroup) {
        t_s_reduce(my_head_idx, 0, lane_id)       = r_o[0];
        t_s_reduce(my_head_idx, 0, lane_id + 32)  = r_o[1];
        t_s_reduce(my_head_idx, 0, lane_id + 64)  = r_o[2];
        t_s_reduce(my_head_idx, 0, lane_id + 96)  = r_o[3];
        if (lane_id == 0) {
            t_s_reduce(my_head_idx, 0, kM) = r_m;
            t_s_reduce(my_head_idx, 0, kL) = r_l;
        }
    }
    __syncthreads();

    if (my_head_idx < GQAGroup) {
        float final_m  = t_s_reduce(my_head_idx, 0, kM);
        float final_l  = t_s_reduce(my_head_idx, 0, kL);
        float final_o0 = t_s_reduce(my_head_idx, 0, lane_id);
        float final_o1 = t_s_reduce(my_head_idx, 0, lane_id + 32);
        float final_o2 = t_s_reduce(my_head_idx, 0, lane_id + 64);
        float final_o3 = t_s_reduce(my_head_idx, 0, lane_id + 96);

        for (int w = 1; w < WarpsPerGroup; ++w) {
            float next_m = t_s_reduce(my_head_idx, w, kM);
            if (next_m == -INFINITY) continue;
            float next_l = t_s_reduce(my_head_idx, w, kL);
            float new_max = max(final_m, next_m);
            float scale_final = exp2f(final_m - new_max);
            float scale_next  = exp2f(next_m - new_max);
            final_l = final_l * scale_final + next_l * scale_next;
            float next_o0 = t_s_reduce(my_head_idx, w, lane_id);
            float next_o1 = t_s_reduce(my_head_idx, w, lane_id + 32);
            float next_o2 = t_s_reduce(my_head_idx, w, lane_id + 64);
            float next_o3 = t_s_reduce(my_head_idx, w, lane_id + 96);
            final_o0 = final_o0 * scale_final + next_o0 * scale_next;
            final_o1 = final_o1 * scale_final + next_o1 * scale_next;
            final_o2 = final_o2 * scale_final + next_o2 * scale_next;
            final_o3 = final_o3 * scale_final + next_o3 * scale_next;
            final_m = new_max;
        }

        if (final_l > 1e-6f) {
            float inv_l = 1.0f / final_l;
            final_o0 *= inv_l; final_o1 *= inv_l; final_o2 *= inv_l; final_o3 *= inv_l;
        }
        float final_lse = (final_m == -INFINITY) ? -INFINITY : (final_m + log2f(final_l));

        int abs_q_head = q_head_start + my_head_idx;
        auto attn_lse_inter = g.get_attn_lse_intermediates();
        auto attn_out_inter = g.get_attn_out_intermediates();
        if (lane_id == 0) attn_lse_inter(abs_q_head, inst.partial_idx) = final_lse;
        attn_out_inter(abs_q_head, inst.partial_idx, lane_id)       = final_o0;
        attn_out_inter(abs_q_head, inst.partial_idx, lane_id + 32)  = final_o1;
        attn_out_inter(abs_q_head, inst.partial_idx, lane_id + 64)  = final_o2;
        attn_out_inter(abs_q_head, inst.partial_idx, lane_id + 96)  = final_o3;
    }

    __syncthreads();

    if (tid < GQAGroup) {
        int abs_q_head = q_head_start + tid;
        int my_opcode_idx = 1;
        auto barriers = g.get_barrier();
        barrier_publish(&barriers(inst.layer_idx, my_opcode_idx, abs_q_head), 1);
    }
}

}  // namespace megakernel
