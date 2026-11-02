#include "globals.cuh"
#include "kernel_config.cuh"
#include "instruction_list.cuh"
#include "utils.cuh"

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace megakernel {

using namespace cute;

constexpr int kBlockSize = 16;
constexpr int kHeadDim = QwenConfig::HEAD_DIM;
constexpr int kGqaGroup =
    QwenConfig::NUM_Q_HEADS / QwenConfig::NUM_KV_HEADS;
constexpr int kGroupWarps =
    DecodeScheduleConfig::ATTENTION_GROUP_WARPS;
constexpr int kTokensPerWarp =
    (kBlockSize + kGroupWarps - 1) / kGroupWarps;
constexpr int kActiveWarps = kGqaGroup * kGroupWarps;
constexpr int kHeadValuesPerLane =
    kHeadDim / KernelConfig::WARP_SIZE;
constexpr int kReduceDim = kHeadDim + 2;
using AttnQkvReadiness = GroupedQkvReadinessContract<
    QwenConfig::NUM_Q_HEADS,
    QwenConfig::NUM_KV_HEADS,
    QwenConfig::HEAD_DIM,
    KernelConfig::NUM_WARPS,
    DecodeScheduleConfig::QKV_RAW_READY_SLOT_BASE>;

static_assert(QwenConfig::NUM_Q_HEADS % QwenConfig::NUM_KV_HEADS == 0);
static_assert(kGroupWarps > 0 && kGroupWarps <= kBlockSize,
              "SplitKV needs between one warp and one warp per tile token");
static_assert(kActiveWarps <= KernelConfig::NUM_WARPS,
              "GQA attention warp groups exceed the CTA warp budget");
static_assert(kHeadDim % KernelConfig::WARP_SIZE == 0,
              "each lane must own an integral number of head values");
static_assert(kHeadDim % 8 == 0,
              "K/V cp.async vectors require eight bf16 values");
static_assert(
    DecodeScheduleConfig::ATTENTION_REDUCTION_WORKSPACE_CAPACITY <=
        DecodeScheduleConfig::ATTENTION_REDUCTION_LIST_CAPACITY,
    "every emitted SplitKV partial must fit the paired reduction instruction");

// Shared memory layout for attention splitkv operation
// Use manual pointer arithmetic to overlay Q vectors and reduction workspace on KV tiles
// since they're used at different stages
struct AttnSplitKVSmem {
    alignas(128) bfloat16_t k_tiles[2][kBlockSize * kHeadDim];
    alignas(128) bfloat16_t v_tiles[2][kBlockSize * kHeadDim];
    alignas(128) float q_vectors[kGqaGroup * kHeadDim];
    alignas(128) float reduce_workspace[kGqaGroup * kGroupWarps * kReduceDim];
};

static_assert(sizeof(AttnSplitKVSmem) <= KernelConfig::SHARED_MEMORY_SIZE,
              "SplitKV workspace exceeds the selected Ampere CTA budget");

__device__ __forceinline__ void decode_attention_splitkv(
    QwenGlobals& g,
    SmemLayout&   smem
) {
    const int encoded_layer_idx = smem.instruction[1];
    const int encoded_kv_head_idx = smem.instruction[2];
    const int encoded_num_partials = smem.instruction[3];
    const int encoded_partial_idx = smem.instruction[4];
    if (encoded_layer_idx < 0 || encoded_layer_idx >= QwenConfig::NUM_LAYERS ||
        encoded_kv_head_idx < 0 ||
        encoded_kv_head_idx >= QwenConfig::NUM_KV_HEADS ||
        encoded_num_partials <= 0 ||
        encoded_num_partials >
            DecodeScheduleConfig::ATTENTION_REDUCTION_WORKSPACE_CAPACITY ||
        encoded_num_partials >
            DecodeScheduleConfig::ATTENTION_REDUCTION_LIST_CAPACITY ||
        encoded_partial_idx < 0 || encoded_partial_idx >= encoded_num_partials) {
        __trap();
    }

    // ========================================================================
    // 0. Setup shared memory layout
    // ========================================================================
    AttnSplitKVSmem& s = *reinterpret_cast<AttnSplitKVSmem*>(smem.main_workspace);

    // ========================================================================
    // 1. Thread/Warp IDs
    // ========================================================================
    int tid = threadIdx.x;
    int lane_id = tid % KernelConfig::WARP_SIZE;
    int warp_id = tid / KernelConfig::WARP_SIZE;

    int my_head_idx = warp_id / kGroupWarps;
    int my_group_warp_idx = warp_id % kGroupWarps;

    // ========================================================================
    // 2. Instruction Parsing
    // ========================================================================
    struct ParsedInstruction {
        int layer_idx, kv_head_idx, num_partials, partial_idx;
        int kv_start, kv_len;

        __device__ __forceinline__ ParsedInstruction(const int32_t* inst_ptr, uint32_t pos_id) {
            layer_idx    = inst_ptr[1];
            kv_head_idx  = inst_ptr[2];
            num_partials = inst_ptr[3];
            partial_idx  = inst_ptr[4];

            int total_len = pos_id + 1;
            int block_size = kBlockSize;

            int total_blocks = (total_len + block_size - 1) / block_size;
            int blocks_per_partial = (total_blocks + num_partials - 1) / num_partials;

            int start_block = partial_idx * blocks_per_partial;
            int end_block   = min(start_block + blocks_per_partial, total_blocks);

            kv_start = start_block * block_size;
            int end_token = min(end_block * block_size, total_len);

            kv_len = max(0, end_token - kv_start);
        }
    };

    ParsedInstruction inst(smem.instruction, g.pos_id);

    // ========================================================================
    // 3. Attention-side preparation slot
    // ========================================================================
    constexpr int GQAGroup = kGqaGroup;
    constexpr int QDim = QwenConfig::NUM_Q_HEADS * kHeadDim;
    constexpr int KDim = QwenConfig::NUM_KV_HEADS * kHeadDim;
    constexpr float Log2E = 1.44269504f;
    int q_head_start = inst.kv_head_idx * GQAGroup;

    // Get KV cache tensors (5D layout: [Layers, Batch=1, Seq, Head, Dim])
    // Slice to this layer and batch: [Seq, Head, Dim]
    auto k_cache_layer = g.get_k_cache()(inst.layer_idx, 0, _, _, _);
    auto v_cache_layer = g.get_v_cache()(inst.layer_idx, 0, _, _, _);

    static_assert(QwenConfig::QKV_OUT_DIM % kBlockSize == 0);
    static_assert(
        AttnQkvReadiness::TOTAL_BLOCKS == QwenConfig::QKV_OUT_DIM / kBlockSize);

    auto barriers = g.get_barrier();
    int32_t* raw_ready = &barriers(
        inst.layer_idx,
        BARRIER_ROW(OPCODE_RMS_QKV_GEMV),
        AttnQkvReadiness::raw_ready_slot(inst.kv_head_idx));
    if (tid == 0) {
        barrier_wait(
            raw_ready,
            AttnQkvReadiness::expected_raw_arrivals(inst.kv_head_idx));
    }
    __syncthreads();

    const int prepared_slot =
        AttnQkvReadiness::prepared_slot(inst.kv_head_idx);
    int32_t* prepared = &barriers(
        inst.layer_idx,
        BARRIER_ROW(OPCODE_RMS_QKV_GEMV),
        prepared_slot);

    // QKNorm needs a complete head, so it is the attention pre-slot rather
    // than a tail on each flat QKV GEMV block.  Exactly one SplitKV CTA per KV
    // group prepares its two Q heads and one K/V head. The producer then uses
    // release/publish, and sibling partials use acquire/wait.
    if (inst.partial_idx ==
        DecodeScheduleConfig::ATTENTION_PREPARE_PRODUCER_PARTIAL) {
        auto qkv_raw = g.get_qkv_raw();
        auto q_norm = g.get_q_norm_weight()(inst.layer_idx, _);
        auto k_norm = g.get_k_norm_weight()(inst.layer_idx, _);
        auto rope_cos = g.get_rope_cos()(g.pos_id, _);
        auto rope_sin = g.get_rope_sin()(g.pos_id, _);
        auto q_post_rope = g.get_q_post_rope();
        auto reduction = make_tensor(
            make_smem_ptr(s.reduce_workspace),
            make_shape(Int<KernelConfig::NUM_WARPS>{}),
            LayoutRight{});

        #pragma unroll
        for (int local_head = 0; local_head < GQAGroup; ++local_head) {
            int raw_idx = (q_head_start + local_head) * kHeadDim + tid;
            float raw = (tid < kHeadDim)
                ? static_cast<float>(qkv_raw(raw_idx))
                : 0.0f;
            float sum_sq = block_reduce_sum<KernelConfig::NUM_WARPS>(
                raw * raw, reduction);
            float rms_scale = rsqrtf(sum_sq / float(kHeadDim) + g.rms_norm_eps);

            if (tid < kHeadDim) {
                int pair_idx = raw_idx ^ 1;
                float value = raw * rms_scale * static_cast<float>(q_norm(tid));
                float pair = static_cast<float>(qkv_raw(pair_idx)) * rms_scale *
                    static_cast<float>(q_norm(tid ^ 1));
                float sign = (tid & 1) ? 1.0f : -1.0f;
                float rotated = value * rope_cos(tid) +
                    sign * pair * rope_sin(tid);
                q_post_rope(raw_idx) = static_cast<bfloat16_t>(rotated);
            }
            __syncthreads();
        }

        int k_raw_base = QDim + inst.kv_head_idx * kHeadDim;
        float k_raw = (tid < kHeadDim)
            ? static_cast<float>(qkv_raw(k_raw_base + tid))
            : 0.0f;
        float k_sum_sq = block_reduce_sum<KernelConfig::NUM_WARPS>(
            k_raw * k_raw, reduction);
        float k_rms_scale =
            rsqrtf(k_sum_sq / float(kHeadDim) + g.rms_norm_eps);
        if (tid < kHeadDim) {
            float value = k_raw * k_rms_scale * static_cast<float>(k_norm(tid));
            float pair = static_cast<float>(qkv_raw(k_raw_base + (tid ^ 1))) *
                k_rms_scale * static_cast<float>(k_norm(tid ^ 1));
            float sign = (tid & 1) ? 1.0f : -1.0f;
            float rotated = value * rope_cos(tid) +
                sign * pair * rope_sin(tid);
            k_cache_layer(g.pos_id, inst.kv_head_idx, tid) =
                static_cast<bfloat16_t>(rotated);

            int v_raw_base = QDim + KDim + inst.kv_head_idx * kHeadDim;
            v_cache_layer(g.pos_id, inst.kv_head_idx, tid) =
                qkv_raw(v_raw_base + tid);
        }
        __syncthreads();
        if (tid == 0) {
            barrier_publish(prepared, 1);
        }
    } else {
        if (tid == 0) {
            barrier_wait(prepared, 1);
        }
        __syncthreads();
    }

    // Load the resident Q vectors after acquiring all Q/K/V dependencies.
    // Keep the cooperative load valid when the Q group has more elements
    // than the CTA has threads.
    constexpr int kQLoadElems = GQAGroup * kHeadDim;
    for (int q_idx = tid; q_idx < kQLoadElems;
         q_idx += KernelConfig::NUM_THREADS) {
        int h = q_idx / kHeadDim;
        int d = q_idx % kHeadDim;
        long q_offset = (long)(q_head_start + h) * kHeadDim + d;
        s.q_vectors[q_idx] = static_cast<float>(g.get_q_post_rope()(q_offset));
    }
    __syncthreads();

    float r_q[kHeadValuesPerLane];
    if (my_head_idx < GQAGroup) {
        #pragma unroll
        for (int value = 0; value < kHeadValuesPerLane; ++value) {
            r_q[value] = s.q_vectors[
                my_head_idx * kHeadDim + lane_id +
                value * KernelConfig::WARP_SIZE];
        }
    }

    // ========================================================================
    // Main Loop (Pipeline)
    // ========================================================================

    // Accumulators
    float r_m = -INFINITY;
    float r_l = 0.0f;
    float r_o[kHeadValuesPerLane] = {};

    int num_tiles = (inst.kv_len + kBlockSize - 1) / kBlockSize;
    int stage = 0;

    // CuTe: K/V tile smem [kBlockSize, HEAD_DIM] per stage
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
        int t_start = inst.kv_start + tile_idx * kBlockSize;
        int t_count = min(kBlockSize, inst.kv_len - tile_idx * kBlockSize);
        auto t_k_dst = get_k_tile(s_idx);
        auto t_v_dst = get_v_tile(s_idx);

        // 8 bfloat16 = 16 bytes per vector.
        constexpr int kVectorsPerToken = kHeadDim / 8;
        int num_vecs = (t_count * kHeadDim) / 8;
        for (int i = tid; i < num_vecs; i += blockDim.x) {
            int token_in_tile = i / kVectorsPerToken;
            int vec_in_token  = i % kVectorsPerToken;
            int dim_offset    = vec_in_token * 8;
            int abs_token_idx = t_start + token_in_tile;

            cp_async_cg<16>(&t_k_dst(token_in_tile, dim_offset),
                            &k_cache_layer(abs_token_idx, inst.kv_head_idx, dim_offset));
            cp_async_cg<16>(&t_v_dst(token_in_tile, dim_offset),
                            &v_cache_layer(abs_token_idx, inst.kv_head_idx, dim_offset));
        }
        cp_async_fence();
    };

    if (num_tiles > 0) {
        load_tile_manual(0, 0);
    }

    for (int tile = 0; tile < num_tiles; ++tile) {
        if (tile + 1 < num_tiles) load_tile_manual(tile + 1, stage ^ 1);

        if (tile + 1 < num_tiles) cp_async_wait<1>();
        else cp_async_wait<0>();
        __syncthreads();

        if (my_head_idx < GQAGroup) {
            int t_count = min(kBlockSize, inst.kv_len - tile * kBlockSize);
            int my_t_start = my_group_warp_idx * kTokensPerWarp;
            int my_t_end   = min(my_t_start + kTokensPerWarp, t_count);

            auto t_k_curr = get_k_tile(stage);
            auto t_v_curr = get_v_tile(stage);

            for (int t = my_t_start; t < my_t_end; ++t) {
                // Q * K
                float score = 0.0f;
                #pragma unroll
                for (int value = 0; value < kHeadValuesPerLane; ++value) {
                    int dim = lane_id + value * KernelConfig::WARP_SIZE;
                    score = fmaf(
                        r_q[value],
                        static_cast<float>(t_k_curr(t, dim)),
                        score);
                }
                score = warp_reduce_sum(score);
                score *= (g.attn_scale * Log2E);

                // Update Accumulators (Online Softmax)
                float m_prev = r_m;
                float m_new = max(m_prev, score);
                float exp_score = exp2f(score - m_new);
                float correction = exp2f(m_prev - m_new);

                r_l = r_l * correction + exp_score;
                #pragma unroll
                for (int value = 0; value < kHeadValuesPerLane; ++value) {
                    int dim = lane_id + value * KernelConfig::WARP_SIZE;
                    float v = static_cast<float>(t_v_curr(t, dim));
                    r_o[value] =
                        r_o[value] * correction + exp_score * v;
                }
                r_m = m_new;
            }
        }

        stage ^= 1;
        __syncthreads();
    }

    // ========================================================================
    // Epilogue (Intra-Group Reduction)
    // ========================================================================

    // CuTe: [GqaGroup][GroupWarps][HeadDim + m + l]
    auto t_s_reduce = make_tensor(
        make_smem_ptr(s.reduce_workspace),
        make_shape(Int<GQAGroup>{}, Int<kGroupWarps>{}, Int<kReduceDim>{}),
        make_stride(Int<kGroupWarps * kReduceDim>{}, Int<kReduceDim>{}, _1{})
    );

    // Write Partial Results to Shared Memory (dim 64 = m, 65 = l)
    constexpr int kM = QwenConfig::HEAD_DIM;
    constexpr int kL = QwenConfig::HEAD_DIM + 1;
    if (my_head_idx < GQAGroup) {
        #pragma unroll
        for (int value = 0; value < kHeadValuesPerLane; ++value) {
            t_s_reduce(
                my_head_idx,
                my_group_warp_idx,
                lane_id + value * KernelConfig::WARP_SIZE) = r_o[value];
        }
        if (lane_id == 0) {
            t_s_reduce(my_head_idx, my_group_warp_idx, kM)     = r_m;
            t_s_reduce(my_head_idx, my_group_warp_idx, kL) = r_l;
        }
    }
    __syncthreads();

    // Reduce across group warps (only warp 0 of each head group)
    if (my_head_idx < GQAGroup && my_group_warp_idx == 0) {
        float final_m  = t_s_reduce(my_head_idx, 0, kM);
        float final_l  = t_s_reduce(my_head_idx, 0, kL);
        float final_o[kHeadValuesPerLane];
        #pragma unroll
        for (int value = 0; value < kHeadValuesPerLane; ++value) {
            final_o[value] = t_s_reduce(
                my_head_idx,
                0,
                lane_id + value * KernelConfig::WARP_SIZE);
        }

        for (int w = 1; w < kGroupWarps; ++w) {
            float next_m = t_s_reduce(my_head_idx, w, kM);
            if (next_m == -INFINITY) continue;

            float next_l = t_s_reduce(my_head_idx, w, kL);
            float new_max = max(final_m, next_m);
            float scale_final = exp2f(final_m - new_max);
            float scale_next  = exp2f(next_m - new_max);

            final_l = final_l * scale_final + next_l * scale_next;

            #pragma unroll
            for (int value = 0; value < kHeadValuesPerLane; ++value) {
                float next_o = t_s_reduce(
                    my_head_idx,
                    w,
                    lane_id + value * KernelConfig::WARP_SIZE);
                final_o[value] =
                    final_o[value] * scale_final + next_o * scale_next;
            }

            final_m = new_max;
        }

        // Normalize
        if (final_l > 1e-6f) {
            float inv_l = 1.0f / final_l;
            #pragma unroll
            for (int value = 0; value < kHeadValuesPerLane; ++value) {
                final_o[value] *= inv_l;
            }
        }
        float final_lse = (final_m == -INFINITY) ? -INFINITY : (final_m + log2f(final_l));

        // Write Output using CuTe tensor accessors
        int abs_q_head = q_head_start + my_head_idx;

        auto attn_lse_inter = g.get_attn_lse_intermediates();
        auto attn_out_inter = g.get_attn_out_intermediates();

        if (lane_id == 0) {
            attn_lse_inter(abs_q_head, inst.partial_idx) = final_lse;
        }

        #pragma unroll
        for (int value = 0; value < kHeadValuesPerLane; ++value) {
            attn_out_inter(
                abs_q_head,
                inst.partial_idx,
                lane_id + value * KernelConfig::WARP_SIZE) = final_o[value];
        }
    }

    __syncthreads();

    // Update barriers
    if (tid < GQAGroup) {
        int abs_q_head = q_head_start + tid;
        int my_opcode_idx =
            BARRIER_ROW(OPCODE_ATTENTION_PARTIAL);
        auto barriers = g.get_barrier();
        barrier_publish(&barriers(inst.layer_idx, my_opcode_idx, abs_q_head), 1);
    }
}

} // namespace megakernel
