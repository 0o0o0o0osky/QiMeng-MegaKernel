#include "instruction_list.cuh"
#include "utils.cuh"
#include <math.h>

using namespace kittens;
using namespace megakernel;

template <typename config, typename globals> struct attention_partial {
    static constexpr int opcode = OPCODE_ATTENTION_PARTIAL;
    static constexpr int NUM_STAGES = 3;
    static constexpr int GQA_RATIO = QWEN3_0_6B_NUM_ATTENTION_HEADS / QWEN3_0_6B_NUM_KV_HEADS;
    static constexpr int HEAD_DIM = QWEN3_0_6B_HEAD_DIM;
    static constexpr int QKV_BLOCK_SIZE = QWEN3_0_6B_MATVEC_BLOCK_SIZE;
    static constexpr int BLOCKS_PER_HEAD = QWEN3_0_6B_HEAD_DIM / QWEN3_0_6B_MATVEC_BLOCK_SIZE;
    static constexpr int HEAD_CHUNKS = BLOCKS_PER_HEAD;
    // Op1 publishes BLOCKS_PER_HEAD raw Q chunks in the per-Q-head slot.  Exactly
    // one Op2 partial then prepares the complete Q/K/V head and advances that same
    // slot once. All sibling partials use acquire/wait for the advanced value.
    static constexpr int PREPARE_PRODUCER_PARTIAL = 0;
    static constexpr int PREPARED_HEAD_COUNT = BLOCKS_PER_HEAD + 1;
    static constexpr int QOL_PAGE = 0;
    static constexpr int KV_PAGE0 = 1;
    static constexpr int KV_PAGE1 = 2;
    // Grouped attention launches one CTA per KV head and partial, so one wave
    // can contain at most floor(SMs / KV heads) partials for a group.
    static constexpr int MAX_ATTN_PARTIALS =
        globals::sm_count / QWEN3_0_6B_NUM_KV_HEADS;
    static constexpr int ROUNDED_MAX_ATTN_PARTIALS =
        ((MAX_ATTN_PARTIALS + 15) / 16) * 16;
    static constexpr int REDUCE_O_STAGES = 2;

    static_assert(GQA_RATIO <= 8, "GQA_RATIO must be <= 8");
    static_assert(GQA_RATIO == 2,
                  "Qwen3 grouped attention maps exactly two Q heads per CTA");
    static_assert(NUM_STAGES == 3, "NUM_STAGES must be 3 for Qwen3 H100");
    static_assert(HEAD_DIM % QKV_BLOCK_SIZE == 0,
                  "Full-head QKNorm requires exact flat head coverage.");
    static_assert(QKV_BLOCK_SIZE == 16,
                  "Attention QKNorm prologue assumes 16-element QKV output blocks.");

    using q_rt = kittens::rt_bf<16, QWEN3_0_6B_HEAD_DIM>;
    using q_st = kittens::st_bf<16, QWEN3_0_6B_HEAD_DIM>;
    using k_rt = kittens::rt_bf<QWEN3_0_6B_KV_BLOCK_SIZE, QWEN3_0_6B_HEAD_DIM>;
    using v_rt = kittens::rt_bf<QWEN3_0_6B_KV_BLOCK_SIZE, QWEN3_0_6B_HEAD_DIM, col_l>;
    using kv_st = kittens::st_bf<QWEN3_0_6B_KV_BLOCK_SIZE, QWEN3_0_6B_HEAD_DIM>;
    using attn_fl_rt = kittens::rt_fl<16, QWEN3_0_6B_KV_BLOCK_SIZE>;
    using attn_bf_rt = kittens::rt_bf<16, QWEN3_0_6B_KV_BLOCK_SIZE>;
    using max_vec_rv = col_vec<kittens::rt_fl<16, QWEN3_0_6B_HEAD_DIM>>;
    using norm_vec_rv = col_vec<kittens::rt_fl<16, QWEN3_0_6B_HEAD_DIM>>;
    using l_rv = col_vec<kittens::rt_fl<16, QWEN3_0_6B_HEAD_DIM>>;
    using l_sv = kittens::sv_fl<16>;
    using o_rt = kittens::rt_fl<16, QWEN3_0_6B_HEAD_DIM>;
    using o_sv = kittens::sv_fl<QWEN3_0_6B_HEAD_DIM>;
    using o_sv_bf = kittens::sv_bf<QWEN3_0_6B_HEAD_DIM>;
    using reduce_l_sv = kittens::sv_fl<ROUNDED_MAX_ATTN_PARTIALS>;
    using reduce_o_rv = kittens::rv_fl<QWEN3_0_6B_HEAD_DIM>;

    struct parsed_instruction {
        int layer_idx;
        int attention_head_idx;
        int num_partials;
        int partial_idx;
        __device__ inline int kv_head_idx() const {
            return attention_head_idx / GQA_RATIO;
        }
        __device__ inline parsed_instruction(typename config::instruction_t &instruction) {
            layer_idx = instruction[1];
            attention_head_idx = instruction[2];
            num_partials = instruction[3];
            partial_idx = instruction[4];
        }
        __device__ inline parsed_instruction(megakernel::state<config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline kittens::semaphore &Q_arrived(megakernel::state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &O_arrived(megakernel::state<config> &s) {
        return s.semaphores()[1];
    }
    __device__ static inline kittens::semaphore &L_arrived(megakernel::state<config> &s) {
        return s.semaphores()[2];
    }
    __device__ static inline kittens::semaphore &K_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &K_finished(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_finished(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &reduce_L_arrived(
        megakernel::state<config> &s, int q_head_local_idx) {
        return s.semaphores()[3 + NUM_STAGES * 4 + q_head_local_idx];
    }
    __device__ static inline kittens::semaphore &reduce_O_arrived(
        megakernel::state<config> &s, int q_head_local_idx, int stage) {
        return s.semaphores()[3 + NUM_STAGES * 4 + GQA_RATIO +
                              stage * GQA_RATIO + q_head_local_idx];
    }

    __device__ static inline void wait_QOL_page(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(QOL_PAGE));
    }
    __device__ static inline void wait_KV_pages(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(KV_PAGE0));
        s.wait_page_ready(s.pid(KV_PAGE1));
    }
    __device__ static inline void finish_QOL_page(megakernel::state<config> &s) {
        if (kittens::warp::laneid() == 0) {
            s.finish_page(s.pid(QOL_PAGE), config::NUM_CONSUMER_WARPS);
        }
    }
    __device__ static inline void finish_KV_pages(megakernel::state<config> &s) {
        if (kittens::warp::laneid() == 0) {
            s.finish_page(s.pid(KV_PAGE0), config::NUM_CONSUMER_WARPS);
            s.finish_page(s.pid(KV_PAGE1), config::NUM_CONSUMER_WARPS);
        }
    }
    __device__ static inline q_st &get_Q_smem(megakernel::state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<q_st *>(s.pages[pid].data);
    }
    __device__ static inline o_sv (&get_O_smem(megakernel::state<config> &s))[4] {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<o_sv(*)[4]>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st));
    }
    __device__ static inline l_sv &get_L_smem(megakernel::state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<l_sv *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st) + sizeof(o_sv) * 4);
    }
    static constexpr size_t REDUCE_L_OFFSET =
        sizeof(q_st) + sizeof(o_sv) * 4 + sizeof(l_sv);
    static constexpr size_t REDUCE_O_OFFSET =
        REDUCE_L_OFFSET + sizeof(reduce_l_sv) * GQA_RATIO;
    static_assert(REDUCE_O_OFFSET +
                      sizeof(o_sv) * GQA_RATIO * REDUCE_O_STAGES <=
                      config::PAGE_SIZE,
                  "Last-arrival reduction scratch exceeds QOL page.");

    __device__ static inline reduce_l_sv &get_reduce_L_smem(
        megakernel::state<config> &s, int q_head_local_idx) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<reduce_l_sv *>(
            reinterpret_cast<char *>(s.pages[pid].data) + REDUCE_L_OFFSET +
            sizeof(reduce_l_sv) * q_head_local_idx);
    }
    __device__ static inline o_sv &get_reduce_O_smem(
        megakernel::state<config> &s, int q_head_local_idx, int stage) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<o_sv *>(
            reinterpret_cast<char *>(s.pages[pid].data) + REDUCE_O_OFFSET +
            sizeof(o_sv) * (stage * GQA_RATIO + q_head_local_idx));
    }
    __device__ static inline kv_st &get_K_smem(megakernel::state<config> &s, int stage) {
        int page = (stage < 2) ? KV_PAGE0 : KV_PAGE1;
        int pair_idx = stage % 2;
        int pid = s.pid(page);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(kv_st) * (pair_idx * 2));
    }
    __device__ static inline kv_st &get_V_smem(megakernel::state<config> &s, int stage) {
        int page = (stage < 2) ? KV_PAGE0 : KV_PAGE1;
        int pair_idx = stage % 2;
        int pid = s.pid(page);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(kv_st) * (pair_idx * 2 + 1));
    }

    template <ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void store_4_rows(SV (&dst)[4], const RT &src, int row4idx) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == src.cols, "dst length must match src cols.");
        using T2 = RT::dtype;
        using T = base_types::packing<T2>::unpacked_type;
        using U = SV::dtype;
        using U2 = base_types::packing<U>::packed_type;
        uint32_t dst_ptr[4];
        dst_ptr[0] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[0].data[0]));
        dst_ptr[1] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[1].data[0]));
        dst_ptr[2] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[2].data[0]));
        dst_ptr[3] = static_cast<uint32_t>(__cvta_generic_to_shared(&dst[3].data[0]));
        int laneid = kittens::warp::laneid();
        int local_row_idx = (laneid % 16) / 4;
        int local_col_idx = laneid % 4;
        if (row4idx % 2 == 0 && laneid < 16) {
            if (row4idx / 2 == 0) {
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                }
            } else {
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                }
            }
        } else if (row4idx % 2 == 1 && laneid >= 16) {
            if (row4idx / 2 == 0) {
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                }
            } else {
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx, tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * (col_idx + 8), tmp[1]);
                }
            }
        }
    }

    template <ducks::rt::row_layout RT>
    __device__ static inline void right_fill(
        RT &dst, const RT &src, const int col_idx,
        const typename base_types::packing<typename RT::dtype>::unpacked_type &val = 0) {
        if (col_idx >= dst.cols) return;
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int j = 0; j < dst.width; j++) {
#pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    const int col_idx_x = (j * dst.tile_size_col) + ((k / 2) * 8) + ((kittens::warp::laneid() % 4) * 2);
                    const int col_idx_y = (j * dst.tile_size_col) + ((k / 2) * 8) + ((kittens::warp::laneid() % 4) * 2) + 1;
                    dst.tiles[i][j].data[k].x = (col_idx_x >= col_idx) ? val : src.tiles[i][j].data[k].x;
                    dst.tiles[i][j].data[k].y = (col_idx_y >= col_idx) ? val : src.tiles[i][j].data[k].y;
                }
            }
        }
    }

    __device__ static inline void load_Q_async(q_st &dst, const typename globals::q_post_rope_t &src, const int q_head_start_idx) {
        constexpr int elem_per_memcpy = sizeof(float4) / sizeof(typename q_st::dtype);
        constexpr int memcpy_per_row = QWEN3_0_6B_HEAD_DIM / elem_per_memcpy;
        typename globals::q_post_rope_t::dtype *src_ptr = &src.raw_ptr[q_head_start_idx * QWEN3_0_6B_HEAD_DIM];
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&dst.data[0]));
        int laneid = kittens::warp::laneid();
        int row = laneid / memcpy_per_row;
        int col = (laneid * elem_per_memcpy) % QWEN3_0_6B_HEAD_DIM;
        asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(dst.idx(dst_ptr, {row, col})), "l"(&src_ptr[row * QWEN3_0_6B_HEAD_DIM + col]) : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    struct controller {
        static_assert(
            qwen3_attention_release_lids_are_permutations(config::NUM_PAGES),
            "attention release_lid formulas must permute physical pages");

        static __device__ int release_lid(
            const globals &g, typename config::instruction_t &instruction, int &query) {
            // Keep live lids (QOL, KV0, KV1) at the tail to avoid reclaim races.
            return qwen3_attention_partial_release_lid_impl(
                config::NUM_PAGES, query);
        }
        static __device__ int init_semaphores(const globals &g, megakernel::state<config> &s) {
            init_semaphore(Q_arrived(s), 0, 1);
            init_semaphore(O_arrived(s), 0, 1);
            init_semaphore(L_arrived(s), 0, 1);
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(K_arrived(s, i), 0, 1);
                init_semaphore(V_arrived(s, i), 0, 1);
                init_semaphore(K_finished(s, i), 0, 1);
                init_semaphore(V_finished(s, i), 0, 1);
            }
            if (s.instruction()[3] > 1) {
                for (int h = 0; h < GQA_RATIO; ++h) {
                    init_semaphore(reduce_L_arrived(s, h), 0, 1);
                    for (int stage = 0; stage < REDUCE_O_STAGES; ++stage) {
                        init_semaphore(reduce_O_arrived(s, h, stage), 0, 1);
                    }
                }
                return 3 + 4 * NUM_STAGES +
                       GQA_RATIO * (1 + REDUCE_O_STAGES);
            }
            return 3 + 4 * NUM_STAGES;
        }
    };

    struct loader {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            auto laneid = kittens::warp::laneid();
            if (laneid >= 3 && laneid < config::NUM_PAGES) {
                int unused_page = s.pid(laneid);
                s.wait_page_ready(unused_page);
                s.finish_page(unused_page, config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct launcher {
        static __device__ void wait_for_op1_barriers(const globals &g, const parsed_instruction &inst) {
            const int abs_head = inst.attention_head_idx;
            const int kv_head = inst.kv_head_idx();
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{
                    inst.layer_idx, BARRIER_ROW(OPCODE_RMS_QKV_GEMV), abs_head}],
                BLOCKS_PER_HEAD);
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{
                    inst.layer_idx, BARRIER_ROW(OPCODE_RMS_QKV_GEMV), abs_head + 1}],
                BLOCKS_PER_HEAD);
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{
                    inst.layer_idx, BARRIER_ROW(OPCODE_RMS_QKV_GEMV),
                    QWEN3_0_6B_NUM_ATTENTION_HEADS + kv_head}],
                BLOCKS_PER_HEAD);
            megakernel::mk_barrier_wait(
                (volatile int32_t *)&g.Bar[{
                    inst.layer_idx, BARRIER_ROW(OPCODE_RMS_QKV_GEMV),
                    QWEN3_0_6B_NUM_ATTENTION_HEADS +
                        QWEN3_0_6B_NUM_KV_HEADS + kv_head}],
                BLOCKS_PER_HEAD);
        }

        // Attention-side prologue scratch: two Q heads, one K/V head, and one RMS
        // partial per output block. QKNorm cannot operate on Op1's flat chunks.
        static constexpr int PREPARE_QKV_SCRATCH_FLOATS =
            HEAD_DIM * 4 + HEAD_CHUNKS;
        static_assert(PREPARE_QKV_SCRATCH_FLOATS * sizeof(float) <=
                          config::SCRATCH_BYTES,
                      "Attention-side full-head QKNorm scratch exceeds budget.");

        static __device__ void prepare_attention_qknorm_rope(
            const globals &g, const parsed_instruction &inst,
            megakernel::state<config> &s) {
            const int abs_head = inst.attention_head_idx;
            const int kv_head = inst.kv_head_idx();
            const int q_dim = QWEN3_0_6B_NUM_ATTENTION_HEADS * HEAD_DIM;
            const int k_dim = QWEN3_0_6B_NUM_KV_HEADS * HEAD_DIM;
            const int pos_id = (int)g.pos_id;
            const int layer = inst.layer_idx;

            float *scratch = reinterpret_cast<float *>(&s.all_instructions[s.instruction_ring].scratch[0]);
            float *q0_f = scratch;
            float *q1_f = scratch + HEAD_DIM;
            float *k_f = scratch + HEAD_DIM * 2;
            float *v_f = scratch + HEAD_DIM * 3;
            void *rms_scratch = scratch + HEAD_DIM * 4;

            int q_start = abs_head * HEAD_DIM;
            int k_start = q_dim + kv_head * HEAD_DIM;
            int v_start = q_dim + k_dim + kv_head * HEAD_DIM;

            for (int i = kittens::warp::laneid(); i < HEAD_DIM; i += 32) {
                q0_f[i] = __bfloat162float(g.qkv_raw[{0, 0, 0, q_start + i}]);
                q1_f[i] = __bfloat162float(
                    g.qkv_raw[{0, 0, 0, q_start + HEAD_DIM + i}]);
                k_f[i] = __bfloat162float(g.qkv_raw[{0, 0, 0, k_start + i}]);
                v_f[i] = __bfloat162float(g.qkv_raw[{0, 0, 0, v_start + i}]);
            }
            kittens::warp::sync();

            rms_norm_full_head_inplace<HEAD_DIM>(q0_f, rms_scratch,
                                                 g.rms_norm_eps);
            rms_norm_full_head_inplace<HEAD_DIM>(q1_f, rms_scratch,
                                                 g.rms_norm_eps);
            rms_norm_full_head_inplace<HEAD_DIM>(k_f, rms_scratch,
                                                 g.rms_norm_eps);
            kittens::warp::sync();

            for (int i = kittens::warp::laneid(); i < HEAD_DIM; i += 32) {
                const float q_scale = __bfloat162float(
                    g.q_norm_weights[{0, 0, layer, i}]);
                q0_f[i] *= q_scale;
                q1_f[i] *= q_scale;
                k_f[i] *= __bfloat162float(
                    g.k_norm_weights[{0, 0, layer, i}]);
            }
            kittens::warp::sync();

            float *rope_cos_base = &g.rope_cos.raw_ptr[pos_id * HEAD_DIM];
            float *rope_sin_base = &g.rope_sin.raw_ptr[pos_id * HEAD_DIM];
            for (int i = kittens::warp::laneid(); i < HEAD_DIM; i += 32) {
                const int pair = i ^ 1;
                const float sign = (i & 1) ? 1.f : -1.f;
                const float q0_val = q0_f[i];
                const float q0_pair = q0_f[pair];
                const float q1_val = q1_f[i];
                const float q1_pair = q1_f[pair];
                const float k_val = k_f[i];
                const float k_pair = k_f[pair];
                q0_f[i] = q0_val * rope_cos_base[i] +
                          sign * q0_pair * rope_sin_base[i];
                q1_f[i] = q1_val * rope_cos_base[i] +
                          sign * q1_pair * rope_sin_base[i];
                k_f[i] = k_val * rope_cos_base[i] +
                         sign * k_pair * rope_sin_base[i];
            }
            kittens::warp::sync();

            // One head is only 256 bytes. Cooperative stores avoid the old
            // lane-0 tail of 8 Q plus 16 K/V TMA store-and-wait pairs.
            for (int i = kittens::warp::laneid(); i < HEAD_DIM; i += 32) {
                g.q_post_rope[{0, 0, 0, q_start + i}] =
                    __float2bfloat16(q0_f[i]);
                g.q_post_rope[{0, 0, 0, q_start + HEAD_DIM + i}] =
                    __float2bfloat16(q1_f[i]);
                g.k_cache[{layer, pos_id, kv_head, i}] =
                    __float2bfloat16(k_f[i]);
                g.v_cache[{layer, pos_id, kv_head, i}] =
                    __float2bfloat16(v_f[i]);
            }
            kittens::warp::sync();
            // Every lane owns a disjoint subset of the cooperative Q/K/V
            // stores.  Each producer must release its own writes before lane 0
            // publishes PREPARED_HEAD_COUNT to sibling attention CTAs.
            __threadfence();
            kittens::warp::sync();
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
#ifdef KITTENS_BLACKWELL
            if (kittens::warp::laneid() == 0) {
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
            }
#endif
            parsed_instruction inst{s};
            const int laneid = kittens::warp::laneid();

            // Op1 fans out flat QKV blocks. This attention opcode owns the
            // first point where a complete Q/K head is available for QKNorm.
            // Only partial 0 performs the prologue.  Its completed cooperative stores are
            // published by advancing Op1's Q-head counter from 8 (raw-ready) to 9
            // (prepared-ready); no extra barrier row or ABI slot is needed.
            if (inst.partial_idx == PREPARE_PRODUCER_PARTIAL) {
                wait_for_op1_barriers(g, inst);
                prepare_attention_qknorm_rope(g, inst, s);
                if (laneid == 0) {
                    for (int h = 0; h < GQA_RATIO; ++h) {
                        atomicAdd(&g.Bar[{
                            inst.layer_idx, BARRIER_ROW(OPCODE_RMS_QKV_GEMV),
                            inst.attention_head_idx + h}], 1);
                    }
                }
            }
            for (int h = 0; h < GQA_RATIO; ++h) {
                megakernel::mk_barrier_wait(
                    (volatile int32_t *)&g.Bar[{
                        inst.layer_idx, BARRIER_ROW(OPCODE_RMS_QKV_GEMV),
                        inst.attention_head_idx + h}],
                    PREPARED_HEAD_COUNT);
            }
            if (laneid == 0) {
                kittens::warp::arrive(Q_arrived(s));
            }
            if (laneid != 0) return;

            const int seq_len = g.pos_id + 1;
            const int total_attn_blocks =
                (seq_len + QWEN3_0_6B_KV_BLOCK_SIZE - 1) /
                QWEN3_0_6B_KV_BLOCK_SIZE;
            const int blocks_per_partial =
                (total_attn_blocks + inst.num_partials - 1) /
                inst.num_partials;
            const int start_blk_idx =
                inst.partial_idx * blocks_per_partial;
            const int end_blk_idx =
                min(start_blk_idx + blocks_per_partial, total_attn_blocks);

            wait_KV_pages(s);
            if (start_blk_idx >= end_blk_idx) {
                finish_KV_pages(s);
                return;
            }

            for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                const int cur_blk_idx = start_blk_idx + i;
                const int stage = i % NUM_STAGES;
                kv_st &K_smem = get_K_smem(s, stage);
                kv_st &V_smem = get_V_smem(s, stage);

                if (i >= NUM_STAGES) {
                    megakernel::mk_semaphore_wait(
                        K_finished(s, stage),
                        (i / NUM_STAGES - 1) % 2);
                    megakernel::mk_semaphore_wait(
                        V_finished(s, stage),
                        (i / NUM_STAGES - 1) % 2);
                }

                kittens::tma::expect(K_arrived(s, stage), K_smem);
                kittens::tma::load_async<dim::DEPTH,
                                         cache_policy::EVICT_FIRST>(
                    K_smem, g.k_cache,
                    {inst.layer_idx, cur_blk_idx, inst.kv_head_idx(), 0},
                    K_arrived(s, stage));
                kittens::tma::expect(V_arrived(s, stage), V_smem);
                kittens::tma::load_async<dim::DEPTH,
                                         cache_policy::EVICT_FIRST>(
                    V_smem, g.v_cache,
                    {inst.layer_idx, cur_blk_idx, inst.kv_head_idx(), 0},
                    V_arrived(s, stage));
            }
        }
    };

    struct consumer {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            if (kittens::warpid() != 0) return;

            parsed_instruction inst{s};
            int q_head_start_idx = inst.attention_head_idx;
            int seq_len = g.pos_id + 1;
            int total_attn_blocks = (seq_len + QWEN3_0_6B_KV_BLOCK_SIZE - 1) / QWEN3_0_6B_KV_BLOCK_SIZE;
            int blocks_per_partial = (total_attn_blocks + inst.num_partials - 1) / inst.num_partials;
            int start_blk_idx = inst.partial_idx * blocks_per_partial;
            int end_blk_idx = min(start_blk_idx + blocks_per_partial, total_attn_blocks);
            float softmax_temp = g.attn_scale * 1.44269504089f;

            q_rt Q_reg;
            k_rt K_reg;
            v_rt V_reg;
            l_rv L_reg;
            o_rt O_reg;
            attn_fl_rt attn_fl_reg;
            attn_bf_rt attn_bf_reg;
            max_vec_rv max_vec_reg;
            max_vec_rv scaled_max_vec_reg;
            max_vec_rv last_scaled_max_vec_reg;
            max_vec_rv diff_scaled_max_vec_reg;
            norm_vec_rv norm_vec_reg;

            kittens::warp::neg_infty(max_vec_reg);
            kittens::warp::zero(norm_vec_reg);
            kittens::warp::zero(O_reg);

            o_sv(&O_smem)[4] = get_O_smem(s);
            l_sv &L_smem = get_L_smem(s);
            q_st &Q_smem = get_Q_smem(s);

            wait_QOL_page(s);
            megakernel::mk_semaphore_wait(Q_arrived(s), 0);
            load_Q_async(Q_smem, g.q_post_rope, q_head_start_idx);
            kittens::warp::load_async_wait();
            kittens::warp::load(Q_reg, Q_smem);

            for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                int stage = i % NUM_STAGES;
                kv_st &K_smem = get_K_smem(s, stage);
                kv_st &V_smem = get_V_smem(s, stage);

                kittens::warp::zero(attn_fl_reg);
                megakernel::mk_semaphore_wait(
                    K_arrived(s, stage), (i / NUM_STAGES) % 2);
                kittens::warp::load(K_reg, K_smem);
                kittens::warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                kittens::warp::sync();
                kittens::warp::arrive(K_finished(s, stage));

                if ((i + start_blk_idx + 1) * QWEN3_0_6B_KV_BLOCK_SIZE > seq_len) {
                    right_fill(attn_fl_reg, attn_fl_reg, seq_len % QWEN3_0_6B_KV_BLOCK_SIZE, -INFINITY);
                }
                kittens::warp::row_max(max_vec_reg, attn_fl_reg, max_vec_reg);
                kittens::warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                kittens::warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);
                kittens::warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                kittens::warp::exp2(attn_fl_reg, attn_fl_reg);

                // The first block has no previous maximum.  Initializing the
                // previous maximum to zero and rescaling unconditionally can
                // produce exp2(0 - scaled_max) == +inf, followed by 0 * inf.
                if (i != 0) {
                    kittens::warp::sub(diff_scaled_max_vec_reg,
                                       last_scaled_max_vec_reg,
                                       scaled_max_vec_reg);
                    kittens::warp::exp2(diff_scaled_max_vec_reg,
                                        diff_scaled_max_vec_reg);
                    kittens::warp::mul_row(O_reg, O_reg,
                                           diff_scaled_max_vec_reg);
                    kittens::warp::mul(norm_vec_reg, norm_vec_reg,
                                       diff_scaled_max_vec_reg);
                }
                megakernel::mk_semaphore_wait(
                    V_arrived(s, stage), (i / NUM_STAGES) % 2);
                kittens::warp::load(V_reg, V_smem);
                kittens::warp::copy(attn_bf_reg, attn_fl_reg);
                kittens::warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                kittens::warp::sync();
                kittens::warp::arrive(V_finished(s, stage));

                kittens::warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);
                kittens::warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
            }

            if (start_blk_idx < end_blk_idx) {
                finish_KV_pages(s);
                kittens::warp::div_row(O_reg, O_reg, norm_vec_reg);
                kittens::warp::log2(L_reg, norm_vec_reg);
                kittens::warp::add(L_reg, L_reg, last_scaled_max_vec_reg);
            } else {
                kittens::warp::neg_infty(L_reg);
            }

            store_4_rows(O_smem, O_reg, 0);
            kittens::warp::sync();
            kittens::warp::arrive(O_arrived(s));
            kittens::warp::store(L_smem, L_reg);
            kittens::warp::sync();
            kittens::warp::arrive(L_arrived(s));
        }
    };

    struct storer {
        static inline __device__ void store_o_no_skip(const globals &g, megakernel::state<config> &s,
                                                     int q_head_start_idx, parsed_instruction &inst) {
            auto O_smem = get_O_smem(s);
            if (kittens::laneid() == 0) {
                megakernel::mk_semaphore_wait(O_arrived(s), 0);
                for (int h = 0; h < GQA_RATIO; ++h) {
                    kittens::tma::store_async<cache_policy::EVICT_LAST>(
                        g.attn_out_intermediates, O_smem[h],
                        {0, q_head_start_idx + h, inst.partial_idx, 0});
                }
            }
        }

        static inline __device__ void store_o_skip(const globals &g, megakernel::state<config> &s,
                                                   int q_head_start_idx) {
            auto O_smem = get_O_smem(s);
            if (kittens::laneid() == 0) {
                megakernel::mk_semaphore_wait(O_arrived(s), 0);
            }
            kittens::warp::sync();

            kittens::rv_bf<HEAD_DIM> O_bf;
            for (int h = 0; h < GQA_RATIO; ++h) {
                o_sv &smem_fl = O_smem[h];
                o_sv_bf &smem_bf = *reinterpret_cast<o_sv_bf *>(&smem_fl);
                kittens::warp::load(O_bf, smem_fl);
                kittens::warp::sync();
                kittens::warp::store(smem_bf, O_bf);
                kittens::warp::sync();
            }

            if (kittens::laneid() == 0) {
                for (int h = 0; h < GQA_RATIO; ++h) {
                    o_sv_bf &smem_bf =
                        *reinterpret_cast<o_sv_bf *>(&O_smem[h]);
                    kittens::tma::store_async<cache_policy::EVICT_LAST>(
                        g.attn_out, smem_bf,
                        {0, 0, 0, q_head_start_idx + h});
                }
            }
        }

        static inline __device__ void reduce_and_store_pair_last_arrival(
            const globals &g, megakernel::state<config> &s,
            int q_head_start_idx, const parsed_instruction &inst) {
            int laneid = kittens::warp::laneid();

            // Every lane below performs generic global loads for head 1, so
            // each lane takes an acquire after the producer release accounting.
            (void)megakernel::mk_bar_load_acquire(
                (const int *)&g.Bar[{
                    inst.layer_idx, BARRIER_ROW(opcode),
                    q_head_start_idx + 1}]);

            if (laneid == 0) {
                reduce_l_sv &L_partials = get_reduce_L_smem(s, 0);
                kittens::tma::expect(reduce_L_arrived(s, 0), L_partials);
                kittens::tma::load_async<cache_policy::EVICT_FIRST>(
                    L_partials, g.attn_lse_intermediates,
                    {0, 0, q_head_start_idx, 0}, reduce_L_arrived(s, 0));
            }
            megakernel::mk_semaphore_wait(reduce_L_arrived(s, 0), 0);

            static_assert(GQA_RATIO == 2,
                          "pair reducer requires exactly two query heads");
            reduce_o_rv accumulated_out_0, accumulated_out_1;
            reduce_o_rv current_out;
            float accumulated_lse_0 = -INFINITY;
            float accumulated_lse_1 = -INFINITY;
            kittens::warp::zero(accumulated_out_0);
            kittens::warp::zero(accumulated_out_1);

            if (laneid == 0) {
                o_sv &O_partial = get_reduce_O_smem(s, 0, 0);
                kittens::tma::expect(reduce_O_arrived(s, 0, 0), O_partial);
                kittens::tma::load_async<cache_policy::EVICT_FIRST>(
                    O_partial, g.attn_out_intermediates,
                    {0, q_head_start_idx, 0, 0},
                    reduce_O_arrived(s, 0, 0));
            }
            for (int i = 0; i < inst.num_partials; ++i) {
                const int stage = i % REDUCE_O_STAGES;
                megakernel::mk_semaphore_wait(
                    reduce_O_arrived(s, 0, stage),
                    (i / REDUCE_O_STAGES) % 2);
                // Start the next head-0 partial in the other buffer while the
                // current pair is reduced.
                if (laneid == 0 && i + 1 < inst.num_partials) {
                    const int next_stage = (i + 1) % REDUCE_O_STAGES;
                    o_sv &O_next = get_reduce_O_smem(s, 0, next_stage);
                    kittens::tma::expect(
                        reduce_O_arrived(s, 0, next_stage), O_next);
                    kittens::tma::load_async<cache_policy::EVICT_FIRST>(
                        O_next, g.attn_out_intermediates,
                        {0, q_head_start_idx, i + 1, 0},
                        reduce_O_arrived(s, 0, next_stage));
                }
                // Keep the two heads in distinct named accumulators and avoid
                // dynamically indexed local-memory accesses on this path.
                {
                    reduce_l_sv &L_partials = get_reduce_L_smem(s, 0);
                    o_sv &O_partial = get_reduce_O_smem(s, 0, stage);
                    float current_lse;
                    uint32_t lse_ptr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&L_partials.data[i]));
                    move<float>::lds(current_lse, lse_ptr);
                    kittens::warp::load(current_out, O_partial);

                    float max_lse = max(accumulated_lse_0, current_lse);
                    float accumulated_exp =
                        exp2f(accumulated_lse_0 - max_lse);
                    float current_exp = exp2f(current_lse - max_lse);
                    float denominator = accumulated_exp + current_exp;
                    float accumulated_scale = accumulated_exp / denominator;
                    float current_scale = current_exp / denominator;
                    kittens::warp::mul(accumulated_out_0,
                                       accumulated_out_0, accumulated_scale);
                    kittens::warp::mul(current_out, current_out,
                                       current_scale);
                    kittens::warp::add(accumulated_out_0,
                                       accumulated_out_0, current_out);
                    accumulated_lse_0 = max_lse + log2f(denominator);
                }
                {
                    float current_lse = laneid == 0
                        ? g.attn_lse_intermediates.raw_ptr[
                            (q_head_start_idx + 1) *
                                g.attn_lse_intermediates.cols() +
                            i]
                        : 0.0f;
                    current_lse =
                        __shfl_sync(0xffffffff, current_lse, 0);
                    float *O_global =
                        &g.attn_out_intermediates.raw_ptr[
                            ((q_head_start_idx + 1) *
                                 g.attn_out_intermediates.rows() +
                             i) *
                            HEAD_DIM];
#pragma unroll
                    for (int v = 0; v < reduce_o_rv::outer_dim; ++v) {
                        current_out.data[v][0] = O_global[v * 32 + laneid];
                    }

                    float max_lse = max(accumulated_lse_1, current_lse);
                    float accumulated_exp =
                        exp2f(accumulated_lse_1 - max_lse);
                    float current_exp = exp2f(current_lse - max_lse);
                    float denominator = accumulated_exp + current_exp;
                    float accumulated_scale = accumulated_exp / denominator;
                    float current_scale = current_exp / denominator;
                    kittens::warp::mul(accumulated_out_1,
                                       accumulated_out_1, accumulated_scale);
                    kittens::warp::mul(current_out, current_out,
                                       current_scale);
                    kittens::warp::add(accumulated_out_1,
                                       accumulated_out_1, current_out);
                    accumulated_lse_1 = max_lse + log2f(denominator);
                }
            }

            static_assert(reduce_o_rv::length == HEAD_DIM);
            static_assert(reduce_o_rv::inner_dim == 1);
#pragma unroll
            for (int i = 0; i < reduce_o_rv::outer_dim; ++i) {
                int col = i * 32 + laneid;
                g.attn_out.raw_ptr[q_head_start_idx * HEAD_DIM + col] =
                    __float2bfloat16(accumulated_out_0.data[i][0]);
                g.attn_out.raw_ptr[
                    (q_head_start_idx + 1) * HEAD_DIM + col] =
                    __float2bfloat16(accumulated_out_1.data[i][0]);
            }
            // Every lane owns a disjoint subset of the final global stores.
            // Publish each lane's writes before lane 0 releases o_proj.
            __threadfence();
            kittens::warp::sync();
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            parsed_instruction inst{s};
            int laneid = kittens::warp::laneid();
            int q_head_start_idx = inst.attention_head_idx;

            const bool skip_reduction = (inst.num_partials == 1);
            if (skip_reduction) {
                store_o_skip(g, s, q_head_start_idx);
            } else {
                store_o_no_skip(g, s, q_head_start_idx, inst);
            }
            kittens::tma::store_async_wait();
            if (laneid == 0) {
                // Complete the producer-side async-to-generic proxy handoff
                // before publishing completion through generic atomics.
                asm volatile("fence.proxy.async.global;" ::: "memory");
            }
            if (laneid == 0 && !skip_reduction) {
                l_sv &L_smem = get_L_smem(s);
                megakernel::mk_semaphore_wait(L_arrived(s), 0);
                for (int h = 0; h < GQA_RATIO; ++h) {
                    float tmp;
                    uint32_t src_ptr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&L_smem.data[h]));
                    float *dst_ptr = (float *)&g.attn_lse_intermediates.raw_ptr[
                        (q_head_start_idx + h) *
                            g.attn_lse_intermediates.cols() +
                        inst.partial_idx];
                    asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(tmp) : "r"(src_ptr));
                    asm volatile("st.global.f32 [%0], %1;\n" : : "l"(dst_ptr), "f"(tmp));
                }
            }
            kittens::warp::sync();
            if (skip_reduction) {
                if (laneid == 0) {
                    finish_QOL_page(s);
                    __threadfence();
                    atomicAdd(&g.Bar[{
                        inst.layer_idx,
                        BARRIER_ROW(OPCODE_ATTENTION_REDUCTION), 0}],
                        GQA_RATIO);
                }
                return;
            }

            int previous_arrivals = 0;
            if (laneid == 0) {
                // Publish both the completed TMA O store and the generic LSE
                // store before participating in the per-head arrival accounting.
                __threadfence();
                previous_arrivals = atomicAdd(&g.Bar[{
                    inst.layer_idx, BARRIER_ROW(opcode), q_head_start_idx}], 1);
                atomicAdd(&g.Bar[{
                    inst.layer_idx, BARRIER_ROW(opcode), q_head_start_idx + 1}], 1);
            }
            previous_arrivals = __shfl_sync(0xffffffff, previous_arrivals, 0);
            const bool is_last_arrival =
                previous_arrivals == inst.num_partials - 1;

            if (is_last_arrival) {
                if (laneid == 0) {
                    // Acquire both per-head ledgers before crossing from the
                    // generic memory proxy to the TMA async load proxy.
                    megakernel::mk_barrier_wait(
                        (volatile int32_t *)&g.Bar[{
                            inst.layer_idx, BARRIER_ROW(opcode), q_head_start_idx}],
                        inst.num_partials);
                    megakernel::mk_barrier_wait(
                        (volatile int32_t *)&g.Bar[{
                            inst.layer_idx, BARRIER_ROW(opcode), q_head_start_idx + 1}],
                        inst.num_partials);
                    asm volatile("fence.proxy.async.global;" ::: "memory");
                }
                kittens::warp::sync();
                reduce_and_store_pair_last_arrival(g, s, q_head_start_idx,
                                                   inst);
            }

            kittens::warp::sync();
            if (laneid == 0) {
                finish_QOL_page(s);
                if (is_last_arrival) {
                    __threadfence();
                    atomicAdd(&g.Bar[{
                        inst.layer_idx,
                        BARRIER_ROW(OPCODE_ATTENTION_REDUCTION), 0}],
                        GQA_RATIO);
                }
            }
        }
    };
};
