#include "utils.cuh"
#include <math.h>

using namespace kittens;
using namespace megakernel;

template <typename config, typename globals> struct attention_partial {
    static constexpr int opcode = OPCODE_ATTENTION_PARTIAL;
    static constexpr int NUM_STAGES = 3;
    static constexpr int GQA_RATIO =
        LLAMA_1B_NUM_ATTENTION_HEADS / LLAMA_1B_NUM_KV_HEADS;
    static constexpr int QOL_PAGE = 0;
    static constexpr int KV_PAGE = 1;

    static_assert(GQA_RATIO == 4, "GQA_RATIO must be 4.");
    static_assert(NUM_STAGES <= 4, "Modify page allocation for KVs.");

    using q_rt = kittens::rt_bf<16, LLAMA_1B_HEAD_DIM>; // only 4 rows are used
    using q_st = kittens::st_bf<16, LLAMA_1B_HEAD_DIM>; // only 4 rows are used
    using k_rt = kittens::rt_bf<LLAMA_1B_KV_BLOCK_SIZE, LLAMA_1B_HEAD_DIM>;
    using v_rt = kittens::rt_bf<LLAMA_1B_KV_BLOCK_SIZE, LLAMA_1B_HEAD_DIM, col_l>;
    using kv_st = kittens::st_bf<LLAMA_1B_KV_BLOCK_SIZE, LLAMA_1B_HEAD_DIM>;
    using attn_fl_rt =
        kittens::rt_fl<16, LLAMA_1B_KV_BLOCK_SIZE>; // only 4 values are used
    using attn_bf_rt =
        kittens::rt_bf<16, LLAMA_1B_KV_BLOCK_SIZE>; // only 4 values are used
    using max_vec_rv =
        col_vec<kittens::rt_fl<16, LLAMA_1B_HEAD_DIM>>; // only 4 values are used
    using max_vec_sv = kittens::sv_fl<16>;              // only 4 values are used
    using norm_vec_rv =
        col_vec<kittens::rt_fl<16, LLAMA_1B_HEAD_DIM>>; // only 4 values are used
    using norm_vec_sv = kittens::sv_fl<16>;             // only 4 values are used
    using l_rv =
        col_vec<kittens::rt_fl<16, LLAMA_1B_HEAD_DIM>>; // only 4 values are used
    using l_sv = kittens::sv_fl<16>;                    // only 4 values are used
    using o_rt = kittens::rt_fl<16, LLAMA_1B_HEAD_DIM>; // only 4 rows are used
    using o_sv = kittens::sv_fl<LLAMA_1B_HEAD_DIM>;
    using o_sv_bf = kittens::sv_bf<LLAMA_1B_HEAD_DIM>;

    // Python (A100/H100 decode_schedule) serializes DecodeAttentionSplitKV as:
    // [opcode, layer_idx, kv_head_idx, num_partials, partial_idx]
    // Indices:  0        1           2            3            4
    struct parsed_instruction {
        int layer_idx;
        int kv_head_idx;
        int num_partials;
        int partial_idx;
        __device__ inline parsed_instruction(
            typename config::instruction_t &instruction) {
            layer_idx = instruction[1];
            kv_head_idx = instruction[2];
            num_partials = instruction[3];
            partial_idx = instruction[4];
        }
        __device__ inline parsed_instruction(megakernel::state<config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    // Semaphore layout: one-shot O/L plus double-buffered K/V stage rings.
    __device__ static inline kittens::semaphore &O_arrived(megakernel::state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &L_arrived(megakernel::state<config> &s) {
        return s.semaphores()[1];
    }
    __device__ static inline kittens::semaphore &K_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[2 + stage * 2 + 1];
    }
    __device__ static inline kittens::semaphore &K_finished(megakernel::state<config> &s,
                                                   int stage) {
        return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline kittens::semaphore &V_finished(megakernel::state<config> &s,
                                                   int stage) {
        return s.semaphores()[2 + NUM_STAGES * 2 + stage * 2 + 1];
    }

    __device__ static inline void wait_QOL_page(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(QOL_PAGE));
    }
    __device__ static inline void wait_KV_page(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(KV_PAGE));
    }
    __device__ static inline void finish_QOL_page(megakernel::state<config> &s) {
        if (kittens::warp::laneid() == 0)
            s.finish_page(s.pid(QOL_PAGE), config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline void finish_KV_page(megakernel::state<config> &s) {
        if (kittens::warp::laneid() == 0)
            s.finish_page(s.pid(KV_PAGE), config::NUM_CONSUMER_WARPS);
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
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st) +
            sizeof(o_sv) * 4);
    }
    __device__ static inline kv_st &get_K_smem(megakernel::state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) +
            sizeof(kv_st) * (stage * 2));
    }
    __device__ static inline kv_st &get_V_smem(megakernel::state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) +
            sizeof(kv_st) * (1 + stage * 2));
    }

    template <ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void
    store_4_rows(SV (&dst)[4], const RT &src, int row4idx /*= 0, 1, 2, or 3*/) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == src.cols,
                      "dst length must match src cols.");

        using T2 = RT::dtype;
        using T = base_types::packing<T2>::unpacked_type;
        using U = SV::dtype;
        using U2 = base_types::packing<U>::packed_type;

        uint32_t dst_ptr[4];
        dst_ptr[0] =
            static_cast<uint32_t>(__cvta_generic_to_shared(&dst[0].data[0]));
        dst_ptr[1] =
            static_cast<uint32_t>(__cvta_generic_to_shared(&dst[1].data[0]));
        dst_ptr[2] =
            static_cast<uint32_t>(__cvta_generic_to_shared(&dst[2].data[0]));
        dst_ptr[3] =
            static_cast<uint32_t>(__cvta_generic_to_shared(&dst[3].data[0]));

        int laneid = kittens::warp::laneid();
        int local_row_idx = (laneid % 16) / 4;
        int local_col_idx = laneid % 4;

        if (row4idx % 2 == 0 && laneid < 16) { // rows 0~3 or 8~11
            if (row4idx / 2 == 0) {            // rows 0~3
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[2]); // Rows 0-3 use fragments 0 and 2.
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx,
                                  tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] +
                                      sizeof(U) * (col_idx + 8),
                                  tmp[1]);
                }
            } else { // rows 8~11
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx,
                                  tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] +
                                      sizeof(U) * (col_idx + 8),
                                  tmp[1]);
                }
            }
        } else if (row4idx % 2 == 1 && laneid >= 16) { // rows 4~7 or 12~15
            if (row4idx / 2 == 0) {                    // rows 4~7
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[2]); // Rows 4-7 use fragments 0 and 2.
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx,
                                  tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] +
                                      sizeof(U) * (col_idx + 8),
                                  tmp[1]);
                }
            } else { // rows 12~15
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;
                    move<U2>::sts(dst_ptr[local_row_idx] + sizeof(U) * col_idx,
                                  tmp[0]);
                    move<U2>::sts(dst_ptr[local_row_idx] +
                                      sizeof(U) * (col_idx + 8),
                                  tmp[1]);
                }
            }
        }
    }
    template <ducks::rt::row_layout RT>
    __device__ static inline void right_fill(
        RT &dst, const RT &src, const int col_idx,
        const typename base_types::packing<typename RT::dtype>::unpacked_type
            &val = 0) {
        if (col_idx >= dst.cols)
            return;
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int j = 0; j < dst.width; j++) {
#pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    const int col_idx_x = (j * dst.tile_size_col) +
                                          ((k / 2) * 8) +
                                          ((kittens::warp::laneid() % 4) * 2);
                    const int col_idx_y = (j * dst.tile_size_col) +
                                          ((k / 2) * 8) +
                                          ((kittens::warp::laneid() % 4) * 2) + 1;
                    if (col_idx_x >= col_idx) {
                        dst.tiles[i][j].data[k].x = val;
                    } else {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    if (col_idx_y >= col_idx) {
                        dst.tiles[i][j].data[k].y = val;
                    } else {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                }
            }
        }
    }

    __device__ static inline void
    load_Q_async(q_st &dst, const globals::activations_t &src,
                 const int q_head_start_idx /*0, 4, 8, ...*/) {
        static_assert(LLAMA_1B_HEAD_DIM == 64 && GQA_RATIO == 4,
                      "Fix this function.");
        using T = typename q_st::dtype;
        constexpr int elem_per_memcpy =
            sizeof(float4) / sizeof(typename q_st::dtype);                  // 8
        constexpr int memcpy_per_row = LLAMA_1B_HEAD_DIM / elem_per_memcpy; // 8

        typename globals::activations_t::dtype *src_ptr =
            &src.raw_ptr[q_head_start_idx * LLAMA_1B_HEAD_DIM];
        uint32_t dst_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(
            &dst.data[(q_head_start_idx % 16) * LLAMA_1B_HEAD_DIM]));

        int laneid = kittens::warp::laneid();
        int row = laneid / memcpy_per_row;
        int col = (laneid * elem_per_memcpy) % LLAMA_1B_HEAD_DIM;

        asm volatile(
            "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" ::"r"(
                dst.idx(dst_ptr, {row, col})),
            "l"(&src_ptr[row * LLAMA_1B_HEAD_DIM + col])
            : "memory");
        asm volatile("cp.async.commit_group;\n" ::: "memory");
    }

    struct controller {
        static_assert(
            attention_release_lids_are_permutations(config::NUM_PAGES),
            "attention release_lid formulas must permute physical pages");

        static __device__ int
        release_lid(const globals &g,
                    typename config::instruction_t &instruction, int &query) {
            return attention_partial_release_lid_impl(config::NUM_PAGES,
                                                      query);
        }
        static __device__ int init_semaphores(const globals &g,
                                              megakernel::state<config> &s) {
            init_semaphore(O_arrived(s), 0, 1);
            init_semaphore(L_arrived(s), 0, 1);
            for (int i = 0; i < NUM_STAGES; i++) {
                init_semaphore(K_arrived(s, i), 0, 1);
                init_semaphore(V_arrived(s, i), 0, 1);
                init_semaphore(K_finished(s, i), 0, 1);
                init_semaphore(V_finished(s, i), 0, 1);
            }
            return 2 + 4 * NUM_STAGES;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            auto laneid = kittens::warp::laneid();
            if (laneid >= 2 && laneid < config::NUM_PAGES) {
                int unused_page = s.pid(laneid);
                s.wait_page_ready(unused_page);
                s.finish_page(unused_page, config::NUM_CONSUMER_WARPS);
            }
        }
    };
    struct launcher {
        static __device__ void wait_for_kv(const globals &g, megakernel::state<config> &s,
                                           parsed_instruction &inst) {
            // Wait for the previous ops to finish (16 dims each, so 4 ops on
            // the same head)
            megakernel::mk_barrier_wait(
                (volatile int32_t*)&g.Bar[{ inst.layer_idx,
                                            BARRIER_ROW(OPCODE_RMS_QKV_GEMV_ROPE),
                                            LLAMA_1B_NUM_ATTENTION_HEADS + inst.kv_head_idx}],
                4);
            megakernel::mk_barrier_wait(
                (volatile int32_t*)&g.Bar[{inst.layer_idx,
                                           BARRIER_ROW(OPCODE_RMS_QKV_GEMV_ROPE),
                                           LLAMA_1B_NUM_ATTENTION_HEADS + LLAMA_1B_NUM_KV_HEADS + inst.kv_head_idx}],
                4);
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            if (kittens::warp::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
                s.wait_tensor_ready();
                arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
#endif

                // Setup
                parsed_instruction inst{s};
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + LLAMA_1B_KV_BLOCK_SIZE - 1) /
                                        LLAMA_1B_KV_BLOCK_SIZE;
                int blocks_per_partial =
                    (total_attn_blocks + inst.num_partials - 1) /
                    inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx =
                    min(start_blk_idx + blocks_per_partial, total_attn_blocks);

                // Wait for the KV page
                wait_KV_page(s);

                if (start_blk_idx >= end_blk_idx)
                    finish_KV_page(s);

                // Process the assigned KV blocks.
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    auto cur_blk_idx = start_blk_idx + i;
                    // Stage and phase are indexed by the instruction-local
                    // iteration, matching the consumer loop for every split.
                    int stage = i % NUM_STAGES;
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

                    if (cur_blk_idx == total_attn_blocks - 1) {
                        wait_for_kv(g, s, inst);
                    }

                    kittens::tma::expect(K_arrived(s, stage), K_smem);
                    kittens::tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        K_smem, g.k_cache,
                        {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0},
                        K_arrived(s, stage));
                    kittens::tma::expect(V_arrived(s, stage), V_smem);
                    kittens::tma::load_async<dim::DEPTH, cache_policy::EVICT_FIRST>(
                        V_smem, g.v_cache,
                        {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0},
                        V_arrived(s, stage));
                }
            }
        }
    };
    struct consumer {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {

            if (kittens::warpid() == 0) {
                parsed_instruction inst{s};
                int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;

                if (kittens::laneid() == 0) {
                    for (int head_offset = 0; head_offset < GQA_RATIO;
                         head_offset++) {
                        megakernel::mk_barrier_wait(
                            (volatile int32_t*)&g.Bar[{inst.layer_idx,
                                                       BARRIER_ROW(OPCODE_RMS_QKV_GEMV_ROPE),
                                                       q_head_start_idx + head_offset}],
                            4);
                    }
                }
                kittens::warp::sync();

                // Setup (Q placement: the 4 live rows sit at tile-row
                // offset q_head_start_idx % 16; keep load_Q_async, this index
                // and the storer's LSE index in lockstep -- they are one
                // coupled layout decision, not three independent constants).
                int q_head_local_idx =
                    (q_head_start_idx % q_rt::tile_size_row) / 4;
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + LLAMA_1B_KV_BLOCK_SIZE - 1) /
                                        LLAMA_1B_KV_BLOCK_SIZE;
                int blocks_per_partial =
                    (total_attn_blocks + inst.num_partials - 1) /
                    inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx =
                    min(start_blk_idx + blocks_per_partial, total_attn_blocks);
                float softmax_temp =
                    g.attn_scale * 1.44269504089f; // 1 / (sqrt(D_h) * ln(2))
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
                // NOTE: deliberately no zero(last_scaled_max_vec_reg) here -- the
                // register is first written at the bottom of iteration 0 and every
                // consumer sits inside the `i != 0` guard below (NaN fix).
                kittens::warp::zero(norm_vec_reg);
                kittens::warp::zero(O_reg);
                o_sv(&O_smem)[4] = get_O_smem(s);
                l_sv &L_smem = get_L_smem(s);

                // Initiate the load on Q
                wait_QOL_page(s);

                q_st &Q_smem = get_Q_smem(s);

                load_Q_async(Q_smem, g.q_post_rope, q_head_start_idx);

                // Wait for Q to arrive
                kittens::warp::load_async_wait();

                kittens::warp::load(Q_reg, Q_smem);

                // Process the assigned KV blocks.
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);

                    // Perform Q @ K.T
                    kittens::warp::zero(attn_fl_reg);
                    megakernel::mk_semaphore_wait(
                        K_arrived(s, stage),
                        (i / NUM_STAGES) % 2);

                    kittens::warp::load(K_reg, K_smem);
                    kittens::warp::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                    kittens::warp::sync();
                    kittens::warp::arrive(K_finished(s, stage));

                    // Mask out invalid positions at the end
                    if ((i + start_blk_idx + 1) * LLAMA_1B_KV_BLOCK_SIZE >
                        seq_len)
                        // NOTE: must be -INFINITY, not a large finite sentinel
                        // like -999999999999.f -- only -inf gives exactly zero
                        // weight for masked lanes after exp2.
                        right_fill(attn_fl_reg, attn_fl_reg,
                                   seq_len % LLAMA_1B_KV_BLOCK_SIZE,
                                   -INFINITY);

                    // Obtain maximums per row (which is per head)
                    kittens::warp::row_max(max_vec_reg, attn_fl_reg,
                                  max_vec_reg); // includes previous max

                    // Scale attention block and maximums by sqrt(D_h)
                    kittens::warp::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                    kittens::warp::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                    // Compute the softmax numerator.
                    kittens::warp::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                    kittens::warp::exp2(attn_fl_reg, attn_fl_reg);

                    // Online-softmax rescale to the new running max.
                    // Rescaling begins after the running maximum is initialized.
                    // This avoids reading an undefined diff register and avoids
                    // exp2(0 - scaled_max) overflow for negative first-block maxima.
                    // The copy at the bottom initializes last_scaled_max.
                    if (i != 0) {
                        kittens::warp::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg, scaled_max_vec_reg);
                        kittens::warp::exp2(diff_scaled_max_vec_reg, diff_scaled_max_vec_reg);
                        // Rescale numerator (O) and denominator (norm) together.
                        kittens::warp::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                        kittens::warp::mul(norm_vec_reg, norm_vec_reg, diff_scaled_max_vec_reg);
                    }

                    // Accumulate numerator (A @ V)
                    megakernel::mk_semaphore_wait(
                        V_arrived(s, stage),
                        (i / NUM_STAGES) % 2);

                    kittens::warp::load(V_reg, V_smem);
                    kittens::warp::copy(attn_bf_reg,
                               attn_fl_reg); // Convert to bf16 to do matmul
                    kittens::warp::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                    kittens::warp::sync();
                    kittens::warp::arrive(V_finished(s, stage));

                    // Accumulate denominator (the rescale of the running value
                    // already happened above, inside the i != 0 guard).
                    kittens::warp::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                    // Save for next iteration
                    kittens::warp::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
                }

                // Finish
                kittens::warp::sync();

                if (start_blk_idx < end_blk_idx) {
                    finish_KV_page(s);
                    kittens::warp::div_row(O_reg, O_reg, norm_vec_reg);
                    kittens::warp::log2(L_reg, norm_vec_reg);
                    kittens::warp::add(
                        L_reg, L_reg,
                        last_scaled_max_vec_reg); // now L_reg contains the LSE
                } else {
                    // Emit the neutral LSE partial when this split is empty.
                    kittens::warp::neg_infty(L_reg);
                }

                // Store the results
                store_4_rows(O_smem, O_reg, q_head_local_idx);
                kittens::warp::sync();

                kittens::warp::arrive(O_arrived(s));
                kittens::warp::store(L_smem, L_reg);
                kittens::warp::sync();
                kittens::warp::arrive(L_arrived(s));
            }
        }
    };
    struct storer {
        // Materialize O in the partial workspace for attention reduction.
        static inline __device__ void
        store_o_no_skip(const globals &g, megakernel::state<config> &s,
                        int q_head_start_idx, parsed_instruction &inst) {
            auto O_smem = get_O_smem(s);
            if (kittens::laneid() == 0) {
                megakernel::mk_semaphore_wait(O_arrived(s), 0);
                kittens::tma::store_async<cache_policy::EVICT_LAST>(
                    g.attn_out_intermediates, O_smem[0],
                    {0, q_head_start_idx + 0, inst.partial_idx, 0});
                kittens::tma::store_async<cache_policy::EVICT_LAST>(
                    g.attn_out_intermediates, O_smem[1],
                    {0, q_head_start_idx + 1, inst.partial_idx, 0});
                kittens::tma::store_async<cache_policy::EVICT_LAST>(
                    g.attn_out_intermediates, O_smem[2],
                    {0, q_head_start_idx + 2, inst.partial_idx, 0});
                kittens::tma::store_async<cache_policy::EVICT_LAST>(
                    g.attn_out_intermediates, O_smem[3],
                    {0, q_head_start_idx + 3, inst.partial_idx, 0});
            }
        }

        // Skip reduction (num_partials == 1): this partial covers the whole KV
        // range, so its normalized O already *is* the final attention output.
        // Convert O_smem fp32 -> bf16 in place and TMA it straight into
        // g.attn_out; no LSE, no attn_out_intermediates round trip, and no
        // AttentionReduction instruction is emitted for this layer at all
        // (scheduler.schedule_attention_reduction returns [] when P == 1).
        // Gated by the per-instruction field num_partials rather than any
        // global "skip reduction" flag, so scheduler and kernel can never
        // disagree -- they read the same instruction word.
        static inline __device__ void
        store_o_skip(const globals &g, megakernel::state<config> &s,
                     int q_head_start_idx) {
            auto O_smem = get_O_smem(s);
            if (kittens::laneid() == 0) {
                megakernel::mk_semaphore_wait(O_arrived(s), 0);
            }
            kittens::warp::sync(); // all lanes must see O before reading it

            kittens::rv_bf<LLAMA_1B_HEAD_DIM> O_bf;
            for (int h = 0; h < GQA_RATIO; ++h) {
                o_sv &smem_fl = O_smem[h];
                // sizeof(o_sv_bf) < sizeof(o_sv), so the bf16 view aliases the
                // front of the fp32 vector; the full read into registers
                // completes (warp::sync) before the narrowing write starts.
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
                        g.attn_out, smem_bf, {0, 0, 0, q_head_start_idx + h});
                }
            }
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            parsed_instruction inst{s};
            int laneid = kittens::warp::laneid();
            int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;
            // Single source of truth for the skip decision: the instruction
            // field the scheduler wrote. See schedule_attention_reduction().
            const bool skip_reduction = (inst.num_partials == 1);

            if (skip_reduction) {
                store_o_skip(g, s, q_head_start_idx);
            } else {
                // Materialize O and LSE in the partial workspace.
                store_o_no_skip(g, s, q_head_start_idx, inst);
                l_sv &L_smem = get_L_smem(s);
                if (laneid == 0) {
                    megakernel::mk_semaphore_wait(L_arrived(s), 0);
                    // L_smem holds one float per tile row; the 4 live LSEs sit
                    // at rows [q_head_start_idx % 16, +4) -- must match the
                    // consumer's q_head_local_idx (the same Q-placement
                    // decision as load_Q_async above).
                    for (int h = 0; h < GQA_RATIO; ++h) {
                        float lse = L_smem.data[(q_head_start_idx % 16) + h];
                        float *dst = (float *)&g.attn_lse_intermediates.raw_ptr[(q_head_start_idx + h) * g.attn_lse_intermediates.cols() + inst.partial_idx];
                        *dst = lse;
                    }
                }
            }

            kittens::warp::sync(); // ensure all writes are committed

            kittens::tma::store_async_wait();
            if (laneid == 0) {
                finish_QOL_page(s);
            }

            if (laneid == 0) {
                // Device-scope release fence: the O partials (TMA) and the LSE
                // (plain st.global above) must be visible to the consuming CTA
                // before its barrier counter is bumped.
                __threadfence();
                if (skip_reduction) {
                    // Publish the reduction barrier row when reduction is elided.
                    // Conservation: num_kv_heads(8) instructions x GQA_RATIO(4)
                    // = 32 = o_proj's EXPECTED_ARRIVAL_COUNT (num_attention_heads).
                    // Row 1 (this op's own row) then has no producer AND no
                    // consumer, which is consistent.
                    atomicAdd(&g.Bar[{inst.layer_idx,
                                      BARRIER_ROW(OPCODE_ATTENTION_REDUCTION), 0}],
                              GQA_RATIO);
                } else {
                    // Signal reduction: one arrival per Q head (reduction waits per head).
                    for (int h = 0; h < GQA_RATIO; ++h) {
                        atomicAdd(&g.Bar[{inst.layer_idx, BARRIER_ROW(opcode),
                                          q_head_start_idx + h}], 1);
                    }
                }
            }
        }
    };
};
