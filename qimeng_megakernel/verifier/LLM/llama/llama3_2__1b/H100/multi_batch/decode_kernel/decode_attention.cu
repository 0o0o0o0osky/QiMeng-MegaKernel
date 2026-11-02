#pragma once

#include "utils.cuh"

namespace llama_mb {

template <typename Config, typename Globals>
struct decode_attention {
    static constexpr int opcode = OPCODE_ATTENTION;
    using parsed_instruction = range_instruction<Config>;
    static constexpr int STAGES = 3;
    static constexpr int QO_LID = 0;
    static constexpr int KV_LID0 = 1;
    static constexpr int USED_PAGES = 1 + STAGES;

    using q_tile = kittens::st_bf<64, HEAD_DIM>;
    using kv_tile = kittens::st_bf<64, HEAD_DIM>;
    using o_tile = kittens::st_bf<64, HEAD_DIM>;

    static_assert(sizeof(q_tile) + sizeof(o_tile) == Config::PAGE_SIZE);
    static_assert(2 * sizeof(kv_tile) == Config::PAGE_SIZE);
    static_assert(USED_PAGES <= Config::NUM_PAGES);

    template <typename State>
    __device__ static kittens::semaphore &inputs_arrived(State &s,
                                                         int stage) {
        return s.semaphores()[stage];
    }
    template <typename State>
    __device__ static kittens::semaphore &inputs_finished(State &s,
                                                          int stage) {
        return s.semaphores()[STAGES + stage];
    }
    template <typename State>
    __device__ static kittens::semaphore &output_arrived(State &s) {
        return s.semaphores()[2 * STAGES];
    }
    template <typename State>
    __device__ static q_tile &q_smem(State &s) {
        return *reinterpret_cast<q_tile *>(s.pages[s.pid(QO_LID)].ptr());
    }
    template <typename State>
    __device__ static o_tile &o_smem(State &s) {
        return *reinterpret_cast<o_tile *>(
            static_cast<char *>(s.pages[s.pid(QO_LID)].ptr()) +
            sizeof(q_tile));
    }
    template <typename State>
    __device__ static kv_tile &k_smem(State &s, int stage) {
        return *reinterpret_cast<kv_tile *>(
            s.pages[s.pid(KV_LID0 + stage)].ptr());
    }
    template <typename State>
    __device__ static kv_tile &v_smem(State &s, int stage) {
        return *reinterpret_cast<kv_tile *>(
            static_cast<char *>(s.pages[s.pid(KV_LID0 + stage)].ptr()) +
            sizeof(kv_tile));
    }

    __device__ static int dependency_slot(const parsed_instruction &inst) {
        const int batch = inst.begin / NUM_KV_HEADS;
        const int kv_head = inst.begin % NUM_KV_HEADS;
        return (batch / 8) * NUM_KV_HEADS + kv_head;
    }

    struct controller {
        static __device__ int release_lid(
            const Globals &, typename Config::instruction_t &, int &query) {
            return query < 0 || query >= Config::NUM_PAGES ? 0 : query;
        }
        static __device__ int init_semaphores(
            const Globals &, megakernel::state<Config> &s) {
            for (int stage = 0; stage < STAGES; ++stage) {
                kittens::init_semaphore(inputs_arrived(s, stage), 1);
                kittens::init_semaphore(inputs_finished(s, stage),
                                        kittens::WARPGROUP_WARPS);
            }
            kittens::init_semaphore(output_arrived(s),
                                    kittens::WARPGROUP_WARPS);
            return 2 * STAGES + 1;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            const int item = inst.begin;
            const int batch = item / NUM_KV_HEADS;
            const int kv_head = item % NUM_KV_HEADS;
            const bool active = g.row_active(batch);
            const int seq_len = active ? g.position(batch) + 1 : 0;
            const int kv_tiles = (seq_len + 63) / 64;
            const int used_kv_pages = kv_tiles < STAGES ? kv_tiles : STAGES;
            const int lane = kittens::laneid();

            if (lane >= USED_PAGES && lane < Config::NUM_PAGES) {
                const int pid = s.pid(lane);
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            if (lane >= 1 && lane <= STAGES &&
                lane > used_kv_pages) {
                const int pid = s.pid(KV_LID0 + lane - 1);
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            if (lane != 0)
                return;

            for (int tile = 0; tile < kv_tiles; ++tile) {
                const int stage = tile % STAGES;
                kittens::wait(inputs_finished(s, stage),
                              (tile % (2 * STAGES)) < STAGES);
                if (tile < STAGES)
                    s.wait_page_ready(s.pid(KV_LID0 + stage));
                if (tile == kv_tiles - 1)
                    wait_business_dep(g, inst.dep_layer, inst.dep_row,
                                      inst.dep_expected,
                                      dependency_slot(inst));
                auto &arrived = inputs_arrived(s, stage);
                kittens::tma::expect_bytes(arrived, 2 * sizeof(kv_tile));
                const int sequence = batch * NUM_KV_HEADS + kv_head;
                kittens::tma::load_async(
                    k_smem(s, stage), g.k_cache_decode,
                    {inst.layer, sequence, tile, 0}, arrived);
                kittens::tma::load_async(
                    v_smem(s, stage), g.v_cache_decode,
                    {inst.layer, sequence, tile, 0}, arrived);
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &,
                                   megakernel::state<Config> &) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            if (kittens::warpgroup::groupid() != 0)
                return;

            parsed_instruction inst{s};
            const int item = inst.begin;
            const int batch = item / NUM_KV_HEADS;
            const int kv_head = item % NUM_KV_HEADS;
            const bool active = g.row_active(batch);
            const int seq_len = active ? g.position(batch) + 1 : 0;
            const int kv_tiles = (seq_len + 63) / 64;
            const int warp = kittens::warpgroup::warpid();
            const int lane = kittens::laneid();
            const int wg_thread = warp * kittens::WARP_THREADS + lane;

            if (warp == 0 && lane == 0) {
                s.wait_page_ready(s.pid(QO_LID));
                wait_business_dep(g, inst.dep_layer, inst.dep_row,
                                  inst.dep_expected,
                                  dependency_slot(inst));
            }
            kittens::warpgroup::sync(8);

            for (int idx = wg_thread; idx < 64 * HEAD_DIM;
                 idx += kittens::WARPGROUP_THREADS) {
                q_smem(s)[make_int2(idx / HEAD_DIM, idx % HEAD_DIM)] =
                    __float2bfloat16_rn(0.0f);
            }
            for (int idx = wg_thread; idx < GQA_RATIO * HEAD_DIM;
                 idx += kittens::WARPGROUP_THREADS) {
                const int q_in_group = idx / HEAD_DIM;
                const int d = idx % HEAD_DIM;
                const int q_head = kv_head * GQA_RATIO + q_in_group;
                q_smem(s)[make_int2(q_in_group, d)] =
                    active
                        ? g.q_post_rope.raw_ptr[(long)batch * HIDDEN_DIM +
                                                q_head * HEAD_DIM + d]
                        : __float2bfloat16_rn(0.0f);
            }
            kittens::warpgroup::sync(10);

            kittens::rt_fl<16, HEAD_DIM> output;
            kittens::rt_fl<16, 64> attention;
            kittens::rt_bf<16, 64> attention_bf;
            kittens::col_vec<kittens::rt_fl<16, 64>> row_maximum;
            kittens::col_vec<kittens::rt_fl<16, 64>> row_sum_exp;
            kittens::col_vec<kittens::rt_fl<16, 64>> previous_maximum;
            kittens::col_vec<kittens::rt_fl<16, 64>> correction;
            kittens::warp::zero(output);
            kittens::warp::neg_infty(row_maximum);
            kittens::warp::zero(row_sum_exp);

            for (int tile = 0; tile < kv_tiles; ++tile) {
                const int stage = tile % STAGES;
                kittens::wait(inputs_arrived(s, stage),
                              (tile % (2 * STAGES)) >= STAGES);
                kittens::warpgroup::mm_ABt(attention, q_smem(s),
                                            k_smem(s, stage));
                kittens::warpgroup::mma_async_wait();
                kittens::warp::mul(attention, attention,
                                   1.4426950408889634f * 0.125f);
                const int valid = seq_len - tile * 64;
                if (valid < 64)
                    kittens::warp::apply(
                        attention, attention,
                        [valid] __device__(int, int col, float value) {
                            return col < valid ? value : -INFINITY;
                        });

                kittens::warp::copy(previous_maximum, row_maximum);
                kittens::warp::row_max(row_maximum, attention, row_maximum);
                kittens::warp::sub_row(attention, attention, row_maximum);
                kittens::warp::exp2(attention, attention);
                kittens::warp::sub(correction, previous_maximum,
                                   row_maximum);
                kittens::warp::exp2(correction, correction);
                kittens::warp::mul(row_sum_exp, row_sum_exp, correction);
                kittens::warp::row_sum(row_sum_exp, attention, row_sum_exp);
                kittens::warp::mul_row(output, output, correction);
                kittens::warp::copy(attention_bf, attention);
                kittens::warpgroup::mma_AB(output, attention_bf,
                                            v_smem(s, stage));
                kittens::warpgroup::mma_async_wait();
                kittens::warp::arrive(inputs_finished(s, stage));
            }
            if (kv_tiles == 0)
                kittens::warp::zero(output);
            else
                kittens::warp::div_row(output, output, row_sum_exp);
            kittens::warpgroup::store(o_smem(s), output);
            kittens::warpgroup::sync(4);
            kittens::warp::arrive(output_arrived(s));

            if (warp == 0 && lane == 0) {
                const int used_kv_pages =
                    kv_tiles < STAGES ? kv_tiles : STAGES;
                for (int stage = 0; stage < used_kv_pages; ++stage)
                    s.finish_page(s.pid(KV_LID0 + stage),
                                  Config::NUM_CONSUMER_WARPS);
            }
        }
    };

    struct storer {
        static __device__ void run(const Globals &g,
                                   megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            const int item = inst.begin;
            const int batch = item / NUM_KV_HEADS;
            const int kv_head = item % NUM_KV_HEADS;
            const bool active = g.row_active(batch);

            kittens::wait(output_arrived(s), 0);
            for (int idx = kittens::laneid(); idx < GQA_RATIO * HEAD_DIM;
                 idx += kittens::WARP_THREADS) {
                const int q_in_group = idx / HEAD_DIM;
                const int d = idx % HEAD_DIM;
                const int q_head = kv_head * GQA_RATIO + q_in_group;
                const float value = static_cast<float>(
                    o_smem(s)[make_int2(q_in_group, d)]);
                g.attn_out.raw_ptr[(long)batch * HIDDEN_DIM +
                                   q_head * HEAD_DIM + d] =
                    __float2bfloat16_rn(active ? value : 0.0f);
            }
            kittens::warp::sync();
            if (kittens::laneid() == 0) {
                s.finish_page(s.pid(QO_LID),
                              Config::NUM_CONSUMER_WARPS);
                publish_business_dep(g, inst.layer, inst.publish_row,
                                     batch / 8, 1);
            }
        }
    };
};

} // namespace llama_mb
