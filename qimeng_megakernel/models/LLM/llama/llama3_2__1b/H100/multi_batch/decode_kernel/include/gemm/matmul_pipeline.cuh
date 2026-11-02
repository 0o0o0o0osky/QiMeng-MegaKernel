#pragma once

#include <array>

#include "kittens.cuh"

// Hopper GEMM family core.  Concrete model ops own coordinates, dependencies
// and epilogues; this file owns the VM-local stage rings and page map.
namespace qimeng_megakernel::hopper::gemm {

constexpr int ceil_div(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

// The concrete GEMM pipeline_specifics supplies:
//   STAGES, K_ITERS, OUTPUT_SLOTS, IN_FLIGHT_GROUPS
//   wg_active(inst, item, warpgroup)
//   static_stage_t, dependent_stage_t, output_stage_t, accumulator_t
//   tma_bytes / load_static / wait_dependency / load_dependent
//   issue / zero_accumulator / write_output / store / publish
// parsed_instruction:
//   begin, end, stride
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics>
struct matmul_pipeline {
    using static_stage_t = typename pipeline_specifics::static_stage_t;
    using dependent_stage_t = typename pipeline_specifics::dependent_stage_t;
    using output_stage_t = typename pipeline_specifics::output_stage_t;
    using accumulator_t = typename pipeline_specifics::accumulator_t;

    static constexpr int STAGES = pipeline_specifics::STAGES;
    static constexpr int K_ITERS = pipeline_specifics::K_ITERS;
    static constexpr int OUTPUT_SLOTS = pipeline_specifics::OUTPUT_SLOTS;
    static constexpr int IN_FLIGHT_GROUPS =
        pipeline_specifics::IN_FLIGHT_GROUPS;
    static constexpr int NUM_CONSUMER_WARPGROUPS =
        Config::NUM_CONSUMER_WARPS / 4;
    static constexpr int ACTIVE_CONSUMER_WARPGROUPS =
        pipeline_specifics::ACTIVE_CONSUMER_WARPGROUPS;
    static constexpr int ACTIVE_CONSUMER_WARPS =
        ACTIVE_CONSUMER_WARPGROUPS * kittens::WARPGROUP_WARPS;
    static constexpr int SEM_COUNT = 2 * STAGES + 2 * OUTPUT_SLOTS;

    static_assert(sizeof(static_stage_t) > 0 &&
                  sizeof(static_stage_t) <= Config::PAGE_SIZE);
    static_assert(sizeof(dependent_stage_t) > 0 &&
                  sizeof(dependent_stage_t) <= Config::PAGE_SIZE);
    static_assert(sizeof(output_stage_t) > 0 &&
                  sizeof(output_stage_t) <= Config::PAGE_SIZE);
    static constexpr int STATIC_STAGES_PER_PAGE =
        Config::PAGE_SIZE / sizeof(static_stage_t);
    static constexpr int DEPENDENT_STAGES_PER_PAGE =
        Config::PAGE_SIZE / sizeof(dependent_stage_t);
    static constexpr int OUTPUT_STAGES_PER_PAGE =
        Config::PAGE_SIZE / sizeof(output_stage_t);
    static constexpr int STATIC_PAGES =
        ceil_div(STAGES, STATIC_STAGES_PER_PAGE);
    static constexpr int DEPENDENT_PAGES =
        ceil_div(STAGES, DEPENDENT_STAGES_PER_PAGE);
    static constexpr int OUTPUT_PAGES =
        ceil_div(OUTPUT_SLOTS, OUTPUT_STAGES_PER_PAGE);
    static constexpr int STATIC_LID0 = 0;
    static constexpr int DEPENDENT_LID0 = STATIC_PAGES;
    static constexpr int OUTPUT_LID0 = STATIC_PAGES + DEPENDENT_PAGES;
    static constexpr int USED_PAGES =
        STATIC_PAGES + DEPENDENT_PAGES + OUTPUT_PAGES;
    static constexpr int IDLE_PAGES = Config::NUM_PAGES - USED_PAGES;
    static constexpr int INPUT_PAGES = STATIC_PAGES + DEPENDENT_PAGES;

    static_assert(USED_PAGES <= Config::NUM_PAGES,
                  "Hopper GEMM operands/output exceed the VM page budget");

    static_assert(Config::CLUSTER_BLOCKS == 1,
                  "the verified Hopper GEMM family is single-CTA");
    static_assert(Config::NUM_CONSUMER_WARPS == 4 ||
                      Config::NUM_CONSUMER_WARPS == 8,
                  "the GEMM family supports one or two consumer warpgroups");
    static_assert(ACTIVE_CONSUMER_WARPGROUPS == 1 ||
                      ACTIVE_CONSUMER_WARPGROUPS == 2,
                  "a GEMM uses one or two consumer warpgroups");
    static_assert(ACTIVE_CONSUMER_WARPGROUPS <=
                  NUM_CONSUMER_WARPGROUPS);
    static_assert(K_ITERS > 0);
    static_assert(K_ITERS >= STAGES,
                  "the verified GEMM family fills every input stage");
    static_assert(IN_FLIGHT_GROUPS == 1 || IN_FLIGHT_GROUPS == 2,
                  "only serial wait<0> and depth-two issue are supported");
    static_assert(SEM_COUNT <= Config::DYNAMIC_SEMAPHORES,
                  "GEMM stage/output rings exceed dynamic semaphore budget");
    static constexpr int static_lid(int stage) {
        return STATIC_LID0 + stage / STATIC_STAGES_PER_PAGE;
    }
    static constexpr int static_slot(int stage) {
        return stage % STATIC_STAGES_PER_PAGE;
    }
    static constexpr int dependent_lid(int stage) {
        return DEPENDENT_LID0 + stage / DEPENDENT_STAGES_PER_PAGE;
    }
    static constexpr int dependent_slot(int stage) {
        return stage % DEPENDENT_STAGES_PER_PAGE;
    }
    static constexpr int output_lid(int slot) {
        return OUTPUT_LID0 + slot / OUTPUT_STAGES_PER_PAGE;
    }
    static constexpr int output_slot(int slot) {
        return slot % OUTPUT_STAGES_PER_PAGE;
    }
    static constexpr int range_count(int begin, int end, int stride) {
        if (stride <= 0 || begin >= end)
            return 0;
        return (end - begin + stride - 1) / stride;
    }
    static constexpr int page_last_use(int total_iters, int stage_begin,
                                       int stage_end) {
        if (total_iters <= 0 || stage_begin < 0 || stage_begin >= stage_end ||
            stage_end > STAGES)
            return -1;
        const int full_cycles = total_iters / STAGES;
        const int remainder = total_iters % STAGES;
        if (remainder > stage_begin) {
            const int last_stage = remainder < stage_end ? remainder - 1
                                                         : stage_end - 1;
            return full_cycles * STAGES + last_stage;
        }
        if (full_cycles == 0)
            return -1;
        return (full_cycles - 1) * STAGES + stage_end - 1;
    }
    static constexpr int static_page_last_use(int lid, int total_iters) {
        if (lid < STATIC_LID0 || lid >= DEPENDENT_LID0)
            return -1;
        const int page = lid - STATIC_LID0;
        const int begin = page * STATIC_STAGES_PER_PAGE;
        const int end = begin + STATIC_STAGES_PER_PAGE < STAGES
                            ? begin + STATIC_STAGES_PER_PAGE
                            : STAGES;
        return page_last_use(total_iters, begin, end);
    }
    static constexpr int dependent_page_last_use(int lid, int total_iters) {
        if (lid < DEPENDENT_LID0 || lid >= OUTPUT_LID0)
            return -1;
        const int page = lid - DEPENDENT_LID0;
        const int begin = page * DEPENDENT_STAGES_PER_PAGE;
        const int end = begin + DEPENDENT_STAGES_PER_PAGE < STAGES
                            ? begin + DEPENDENT_STAGES_PER_PAGE
                            : STAGES;
        return page_last_use(total_iters, begin, end);
    }
    static constexpr int page_final_stage(int stage, int stages_per_page,
                                          int final_cycle_begin) {
        const int begin = (stage / stages_per_page) * stages_per_page;
        const int end = begin + stages_per_page < STAGES
                            ? begin + stages_per_page
                            : STAGES;
        return final_cycle_begin > begin
                   ? (final_cycle_begin < end ? final_cycle_begin : end) - 1
                   : end - 1;
    }
    static constexpr int input_release_lid(int rank, int total_iters) {
        if (rank < 0 || rank >= INPUT_PAGES || total_iters < STAGES)
            return 0;
        int seen = 0;
        const int first = total_iters - STAGES;
        for (int iter = first; iter < total_iters; ++iter) {
            const int stage = iter % STAGES;
            const int static_page = static_lid(stage);
            if (static_page_last_use(static_page, total_iters) == iter) {
                if (seen == rank)
                    return static_page;
                ++seen;
            }
            const int dependent_page = dependent_lid(stage);
            if (dependent_page_last_use(dependent_page, total_iters) == iter) {
                if (seen == rank)
                    return dependent_page;
                ++seen;
            }
        }
        return 0;
    }
    static constexpr int release_lid_impl(int query, int total_iters,
                                          int rotation = 0) {
        if (query < 0 || query >= Config::NUM_PAGES || rotation < 0 ||
            rotation >= Config::NUM_PAGES)
            return 0;
        if (total_iters < STAGES)
            return query;
        int q = (query + rotation) % Config::NUM_PAGES;
        if (q < IDLE_PAGES)
            return USED_PAGES + q;
        q -= IDLE_PAGES;
        if (q < INPUT_PAGES)
            return input_release_lid(q, total_iters);
        q -= INPUT_PAGES;
        return OUTPUT_LID0 + q;
    }
    template <int Tiles, int Rotation>
    static constexpr std::array<int, Config::NUM_PAGES> release_order() {
        std::array<int, Config::NUM_PAGES> order{};
        for (int query = 0; query < Config::NUM_PAGES; ++query)
            order[query] =
                release_lid_impl(query, Tiles * K_ITERS, Rotation);
        return order;
    }
    template <int Tiles, int Rotation, int Query = 0>
    __device__ static inline int fixed_release_lid(int query) {
        if constexpr (Query == Config::NUM_PAGES) {
            return 0;
        } else {
            if (query == Query)
                return release_lid_impl(Query, Tiles * K_ITERS, Rotation);
            return fixed_release_lid<Tiles, Rotation, Query + 1>(query);
        }
    }
    static constexpr bool release_lid_is_permutation() {
        for (int tiles = 1; tiles <= 8; ++tiles) {
            const int total_iters = tiles * K_ITERS;
            for (int rotation = 0; rotation < Config::NUM_PAGES; ++rotation) {
                bool seen[Config::NUM_PAGES] = {};
                for (int query = 0; query < Config::NUM_PAGES; ++query) {
                    const int lid =
                        release_lid_impl(query, total_iters, rotation);
                    if (lid < 0 || lid >= Config::NUM_PAGES || seen[lid])
                        return false;
                    seen[lid] = true;
                }
            }
        }
        return true;
    }
    static constexpr bool release_order_is_safe(int total_iters) {
        if (total_iters < STAGES)
            return false;
        int previous_last_use = -1;
        for (int rank = 0; rank < INPUT_PAGES; ++rank) {
            const int lid = input_release_lid(rank, total_iters);
            const int last_use = lid < DEPENDENT_LID0
                                     ? static_page_last_use(lid, total_iters)
                                     : dependent_page_last_use(lid, total_iters);
            if (last_use < previous_last_use)
                return false;
            previous_last_use = last_use;
        }
        return true;
    }
    static constexpr bool page_map_is_valid() {
        for (int stage = 0; stage < STAGES; ++stage) {
            if (static_lid(stage) < STATIC_LID0 ||
                static_lid(stage) >= DEPENDENT_LID0)
                return false;
            if (dependent_lid(stage) < DEPENDENT_LID0 ||
                dependent_lid(stage) >= OUTPUT_LID0)
                return false;
        }
        return true;
    }

    static_assert(release_lid_is_permutation(),
                  "release_lid must be a page permutation");
    static_assert(release_order_is_safe(K_ITERS),
                  "input pages must follow their final completed use");
    static_assert(page_map_is_valid(),
                  "operand stage map crosses a page-family boundary");

    template <typename State>
    __device__ static inline kittens::semaphore &inputs_arrived(State &s,
                                                                int stage) {
        return s.semaphores()[stage];
    }
    template <typename State>
    __device__ static inline kittens::semaphore &inputs_finished(State &s,
                                                                 int stage) {
        return s.semaphores()[STAGES + stage];
    }
    template <typename State>
    __device__ static inline kittens::semaphore &outputs_arrived(State &s,
                                                                 int slot) {
        return s.semaphores()[2 * STAGES + slot];
    }
    template <typename State>
    __device__ static inline kittens::semaphore &outputs_finished(State &s,
                                                                  int slot) {
        return s.semaphores()[2 * STAGES + OUTPUT_SLOTS + slot];
    }

    template <typename State>
    __device__ static inline static_stage_t &static_stage(State &s, int stage) {
        return reinterpret_cast<static_stage_t *>(
            s.pages[s.pid(static_lid(stage))].ptr())[static_slot(stage)];
    }
    template <typename State>
    __device__ static inline dependent_stage_t &dependent_stage(State &s,
                                                                 int stage) {
        return reinterpret_cast<dependent_stage_t *>(
            s.pages[s.pid(dependent_lid(stage))].ptr())
            [dependent_slot(stage)];
    }
    template <typename State>
    __device__ static inline output_stage_t &output_stage(State &s, int slot) {
        return reinterpret_cast<output_stage_t *>(
            s.pages[s.pid(output_lid(slot))].ptr())[output_slot(slot)];
    }

    template <int PagesInReserve = 0>
    __device__ static inline int
    release_lid(const Globals &, typename Config::instruction_t &instruction,
                int &query) {
        parsed_instruction inst{instruction};
        const int tiles = range_count(inst.begin, inst.end, inst.stride);
        constexpr int rotation = PagesInReserve % Config::NUM_PAGES;
        if (query < 0 || query >= Config::NUM_PAGES)
            return 0;
        switch (tiles) {
        case 0:
            return fixed_release_lid<0, rotation>(query);
        case 1:
            return fixed_release_lid<1, rotation>(query);
        case 2:
            return fixed_release_lid<2, rotation>(query);
        case 3:
            return fixed_release_lid<3, rotation>(query);
        case 4:
            return fixed_release_lid<4, rotation>(query);
        case 5:
            return fixed_release_lid<5, rotation>(query);
        case 6:
            return fixed_release_lid<6, rotation>(query);
        case 7:
            return fixed_release_lid<7, rotation>(query);
        case 8:
            return fixed_release_lid<8, rotation>(query);
        default:
            return release_lid_impl(query, tiles * K_ITERS, rotation);
        }
    }

    template <typename State>
    __device__ static inline int init_semaphores(State &s) {
        for (int stage = 0; stage < STAGES; ++stage) {
            kittens::init_semaphore(inputs_arrived(s, stage), 1);
            kittens::init_semaphore(inputs_finished(s, stage),
                                    ACTIVE_CONSUMER_WARPS);
        }
        for (int slot = 0; slot < OUTPUT_SLOTS; ++slot) {
            kittens::init_semaphore(outputs_arrived(s, slot),
                                    ACTIVE_CONSUMER_WARPS);
            kittens::init_semaphore(outputs_finished(s, slot), 2);
        }
        return SEM_COUNT;
    }

    template <typename State>
    __device__ static inline void loader_loop(State &s, const Globals &g) {
        parsed_instruction inst{s};
        const int tiles = range_count(inst.begin, inst.end, inst.stride);

        // A malformed or empty range consumes no page. Return the identity map
        // in logical-page order so the next instruction cannot be stranded.
        if (tiles == 0) {
            if (kittens::laneid() < Config::NUM_PAGES) {
                const int pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            return;
        }

        // Return every unused page exactly once by the full per-warp count.
        if (kittens::laneid() < IDLE_PAGES) {
            const int lid = USED_PAGES + kittens::laneid();
            const int pid = s.pid(lid);
            s.wait_page_ready(pid);
            s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
        }
        if (kittens::laneid() != 0)
            return;

        int linear_iter = 0;
        int stage = 0;
        bool inputs_finished_phase = true;
        bool dependency_waited = false;
        for (int item = inst.begin; item < inst.end;
             item += inst.stride) {
            for (int k_iter = 0; k_iter < K_ITERS;
                 ++k_iter, ++linear_iter) {
                megakernel::mk_semaphore_wait(inputs_finished(s, stage),
                                              inputs_finished_phase);

                if (linear_iter < STAGES) {
                    if (static_slot(stage) == 0)
                        s.wait_page_ready(s.pid(static_lid(stage)));
                    if (dependent_slot(stage) == 0)
                        s.wait_page_ready(s.pid(dependent_lid(stage)));
                }

                auto &arrived = inputs_arrived(s, stage);
                kittens::tma::expect_bytes(
                    arrived, pipeline_specifics::tma_bytes(inst, item));
                pipeline_specifics::load_static(
                    g, inst, item, k_iter, static_stage(s, stage), arrived);
                if (!dependency_waited) {
                    pipeline_specifics::wait_dependency(g, s, inst);
                    dependency_waited = true;
                }
                pipeline_specifics::load_dependent(
                    g, inst, item, k_iter, dependent_stage(s, stage), arrived);
                if (++stage == STAGES) {
                    stage = 0;
                    inputs_finished_phase = !inputs_finished_phase;
                }
            }
        }
        release_input_pages(s, tiles);
    }

    template <typename State>
    __device__ static inline void release_input_pages(State &s, int tiles) {
        const int total_iters = tiles * K_ITERS;
        const int final_cycle_begin = total_iters % STAGES;
        for (int completed = total_iters - STAGES; completed < total_iters;
             ++completed) {
            const int stage = completed % STAGES;
            const bool finish_static =
                stage == page_final_stage(stage, STATIC_STAGES_PER_PAGE,
                                          final_cycle_begin);
            const bool finish_dependent =
                stage == page_final_stage(stage, DEPENDENT_STAGES_PER_PAGE,
                                          final_cycle_begin);
            if (!finish_static && !finish_dependent)
                continue;
            megakernel::mk_semaphore_wait(
                inputs_finished(s, stage),
                (completed % (2 * STAGES)) >= STAGES);
            if (kittens::laneid() == 0) {
                if (finish_static)
                    s.finish_page(s.pid(static_lid(stage)),
                                  Config::NUM_CONSUMER_WARPS);
                if (finish_dependent)
                    s.finish_page(s.pid(dependent_lid(stage)),
                                  Config::NUM_CONSUMER_WARPS);
            }
        }
    }

    template <typename State>
    __device__ static inline void consumer_loop(State &s, const Globals &g) {
        const int wg = kittens::warpgroup::groupid();
        if (wg >= ACTIVE_CONSUMER_WARPGROUPS)
            return;
        parsed_instruction inst{s};
        const int tiles = range_count(inst.begin, inst.end, inst.stride);
        if (tiles == 0)
            return;
        int stage = 0;
        bool inputs_arrived_phase = false;
        int item_ordinal = 0;

        for (int item = inst.begin; item < inst.end;
             item += inst.stride, ++item_ordinal) {
            const int out_slot = item_ordinal % OUTPUT_SLOTS;
            if (item_ordinal < OUTPUT_SLOTS)
                s.wait_page_ready(s.pid(output_lid(out_slot)));
            megakernel::mk_semaphore_wait(
                outputs_finished(s, out_slot),
                (item_ordinal % (2 * OUTPUT_SLOTS)) < OUTPUT_SLOTS);

            accumulator_t accum;
            pipeline_specifics::zero_accumulator(accum);
            const bool active =
                pipeline_specifics::wg_active(inst, item, wg);
            int oldest_stage = stage;
            for (int k_iter = 0; k_iter < K_ITERS; ++k_iter) {
                megakernel::mk_semaphore_wait(inputs_arrived(s, stage),
                                              inputs_arrived_phase);
                if (active) {
                    pipeline_specifics::issue(
                        accum, static_stage(s, stage),
                        dependent_stage(s, stage), inst, item, k_iter, wg,
                        k_iter != 0);
                }

                if constexpr (IN_FLIGHT_GROUPS == 1) {
                    if (active)
                        kittens::warpgroup::mma_async_wait<0>();
                    kittens::warp::arrive(inputs_finished(s, stage));
                } else if (k_iter >= 2) {
                    if (active)
                        kittens::warpgroup::mma_async_wait<2>();
                    kittens::warp::arrive(inputs_finished(s, oldest_stage));
                    if (++oldest_stage == STAGES)
                        oldest_stage = 0;
                }
                if (++stage == STAGES) {
                    stage = 0;
                    inputs_arrived_phase = !inputs_arrived_phase;
                }
            }

            if constexpr (IN_FLIGHT_GROUPS == 2) {
                if (active)
                    kittens::warpgroup::mma_async_wait<1>();
                kittens::warp::arrive(inputs_finished(s, oldest_stage));
                if (++oldest_stage == STAGES)
                    oldest_stage = 0;
                if (active)
                    kittens::warpgroup::mma_async_wait<0>();
                kittens::warp::arrive(inputs_finished(s, oldest_stage));
            }

            pipeline_specifics::write_output(
                output_stage(s, out_slot), accum, inst, item, wg, active);
            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, out_slot));
        }

    }

    template <typename State>
    __device__ static inline void store_loop(State &s, const Globals &g,
                                             int worker) {
        parsed_instruction inst{s};
        const int tiles = range_count(inst.begin, inst.end, inst.stride);
        if (tiles == 0)
            return;
        int item_ordinal = 0;
        for (int item = inst.begin; item < inst.end;
            item += inst.stride, ++item_ordinal) {
            const int out_slot = item_ordinal % OUTPUT_SLOTS;
            megakernel::mk_semaphore_wait(
                outputs_arrived(s, out_slot),
                (item_ordinal % (2 * OUTPUT_SLOTS)) >= OUTPUT_SLOTS);
            pipeline_specifics::store(
                g, s, inst, item, output_stage(s, out_slot), worker);
            kittens::warp::sync();
            kittens::warp::arrive(outputs_finished(s, out_slot));
            megakernel::mk_semaphore_wait(
                outputs_finished(s, out_slot),
                (item_ordinal % (2 * OUTPUT_SLOTS)) >= OUTPUT_SLOTS);
            if (worker == 0 && kittens::laneid() == 0)
                pipeline_specifics::publish(g, inst, item);
        }

        if (worker == 0 && kittens::laneid() == 0) {
            for (int page = 0; page < OUTPUT_PAGES; ++page) {
                const int lid = OUTPUT_LID0 + page;
                s.finish_page(s.pid(lid), Config::NUM_CONSUMER_WARPS);
            }
        }
    }

    template <typename State>
    __device__ static inline void launcher_loop(State &s, const Globals &g) {
        store_loop(s, g, 1);
    }

    template <typename State>
    __device__ static inline void storer_loop(State &s, const Globals &g) {
        store_loop(s, g, 0);
    }
};

} // namespace qimeng_megakernel::hopper::gemm
