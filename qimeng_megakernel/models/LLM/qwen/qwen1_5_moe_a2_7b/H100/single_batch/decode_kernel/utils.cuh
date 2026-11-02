#pragma once

// Qwen1.5-MoE-A2.7B H100 matvec pipeline utilities.
//
// REDUCTION_DIM controls page geometry for both K=2048 dense projections and K=1408
// expert projections. The final K=1408 page is 384 columns, and 11 of 16 consumer
// warps contribute output partials. Page release order is derived from the active
// stage count. An optional loader_gmem_wait hook publishes indirect expert indices
// before routed-expert TMA loads.
//
// A 128B-swizzled st<bf16,16,C> requires C % 64 == 0. Both 512- and 384-column
// pages therefore split into layout-compatible 128-column sub-tiles.

#include "instruction_list.cuh"

#include <utility>

// ---------------------------------------------------------------------------
// rms_norm
// ---------------------------------------------------------------------------
template <typename Config, kittens::ducks::sv::all sv_t>
__device__ static inline auto
rms_norm(const sv_t &rms_scale_smem, const sv_t &activations_smem,
         float rms_norm_eps, void *scratch_memory) {
    using rv_t = kittens::rv_fl<sv_t::length>;
    rv_t activations_vec, sq_activations_vec, rms_scale_vec;
    kittens::warp::load(activations_vec, activations_smem);
    kittens::warp::copy(sq_activations_vec, activations_vec);
    kittens::warp::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    float partial_sum = kittens::warp::sum(sq_activations_vec);

    float *smem_rms_partial_sums = (float *)scratch_memory;
    if (kittens::laneid() == 0) {
        smem_rms_partial_sums[kittens::warpid()] = partial_sum;
    }
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);

    float full_sum = 0;
#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += smem_rms_partial_sums[i];
    }

    // Reduction length = NUM_CONSUMER_WARPS * per-warp slice (the llama version
    // hard-codes 2048.0f; identical for hidden_dim == 2048, correct in general).
    constexpr float REDUCTION_LENGTH =
        (float)(Config::NUM_CONSUMER_WARPS * sv_t::length);
    float variance = full_sum / REDUCTION_LENGTH;
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::warp::mul(activations_vec, activations_vec, rms_scale);
    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

// ---------------------------------------------------------------------------
// matvec: one warp, one st_bf<16, C> weight sub-tile x one rv_fl<C> activation
// ---------------------------------------------------------------------------
#ifdef KITTENS_BLACKWELL
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t = kittens::rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename kittens::rt_fl<16, 16>::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    kittens::rt_fl<16, 16> out_activations;
    kittens::warp::zero(out_activations);
    kittens::warp::mma_ABt(out_activations, weights, broadcast_activations,
                           out_activations);
    rcv_t sum_col_vec;
    kittens::warp::row_max(sum_col_vec, out_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#else
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t = kittens::rt_fl<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;

    rrv_t row_activations;
    kittens::warp::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::warp::broadcast_col(broadcast_activations, row_activations);
    kittens::warp::load(weights, weights_smem);
    kittens::warp::mul(broadcast_activations, broadcast_activations, weights);
    rcv_t sum_col_vec;
    kittens::warp::row_sum(sum_col_vec, broadcast_activations);

    rv_t sum_vec;
    kittens::warp::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    kittens::warp::sync();
}
#endif

// ---------------------------------------------------------------------------
// matvec_reduce: fold NUM_PARTIALS per-warp partial sums out of scratch.
//
// NUM_PARTIALS defaults to NUM_CONSUMER_WARPS (llama behaviour).  For a
// K < hidden_dim pipeline it MUST be pipeline::ACTIVE_CONSUMER_WARPS, or the
// stale scratch left by the inactive warps is summed in.
// ---------------------------------------------------------------------------
template <typename Config, kittens::ducks::sv::all sv_t, typename rv_t,
          int SCRATCH_BYTES_PER_WARP,
          int NUM_PARTIALS = Config::NUM_CONSUMER_WARPS>
__device__ static inline void matvec_reduce(uint8_t *scratch, rv_t &sum_vec) {
    static_assert(NUM_PARTIALS > 0 && NUM_PARTIALS <= Config::NUM_CONSUMER_WARPS,
                  "NUM_PARTIALS out of range");
    rv_t part_vec;
    kittens::warp::zero(sum_vec);

#pragma unroll
    for (int i = 0; i < NUM_PARTIALS; i++) {
        // NOTE: deliberately not using sizeof(sv_t) here; alignment issues.
        sv_t &part =
            *reinterpret_cast<sv_t *>(scratch + (i * SCRATCH_BYTES_PER_WARP));
        kittens::warp::load(part_vec, part);
        kittens::warp::add(sum_vec, sum_vec, part_vec);
    }
}

// ===========================================================================
// Attention page release order (free functions so the controller interfaces can
// prove the fixed orders at compile time without retaining hand-written tables).
//
// SplitKV releases the eleven loader-only pages first, then the KV page, then
// the Q/O/L page: {2, 3, ..., 12, 1, 0}. Reduction releases its twelve unused
// pages before its shared-data page: {1, 2, ..., 12, 0}.
// ===========================================================================
constexpr int qwen_moe_attention_partial_release_lid_impl(int num_pages,
                                                           int query) {
    if (num_pages < 2 || query < 0 || query >= num_pages)
        return 0;
    return query < num_pages - 2 ? query + 2 : num_pages - query - 1;
}

constexpr int attention_reduction_release_lid_impl(int num_pages, int query) {
    if (num_pages < 1 || query < 0 || query >= num_pages)
        return 0;
    return (query + 1) % num_pages;
}

constexpr bool qwen_moe_attention_release_lids_are_permutations(int num_pages) {
    if (num_pages < 2 || num_pages > 64)
        return false;
    bool partial_seen[64] = {};
    bool reduction_seen[64] = {};
    for (int query = 0; query < num_pages; ++query) {
        const int partial =
            qwen_moe_attention_partial_release_lid_impl(num_pages, query);
        const int reduction =
            attention_reduction_release_lid_impl(num_pages, query);
        if (partial < 0 || partial >= num_pages || partial_seen[partial] ||
            reduction < 0 || reduction >= num_pages ||
            reduction_seen[reduction])
            return false;
        partial_seen[partial] = true;
        reduction_seen[reduction] = true;
    }
    return true;
}

// ===========================================================================
// Matvec page release order (also a free function so that it can be
// constant-evaluated from inside the class body — a member constexpr function
// cannot be).
//
// `release_lid(prev_instruction, query)` answers: "which logical page of the
// *previous* instruction is the query-th one to be freed?"  The new
// instruction's logical page `query` is then mapped onto that physical page.
// Correctness only requires it to be a permutation of [0, NUM_PAGES); matching
// the true release order is what buys the cross-instruction overlap.
//
// Real release order in this pipeline:
//   1. spare pages [1 + used_stages*STAGE_PAGES, NUM_PAGES) — the loader frees
//      them immediately in its spare-lane branch
//   2. the activation page (lid 0) — the consumer frees it right after the
//      activation load / rms_norm
//   3. the weight stages, in the order the consumer's last INPUT_PIPELINE_STAGES
//      iterations touch them, i.e. starting at stage (iters % STAGES)
//
// For NUM_PAGES=13, STAGE_PAGES=4, STAGES=3 this reproduces all five of the
// llama version's hand-written ret_order[13] tables bit-for-bit.
// ===========================================================================
constexpr int mvp_release_lid_impl(int num_pages, int stage_pages, int stages,
                                   int iters, int query) {
    if (query < 0 || query >= num_pages)
        return 0;
    // Identity mapping defines the zero-work case.
    if (iters <= 0)
        return query;
    const int used_stages = iters < stages ? iters : stages;
    const int needed = 1 + used_stages * stage_pages;
    const int n_spare = num_pages - needed;
    int q = query;
    if (q < n_spare)
        return needed + q;
    q -= n_spare;
    if (q == 0)
        return 0; // ACTIVATION_PAGE
    q -= 1;
    const int slot = q / stage_pages;
    const int off = q % stage_pages;
    const int first_stage = (iters >= stages) ? (iters % stages) : 0;
    const int stage = (first_stage + slot) % used_stages;
    return 1 /* WEIGHTS_START_PAGE */ + stage * stage_pages + off;
}

// Compile-time proof that the map is a bijection on [0, num_pages) for every
// reachable `iters`.  A non-permutation silently drops a physical page and
// deadlocks the next instruction, with zero diagnostics.
constexpr bool mvp_release_lid_is_permutation(int num_pages, int stage_pages,
                                              int stages) {
    if (num_pages < 1 || num_pages > 64 || stage_pages < 1 || stages < 1 ||
        1 + stage_pages * stages > num_pages)
        return false;
    for (int iters = 1; iters <= 3 * stages + 2; iters++) {
        bool seen[64] = {};
        for (int q = 0; q < num_pages; q++) {
            int lid = mvp_release_lid_impl(num_pages, stage_pages, stages, iters, q);
            if (lid < 0 || lid >= num_pages)
                return false;
            if (seen[lid])
                return false;
            seen[lid] = true;
        }
    }
    return true;
}

// ===========================================================================
// matvec_pipeline
//
// One `iter` = one 16-row output tile.  The K dimension is spread across the
// consumer warps, REDUCTION_DIM_PER_WARP columns each.  Weights for one iter
// live in STAGE_PAGES pages of one input-pipeline stage.
//
//   REDUCTION_DIM = 2048:  STAGE_PAGES = 4  (512,512,512,512), 16/16 warps
//   REDUCTION_DIM = 1408:  STAGE_PAGES = 3  (512,512,384),     11/16 warps
//
// Page budget: 1 (activations) + INPUT_PIPELINE_STAGES * STAGE_PAGES
//   K=2048 -> 1 + 3*4 = 13 = NUM_PAGES (exactly full)
//   K=1408 -> 1 + 3*3 = 10, three pages spare (freed by the loader at once)
// ===========================================================================
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics,
          int REDUCTION_DIM_ = Globals::hidden_dim>
struct matvec_pipeline {
    static constexpr int INPUT_PIPELINE_STAGES = 3;
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    // ---- K-derived geometry -------------------------------------------------
    static constexpr int OUT_TILE_ROWS = 16;
    static constexpr int REDUCTION_DIM = REDUCTION_DIM_;
    // One full page holds a 16 x COLS_PER_PAGE bf16 tile == PAGE_SIZE bytes.
    static constexpr int COLS_PER_PAGE =
        Config::PAGE_SIZE / (OUT_TILE_ROWS * (int)sizeof(kittens::bf16));
    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;
    static constexpr int WARPS_PER_PAGE = COLS_PER_PAGE / REDUCTION_DIM_PER_WARP;
    static constexpr int STAGE_PAGES =
        (REDUCTION_DIM + COLS_PER_PAGE - 1) / COLS_PER_PAGE;
    static constexpr int LAST_PAGE_COLS =
        REDUCTION_DIM - (STAGE_PAGES - 1) * COLS_PER_PAGE;
    static constexpr int ACTIVE_CONSUMER_WARPS =
        REDUCTION_DIM / REDUCTION_DIM_PER_WARP;
    static constexpr int TOTAL_PAGES = 1 + INPUT_PIPELINE_STAGES * STAGE_PAGES;

    static_assert(OUT_TILE_ROWS * COLS_PER_PAGE * (int)sizeof(kittens::bf16) ==
                      Config::PAGE_SIZE,
                  "a full weight page must be exactly PAGE_SIZE");
    static_assert(REDUCTION_DIM % REDUCTION_DIM_PER_WARP == 0,
                  "K must be an integral number of per-warp column slices");
    static_assert(REDUCTION_DIM_PER_WARP % 64 == 0,
                  "sub-tile width must be a multiple of 64 for 128B-swizzle "
                  "compatible reinterpret_cast slicing");
    static_assert(COLS_PER_PAGE % REDUCTION_DIM_PER_WARP == 0,
                  "a full page must split evenly into per-warp sub-tiles");
    static_assert(LAST_PAGE_COLS % 64 == 0,
                  "narrow tail page must keep the 128B swizzle (C % 64 == 0)");
    static_assert(LAST_PAGE_COLS % REDUCTION_DIM_PER_WARP == 0,
                  "tail page must split evenly into per-warp sub-tiles");
    static_assert(ACTIVE_CONSUMER_WARPS <= Config::NUM_CONSUMER_WARPS,
                  "K/REDUCTION_DIM_PER_WARP exceeds the consumer warp count");
    static_assert(TOTAL_PAGES <= Config::NUM_PAGES,
                  "1 + INPUT_PIPELINE_STAGES*STAGE_PAGES exceeds NUM_PAGES");
    static_assert(INPUT_PIPELINE_STAGES == 3,
                  "release_lid / phase-bit expressions assume 3 input stages");

    // ---- warp -> (page, sub-tile) ------------------------------------------
    // Every page except the tail is exactly WARPS_PER_PAGE (= 4) warps wide,
    // and the tail page is a prefix of one, so the closed form below is exact
    // for both K = 2048 and K = 1408.  For K = 1408 it is, explicitly:
    //
    //   warp : 0  1  2  3 | 4  5  6  7 | 8  9 10 | 11..15
    //   page : 0  0  0  0 | 1  1  1  1 | 2  2  2 |  --
    //   sub  : 0  1  2  3 | 0  1  2  3 | 0  1  2 |  --
    //   cols : 0..511     | 512..1023  | 1024..1407
    //
    // (Note: `warpid / (NUM_CONSUMER_WARPS / STAGE_PAGES)` — the llama form —
    //  is 16/3 = 5 here and would be wrong; the divisor must come from the
    //  page geometry, not from the warp count.)
    static constexpr int warp_page_index(int w) { return w / WARPS_PER_PAGE; }
    static constexpr int warp_sub_index(int w) { return w % WARPS_PER_PAGE; }
    // (spelled out arithmetically rather than via the helpers above: a
    //  constexpr member function is not usable in a static_assert inside its
    //  own class body)
    static_assert((ACTIVE_CONSUMER_WARPS - 1) / WARPS_PER_PAGE == STAGE_PAGES - 1,
                  "warp->page map must reach exactly the last stage page");
    static_assert((ACTIVE_CONSUMER_WARPS - 1) % WARPS_PER_PAGE ==
                      LAST_PAGE_COLS / REDUCTION_DIM_PER_WARP - 1,
                  "warp->sub map must reach exactly the last tail sub-tile");

    // ---- semaphores ---------------------------------------------------------
    // 3 input stages x {arrived, finished} + 3 output stages x {arrived,
    // finished} = 12.  Every one of them has a named waiter:
    //   weights_arrived  -> consumer          weights_finished -> loader
    //   outputs_arrived  -> storer            outputs_finished -> consumer
    static constexpr int SEM_COUNT =
        (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static constexpr int SCRATCH_BYTES_PER_WARP = 16 * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE =
        SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;
    static constexpr int USED_SCRATCH_BYTES =
        OUTPUT_PIPELINE_STAGES * SCRATCH_BYTES_PER_STAGE;
    // +1 stage worth of slack is used by rms_norm as its cross-warp scratch.
    static_assert(USED_SCRATCH_BYTES + Config::NUM_CONSUMER_WARPS * 4 <=
                      Config::SCRATCH_BYTES,
                  "USED_SCRATCH_BYTES (+ rms scratch) must fit in SCRATCH_BYTES");

    // ---- page accessors -----------------------------------------------------
    __device__ static inline int get_activation_page(megakernel::state<Config> &s) {
        return s.pid(ACTIVATION_PAGE);
    }
    __device__ static inline int get_weight_page(megakernel::state<Config> &s,
                                                 int stage, int offset) {
        return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset);
    }

    // The shared tile type of stage page I (the tail page may be narrower).
    template <int I>
    using weight_page_t =
        kittens::st_bf<OUT_TILE_ROWS,
                       (I == STAGE_PAGES - 1 ? LAST_PAGE_COLS : COLS_PER_PAGE)>;

    __device__ static inline kittens::semaphore &
    weights_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[stage];
    }
    __device__ static inline kittens::semaphore &
    weights_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &
    outputs_arrived(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[2 * INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &
    outputs_finished(megakernel::state<Config> &s, int stage) {
        return s.semaphores()[2 * INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES +
                              stage];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_activations(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[get_activation_page(s)].ptr());
    }

    __device__ static inline uint8_t *get_output_start(megakernel::state<Config> &s,
                                                       int stage) {
        return (uint8_t *)s.scratch() + (stage * SCRATCH_BYTES_PER_STAGE);
    }

    // ---- page release order (see mvp_release_lid_impl above) ----------------
    static_assert(mvp_release_lid_is_permutation(Config::NUM_PAGES, STAGE_PAGES,
                                                 INPUT_PIPELINE_STAGES),
                  "release_lid must be a permutation of the physical pages");
    static_assert(WEIGHTS_START_PAGE == 1 && ACTIVATION_PAGE == 0,
                  "mvp_release_lid_impl hard-codes lid 0 = activations, "
                  "weights starting at lid 1");

    __device__ static inline int
    release_lid(const Globals &g, typename Config::instruction_t &instruction,
                int &query) {
        parsed_instruction inst{instruction};
        return mvp_release_lid_impl(Config::NUM_PAGES, STAGE_PAGES,
                                    INPUT_PIPELINE_STAGES, inst.iters, query);
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(weights_arrived(s, i), 1);
            init_semaphore(weights_finished(s, i), Config::NUM_CONSUMER_WARPS);
        }
        for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
            // only the ACTIVE warps produce partial sums
            init_semaphore(outputs_arrived(s, i), ACTIVE_CONSUMER_WARPS);
            init_semaphore(outputs_finished(s, i), 1);
        }
        return SEM_COUNT;
    }

    // ---- loader -------------------------------------------------------------
    template <int I>
    __device__ static inline void
    load_one_page(megakernel::state<Config> &s, const Globals &g,
                  parsed_instruction &inst, int iter, int input_stage,
                  kittens::semaphore &sem) {
        int weight_page = get_weight_page(s, input_stage, I);
        if (iter < INPUT_PIPELINE_STAGES) {
            s.wait_page_ready(weight_page);
        }
        auto &weight_chunk =
            *reinterpret_cast<weight_page_t<I> *>(s.pages[weight_page].ptr());
        // col_idx is in units of COLS_PER_PAGE; the op computes the byte/element
        // offset as inst.start_reduction_col + COLS_PER_PAGE * col_idx.
        pipeline_specifics::load_iter(s, g, inst, iter, I, weight_chunk, sem);
    }

    template <int... Is>
    __device__ static inline void
    load_stage(std::integer_sequence<int, Is...>, megakernel::state<Config> &s,
               const Globals &g, parsed_instruction &inst, int iter,
               int input_stage, kittens::semaphore &sem) {
        (load_one_page<Is>(s, g, inst, iter, input_stage, sem), ...);
    }

    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g) {
        parsed_instruction inst{s};

        auto needed_pages =
            1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

        if (kittens::laneid() == 0) {
            // Optional pre-loop hook: ops whose weight *address* depends on an
            // upstream GMEM barrier (MoE routed experts) block here.  Default
            // is a no-op for ops that do not define it.
            if constexpr (requires {
                              pipeline_specifics::loader_gmem_wait(s, g, inst);
                          }) {
                pipeline_specifics::loader_gmem_wait(s, g, inst);
            }

            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                kittens::wait(weights_finished(s, input_stage),
                              (iter % (2 * INPUT_PIPELINE_STAGES)) <
                                  INPUT_PIPELINE_STAGES);

                auto &sem = weights_arrived(s, input_stage);
                // tma::expect is a SET, not an accumulate: one call for the
                // whole stage, covering all STAGE_PAGES loads.
                kittens::tma::expect_bytes(
                    sem, sizeof(kittens::bf16) * REDUCTION_DIM * OUT_TILE_ROWS);

                load_stage(std::make_integer_sequence<int, STAGE_PAGES>{}, s, g,
                           inst, iter, input_stage, sem);

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        } else if (kittens::laneid() >= needed_pages &&
                   kittens::laneid() < Config::NUM_PAGES) {
            auto pid = s.pid(kittens::laneid());
            s.wait_page_ready(pid);
            s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
        }
    }

    // ---- consumer -----------------------------------------------------------
    template <typename rv_t>
    __device__ static inline void consumer_loop(megakernel::state<Config> &s,
                                                const Globals &g,
                                                rv_t &activations_vec) {
        parsed_instruction inst{s};

        const int warpid = kittens::warpid();
        const bool active = warpid < ACTIVE_CONSUMER_WARPS;
        const int page_index = active ? warp_page_index(warpid) : 0;
        const int sub_index = warp_sub_index(warpid);

        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            // Every warp — active or not — waits here.  For the inactive warps
            // this is purely a pacing device: without it they would race ahead
            // and over-arrive on weights_finished, corrupting its phase.
            kittens::wait(weights_arrived(s, input_stage),
                          (i % (2 * INPUT_PIPELINE_STAGES)) >=
                              INPUT_PIPELINE_STAGES);

            if (active) {
                kittens::wait(outputs_finished(s, output_stage),
                              (i % (2 * OUTPUT_PIPELINE_STAGES)) <
                                  OUTPUT_PIPELINE_STAGES);

                int weight_page = get_weight_page(s, input_stage, page_index);
                kittens::st_bf<OUT_TILE_ROWS, REDUCTION_DIM_PER_WARP> &weights =
                    reinterpret_cast<
                        kittens::st_bf<OUT_TILE_ROWS, REDUCTION_DIM_PER_WARP> *>(
                        s.pages[weight_page].ptr())[sub_index];

                kittens::sv_fl<OUT_TILE_ROWS> &out_smem =
                    *reinterpret_cast<kittens::sv_fl<OUT_TILE_ROWS> *>(
                        get_output_start(s, output_stage) +
                        (warpid * SCRATCH_BYTES_PER_WARP));

                matvec(out_smem, weights, activations_vec);

                kittens::warp::sync();
                kittens::warp::arrive(outputs_arrived(s, output_stage));
            }

            // Must be after warp::sync(): Volta+ ITS does not reconverge
            // automatically, and lane 0 releasing the page before the other
            // lanes are done reading it lets the next instruction overwrite
            // smem underneath them.
            kittens::warp::sync();
            kittens::warp::arrive(weights_finished(s, input_stage));

            if (i >= inst.iters - INPUT_PIPELINE_STAGES) {
#pragma unroll
                for (int j = 0; j < STAGE_PAGES; j++) {
                    s.warp_finish_page(get_weight_page(s, input_stage, j), 1);
                }
            }

            input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }

    // ---- storer -------------------------------------------------------------
    template <int iter_scale = 1>
    __device__ static inline void storer_loop(megakernel::state<Config> &s,
                                              const Globals &g) {
        parsed_instruction inst{s};

        int output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            auto &sem = outputs_arrived(s, output_stage);
            auto bit =
                (i % (2 * OUTPUT_PIPELINE_STAGES)) >= OUTPUT_PIPELINE_STAGES;

            kittens::wait(sem, bit);

            pipeline_specifics::store(s, g, inst, i, output_stage);

            if ((i + 1) % iter_scale == 0) {
                for (int j = 0; j < iter_scale; j++) {
                    auto stage_to_arrive = (i - j) % OUTPUT_PIPELINE_STAGES;
                    kittens::warp::arrive(outputs_finished(s, stage_to_arrive));
                }
            }
            output_stage = (output_stage + 1) % OUTPUT_PIPELINE_STAGES;
        }
    }
};

// ===========================================================================
// rms_matvec_pipeline — matvec_pipeline preceded by an in-page RMSNorm.
// The RMSNorm always spans hidden_dim across all NUM_CONSUMER_WARPS, so this
// variant is only meaningful for REDUCTION_DIM == hidden_dim.
// ===========================================================================
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics, auto ActPtr, auto RmsPtr>
struct rms_matvec_pipeline
    : public matvec_pipeline<Config, Globals, parsed_instruction,
                             pipeline_specifics, Globals::hidden_dim> {
    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics, Globals::hidden_dim>;

    static constexpr int REDUCTION_DIM_PER_WARP =
        pipeline::REDUCTION_DIM_PER_WARP;

    static constexpr int SEM_COUNT = 1 + pipeline::SEM_COUNT;

    __device__ static inline kittens::semaphore &
    rms_scale_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_rms_scale(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[pipeline::get_activation_page(s)].ptr(
                sizeof(kittens::sv_bf<Globals::hidden_dim>)));
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        pipeline::init_semaphores(s);
        init_semaphore(rms_scale_arrived(s), 1);
        return SEM_COUNT;
    }

    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g, int layer_idx) {
        if (kittens::laneid() == 0) {
            int activation_page = pipeline::get_activation_page(s);
            s.wait_page_ready(activation_page);

            auto &rms_scale = get_rms_scale(s);
            auto &sem = rms_scale_arrived(s);

            kittens::tma::expect(sem, rms_scale);
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(
                rms_scale, g.*RmsPtr, {layer_idx, 0}, sem);
        }

        pipeline::loader_loop(s, g);
    }

    __device__ static inline void launcher_loop(megakernel::state<Config> &s,
                                                const Globals &g) {
        if (kittens::laneid() == 0) {
#ifdef KITTENS_BLACKWELL
            s.wait_tensor_ready();
            arrive(s.tensor_finished, Config::NUM_CONSUMER_WARPS);
#endif
        }
    }

    __device__ static inline void consumer_loop(megakernel::state<Config> &s,
                                                const Globals &g) {
        using sv_t = kittens::sv_bf<REDUCTION_DIM_PER_WARP>;
        auto &rms_scale_smem =
            reinterpret_cast<sv_t *>(&get_rms_scale(s))[kittens::warpid()];
        auto &activations_smem =
            reinterpret_cast<sv_t *>(&pipeline::get_activations(s))[kittens::warpid()];

        if (kittens::laneid() == 0 && kittens::warpid() == 0) {
            int activation_page = pipeline::get_activation_page(s);
            s.wait_page_ready(activation_page);
            pipeline_specifics::gmem_wait(g, s);
        }
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

        kittens::warp::load(activations_smem, g.*ActPtr, {kittens::warpid()});

        auto activation_page = pipeline::get_activation_page(s);

        kittens::wait(rms_scale_arrived(s), 0);

        auto activations_vec = rms_norm<Config>(
            rms_scale_smem, activations_smem, g.rms_norm_eps,
            pipeline::get_output_start(s, pipeline::OUTPUT_PIPELINE_STAGES));

        kittens::warp::sync();
        s.warp_finish_page(activation_page, 1);

        pipeline::consumer_loop(s, g, activations_vec);
    }
};
