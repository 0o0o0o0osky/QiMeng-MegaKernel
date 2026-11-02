#pragma once

#include "instruction_list.cuh"

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

    float variance = full_sum / 2048.0f;
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::warp::mul(activations_vec, activations_vec, rms_scale);
    // Match the official framework reference, fused PyTorch reference, and
    // Python VM (PyVM) RMSNorm dtype boundary: normalization is
    // computed in FP32, then narrowed to the input dtype (BF16) before the
    // learned weight is applied.  Keep the following GEMV accumulator in FP32
    // by widening the rounded values back into the existing register vector.
    {
        kittens::rv_bf<sv_t::length> rounded_activations;
        kittens::warp::copy(rounded_activations, activations_vec);
        kittens::warp::copy(activations_vec, rounded_activations);
    }
    kittens::warp::load(rms_scale_vec, rms_scale_smem);
    kittens::warp::mul(activations_vec, activations_vec, rms_scale_vec);
    // BF16 * BF16 produces a BF16 RMSNorm output in each reference. Round once more
    // before widening for the fused FP32 GEMV path.
    {
        kittens::rv_bf<sv_t::length> rounded_activations;
        kittens::warp::copy(rounded_activations, activations_vec);
        kittens::warp::copy(activations_vec, rounded_activations);
    }

    return activations_vec;
}

#ifdef KITTENS_BLACKWELL
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t = kittens::rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename kittens::rt_fl<16, 16>::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;
    using sv_t = kittens::sv_bf<st_t::rows>;

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
    using sv_t = kittens::sv_bf<st_t::rows>;

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

template <typename Config, kittens::ducks::sv::all sv_t, typename rv_t,
          int SCRATCH_BYTES_PER_WARP>
__device__ static inline void matvec_reduce(uint8_t *scratch, rv_t &sum_vec) {
    rv_t part_vec;
    kittens::warp::zero(sum_vec);

#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {

        sv_t &part =
            *reinterpret_cast<sv_t *>(scratch + (i * SCRATCH_BYTES_PER_WARP));

        kittens::warp::load(part_vec, part);
        kittens::warp::add(sum_vec, sum_vec, part_vec);
    }
}

// Attention page-release orders.  Partial attention releases all pages except
// lids 1 and 0 in ascending order, then releases 1 before 0.  Reduction is a
// one-page left rotation.  Keeping these as formulas avoids per-controller
// lookup tables while preserving the controller query ABI.
constexpr int attention_partial_release_lid_impl(int num_pages, int query) {
    return query < num_pages - 2 ? query + 2 : num_pages - 1 - query;
}

constexpr int attention_reduction_release_lid_impl(int num_pages, int query) {
    return (query + 1) % num_pages;
}

constexpr bool attention_release_lids_are_permutations(int num_pages) {
    bool partial_seen[64] = {};
    bool reduction_seen[64] = {};
    if (num_pages < 2 || num_pages > 64)
        return false;
    for (int query = 0; query < num_pages; ++query) {
        const int partial =
            attention_partial_release_lid_impl(num_pages, query);
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

// Page-release order shared by the matvec pipelines.  Spare pages are
// released first, followed by the activation page and then the weight stages
// in the order in which the consumer finishes using them.
constexpr int mvp_release_lid_impl(int num_pages, int stage_pages, int stages,
                                   int iters, int query) {
    const int used_stages = iters < stages ? iters : stages;
    const int needed = 1 + used_stages * stage_pages;
    const int n_spare = num_pages - needed;
    int q = query;
    if (q < n_spare)
        return needed + q;
    q -= n_spare;
    if (q == 0)
        return 0;
    q -= 1;
    const int slot = q / stage_pages;
    const int off = q % stage_pages;
    const int first_stage = (iters >= stages) ? (iters % stages) : 0;
    const int stage = (first_stage + slot) % used_stages;
    return 1 + stage * stage_pages + off;
}

constexpr bool mvp_release_lid_is_permutation(int num_pages, int stage_pages,
                                              int stages) {
    for (int iters = 1; iters <= 3 * stages + 2; iters++) {
        bool seen[64] = {};
        for (int q = 0; q < num_pages; q++) {
            int lid =
                mvp_release_lid_impl(num_pages, stage_pages, stages, iters, q);
            if (lid < 0 || lid >= num_pages || seen[lid])
                return false;
            seen[lid] = true;
        }
    }
    return true;
}

// ========== matvec_pipeline (inlined from matvec_pipeline.cuh) ==========
template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics>
struct matvec_pipeline {
    static constexpr int INPUT_PIPELINE_STAGES = 3;
    static constexpr int OUTPUT_PIPELINE_STAGES = 3;
    static constexpr int STAGE_PAGES = 4;
    static constexpr int ACTIVATION_PAGE = 0;
    static constexpr int WEIGHTS_START_PAGE = 1;

    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    static constexpr int SEM_COUNT =
        1 + (INPUT_PIPELINE_STAGES + OUTPUT_PIPELINE_STAGES) * 2;

    static constexpr int SCRATCH_BYTES_PER_WARP = 16 * sizeof(float);
    static constexpr int SCRATCH_BYTES_PER_STAGE =
        SCRATCH_BYTES_PER_WARP * Config::NUM_CONSUMER_WARPS;
    static constexpr int USED_SCRATCH_BYTES =
        OUTPUT_PIPELINE_STAGES * SCRATCH_BYTES_PER_STAGE;
    static_assert(USED_SCRATCH_BYTES <= Config::SCRATCH_BYTES,
                  "USED_SCRATCH_BYTES must be less than SCRATCH_BYTES");

    __device__ static inline int get_activation_page(megakernel::state<Config> &s) {
        return s.pid(ACTIVATION_PAGE);
    }

    __device__ static inline int get_weight_page(megakernel::state<Config> &s, int stage,
                                                 int offset) {
        return s.pid(WEIGHTS_START_PAGE + stage * STAGE_PAGES + offset);
    }

    __device__ static inline kittens::semaphore &activations_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::semaphore &weights_arrived(megakernel::state<Config> &s,
                                                        int stage) {
        return s.semaphores()[1 + stage];
    }
    __device__ static inline kittens::semaphore &weights_finished(megakernel::state<Config> &s,
                                                         int stage) {
        return s.semaphores()[1 + INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_arrived(megakernel::state<Config> &s,
                                                        int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES + stage];
    }
    __device__ static inline kittens::semaphore &outputs_finished(megakernel::state<Config> &s,
                                                         int stage) {
        return s.semaphores()[1 + 2 * INPUT_PIPELINE_STAGES +
                              OUTPUT_PIPELINE_STAGES + stage];
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

    static_assert(mvp_release_lid_is_permutation(Config::NUM_PAGES, STAGE_PAGES,
                                                 INPUT_PIPELINE_STAGES),
                  "release_lid must be a permutation of the physical pages");
    static_assert(WEIGHTS_START_PAGE == 1 && ACTIVATION_PAGE == 0,
                  "mvp_release_lid_impl assumes activation lid 0 and weights "
                  "starting at lid 1");

    __device__ static inline int
    release_lid(const Globals &g, typename Config::instruction_t &instruction,
                int &query) {
        parsed_instruction inst{instruction};
        return mvp_release_lid_impl(Config::NUM_PAGES, STAGE_PAGES,
                                    INPUT_PIPELINE_STAGES, inst.iters, query);
    }

    __device__ static inline int init_semaphores(megakernel::state<Config> &s) {
        init_semaphore(activations_arrived(s), 1);
        for (int i = 0; i < INPUT_PIPELINE_STAGES; i++) {
            init_semaphore(weights_arrived(s, i), 1);
            init_semaphore(weights_finished(s, i), Config::NUM_CONSUMER_WARPS);
        }
        for (int i = 0; i < OUTPUT_PIPELINE_STAGES; i++) {
            init_semaphore(outputs_arrived(s, i), Config::NUM_CONSUMER_WARPS);
            init_semaphore(outputs_finished(s, i), 1);
        }
        return SEM_COUNT;
    }

    __device__ static inline void loader_loop(megakernel::state<Config> &s,
                                              const Globals &g) {
        parsed_instruction inst{s};

        auto needed_pages =
            1 + min(inst.iters, INPUT_PIPELINE_STAGES) * STAGE_PAGES;

        if (kittens::laneid() == 0) {

            int input_stage = 0;
            for (int iter = 0; iter < inst.iters; iter++) {
                kittens::wait(weights_finished(s, input_stage),
                     (iter % (2 * INPUT_PIPELINE_STAGES)) <
                         INPUT_PIPELINE_STAGES);

                auto &sem = weights_arrived(s, input_stage);
                kittens::tma::expect_bytes(sem, sizeof(kittens::bf16) * 2048 * 16);
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    int weight_page = get_weight_page(s, input_stage, i);
                    if (iter < INPUT_PIPELINE_STAGES) {
                        s.wait_page_ready(weight_page);
                    }
                    auto &weight_chunk = reinterpret_cast<kittens::st_bf<16, 512> &>(
                        s.pages[weight_page]);

                    pipeline_specifics::load_iter(s, g, inst, iter, i,
                                                  weight_chunk, sem);
                }

                input_stage = (input_stage + 1) % INPUT_PIPELINE_STAGES;
            }
        } else if (kittens::laneid() >= needed_pages && kittens::laneid() < Config::NUM_PAGES) {
            auto pid = s.pid(kittens::laneid());
            s.wait_page_ready(pid);
            s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
        }
    }

    template <typename rv_t>
    __device__ static inline void
    consumer_loop(megakernel::state<Config> &s, const Globals &g, rv_t &activations_vec) {
        parsed_instruction inst{s};

        static_assert(Config::NUM_CONSUMER_WARPS % STAGE_PAGES == 0,
                      "NUM_CONSUMER_WARPS must be divisible by STAGE_PAGES");
        constexpr int WARPS_PER_PAGE = Config::NUM_CONSUMER_WARPS / STAGE_PAGES;

        int page_index = kittens::warpid() / WARPS_PER_PAGE;

        int input_stage = 0, output_stage = 0;
        for (int i = 0; i < inst.iters; i++) {
            int weight_page = get_weight_page(s, input_stage, page_index);
            kittens::wait(weights_arrived(s, input_stage),
                 (i % (2 * INPUT_PIPELINE_STAGES)) >= INPUT_PIPELINE_STAGES);
            kittens::wait(outputs_finished(s, output_stage),
                 (i % (2 * OUTPUT_PIPELINE_STAGES)) < OUTPUT_PIPELINE_STAGES);
            kittens::st_bf<16, REDUCTION_DIM_PER_WARP> &weights =
                reinterpret_cast<kittens::st_bf<16, REDUCTION_DIM_PER_WARP> *>(
                    s.pages[weight_page].ptr())[kittens::warpid() % WARPS_PER_PAGE];

            kittens::sv_fl<16> &out_smem = *reinterpret_cast<kittens::sv_fl<16> *>(
                get_output_start(s, output_stage) +
                (kittens::warpid() * SCRATCH_BYTES_PER_WARP));

            matvec(out_smem, weights, activations_vec);

            kittens::warp::sync();
            kittens::warp::arrive(outputs_arrived(s, output_stage));
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

template <typename Config, typename Globals, typename parsed_instruction,
          typename pipeline_specifics, auto ActPtr, auto RmsPtr>
struct rms_matvec_pipeline
    : public matvec_pipeline<Config, Globals, parsed_instruction,
                             pipeline_specifics> {
    using pipeline = matvec_pipeline<Config, Globals, parsed_instruction,
                                     pipeline_specifics>;

    static constexpr int REDUCTION_DIM_PER_WARP =
        Globals::hidden_dim / Config::NUM_CONSUMER_WARPS;

    static constexpr int SEM_COUNT = 1 + pipeline::SEM_COUNT;

    __device__ static inline kittens::semaphore &rms_scale_arrived(megakernel::state<Config> &s) {
        return s.semaphores()[pipeline::SEM_COUNT];
    }

    __device__ static inline kittens::sv_bf<Globals::hidden_dim> &
    get_rms_scale(megakernel::state<Config> &s) {
        return *reinterpret_cast<kittens::sv_bf<Globals::hidden_dim> *>(
            s.pages[get_activation_page(s)].ptr(
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
            int activation_page = get_activation_page(s);
            s.wait_page_ready(activation_page);

            auto &rms_scale = get_rms_scale(s);
            auto &sem = rms_scale_arrived(s);

            kittens::tma::expect(sem, rms_scale);
            kittens::tma::load_async<kittens::cache_policy::EVICT_LAST>(rms_scale, g.*RmsPtr,
                                                      {layer_idx, 0}, sem);
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
            reinterpret_cast<sv_t *>(&get_activations(s))[kittens::warpid()];

        if (kittens::laneid() == 0 && kittens::warpid() == 0) {
            parsed_instruction inst{s};

            int activation_page = get_activation_page(s);

            s.wait_page_ready(activation_page);
            auto &activations = get_activations(s);

            auto &sem = activations_arrived(s);

            pipeline_specifics::gmem_wait(g, s);
        }
        kittens::group<Config::NUM_CONSUMER_WARPS>::sync(3);

        kittens::warp::load(activations_smem, g.*ActPtr, {kittens::warpid()});

        auto activation_page = get_activation_page(s);

        kittens::wait(rms_scale_arrived(s), 0);

        auto activations_vec = rms_norm<Config>(
            rms_scale_smem, activations_smem, g.rms_norm_eps,
            pipeline::get_output_start(s, pipeline::OUTPUT_PIPELINE_STAGES));

        kittens::warp::sync();
        s.warp_finish_page(activation_page, 1);

        pipeline::consumer_loop(s, g, activations_vec);
    }
};
