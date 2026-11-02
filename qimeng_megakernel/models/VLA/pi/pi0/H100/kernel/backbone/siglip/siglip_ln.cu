#pragma once

#include "kittens.cuh"
#include "common/globals.cuh"
#include "common/instruction_list.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"

namespace pi0 {

static constexpr float SIGLIP_LN_EPS = 1.0e-5f;

template <typename Config, typename Globals, int Opcode, int BarrierRow>
struct siglip_ln {
    static constexpr int opcode = Opcode;

    using x_vec = kittens::sv_bf<vis::WIDTH>;  // 1152 * 2B = 2304 B

    static constexpr int TOKENS_PER_INST = Config::NUM_CONSUMER_WARPS;
    static_assert(TOKENS_PER_INST * sizeof(x_vec) <= Config::PAGE_SIZE,
                  "The tokens processed by one fusion instruction must fit in one page");

    struct parsed_instruction {
        int opcode;
        int layer;
        int token_row;
        int weight_row;
        int dep_layer, dep_row, dep_expected;
        int num_tokens;
        int flags;

        __device__ inline parsed_instruction(typename Config::instruction_t &inst) {
            opcode       = inst[0];
            flags        = inst[11];
            layer        = inst[1];
            token_row    = inst[2];
            weight_row   = inst[3];
            dep_layer    = inst[4];
            dep_row      = inst[5];
            dep_expected = inst[6];
            num_tokens   = inst[7];
        }
        __device__ inline parsed_instruction(::megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline kittens::semaphore &x_arrived(::megakernel::state<Config> &s)     { return s.semaphores()[0]; }
    __device__ static inline kittens::semaphore &gamma_arrived(::megakernel::state<Config> &s) { return s.semaphores()[1]; }
    __device__ static inline kittens::semaphore &beta_arrived(::megakernel::state<Config> &s)  { return s.semaphores()[2]; }
    __device__ static inline kittens::semaphore &output_ready(::megakernel::state<Config> &s)  { return s.semaphores()[3]; }

    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &instruction, int &query) {

            return (query + Config::NUM_PAGES - 3) % Config::NUM_PAGES;
        }

        static __device__ int init_semaphores(const Globals &g, ::megakernel::state<Config> &s) {

            for (int i = 0; i < 4; i++) {
                if (i == 3) {
                    kittens::init_semaphore(s.semaphores()[i], Config::NUM_CONSUMER_WARPS);
                } else {
                    kittens::init_semaphore(s.semaphores()[i], 1);
                }
            }
            return 4;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};

            if (kittens::laneid() == 0) {

                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

                int x_pid = s.pid(0);
                s.wait_page_ready(x_pid);
                x_vec *x_smem = reinterpret_cast<x_vec *>(s.pages[x_pid].data);

                kittens::tma::expect_bytes(x_arrived(s), inst.num_tokens * sizeof(x_vec));
                for (int i = 0; i < inst.num_tokens; i++)
                    kittens::tma::load_async(x_smem[i], g.vision_x,
                                             {inst.token_row + i, 0}, x_arrived(s));

                int g_pid = s.pid(1);
                s.wait_page_ready(g_pid);
                x_vec &gamma_smem = *reinterpret_cast<x_vec *>(s.pages[g_pid].data);
                kittens::tma::expect(gamma_arrived(s), gamma_smem);
                kittens::tma::load_async(gamma_smem, g.ln_gamma, {inst.weight_row, 0}, gamma_arrived(s));

                int b_pid = s.pid(2);
                s.wait_page_ready(b_pid);
                x_vec &beta_smem = *reinterpret_cast<x_vec *>(s.pages[b_pid].data);
                kittens::tma::expect(beta_arrived(s), beta_smem);
                kittens::tma::load_async(beta_smem, g.ln_beta, {inst.weight_row, 0}, beta_arrived(s));

            } else if (kittens::laneid() >= 3 && kittens::laneid() < Config::NUM_PAGES) {
                int pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            kittens::warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {

        }
    };

    struct consumer {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(x_arrived(s), 0);
            ::megakernel::mk_semaphore_wait(gamma_arrived(s), 0);
            ::megakernel::mk_semaphore_wait(beta_arrived(s), 0);

            const int w = kittens::warpid();          // 0..NUM_CONSUMER_WARPS-1
            if (w >= inst.num_tokens) {
                kittens::warp::arrive(output_ready(s));
                return;
            }

            kittens::bf16 *x_smem =
                reinterpret_cast<kittens::bf16 *>(s.pages[s.pid(0)].data) + w * vis::WIDTH;
            kittens::bf16 *gamma_smem = reinterpret_cast<kittens::bf16 *>(s.pages[s.pid(1)].data);
            kittens::bf16 *beta_smem  = reinterpret_cast<kittens::bf16 *>(s.pages[s.pid(2)].data);

            const int lane = kittens::laneid();

            float sum = 0.0f;
            for (int c = lane; c < vis::WIDTH; c += 32) sum += __bfloat162float(x_smem[c]);
            for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
            const float mean = sum / float(vis::WIDTH);

            float sq = 0.0f;
            for (int c = lane; c < vis::WIDTH; c += 32) {
                float dv = __bfloat162float(x_smem[c]) - mean;
                sq += dv * dv;
            }
            for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
            const float inv_std = rsqrtf(sq / float(vis::WIDTH) + SIGLIP_LN_EPS);

            for (int c = lane; c < vis::WIDTH; c += 32) {
                float xn = (__bfloat162float(x_smem[c]) - mean) * inv_std;
                xn = xn * __bfloat162float(gamma_smem[c]) + __bfloat162float(beta_smem[c]);
                x_smem[c] = __float2bfloat16(xn);
            }
            __syncwarp();
            kittens::warp::arrive(output_ready(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(output_ready(s), 0);

            if (kittens::laneid() == 0) {

                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

                int x_pid = s.pid(0);
                x_vec *out_smem = reinterpret_cast<x_vec *>(s.pages[x_pid].data);
                for (int i = 0; i < inst.num_tokens; i++)
                    kittens::tma::store_async(g.siglip_x_normed, out_smem[i],
                                              {inst.token_row + i, 0});

                kittens::tma::store_async_read_wait();

                s.finish_page(x_pid,    Config::NUM_CONSUMER_WARPS);
                s.finish_page(s.pid(1), Config::NUM_CONSUMER_WARPS);
                s.finish_page(s.pid(2), Config::NUM_CONSUMER_WARPS);
            }

            kittens::tma::store_async_wait();
            kittens::warp::sync();

            if (kittens::laneid() == 0) {
                mk_phase_release((uint32_t *)&g.Bar[kittens::coord<>{inst.layer, BarrierRow, 0}],
                                 inst.flags & 8);
            }
        }
    };
};

template <typename Config, typename Globals, int Opcode, int BarrierRow>
struct siglip_stats {
    static constexpr int opcode = Opcode;
    using x_vec = kittens::sv_bf<vis::WIDTH>;
    static constexpr int TOKENS_PER_INST = Config::NUM_CONSUMER_WARPS;

    struct parsed_instruction {
        int opcode, layer, token_row, stats_row;
        int dep_layer, dep_row, dep_expected, num_tokens, flags, sq_row;

        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), token_row(i[2]), stats_row(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]), num_tokens(i[7]),
              flags(i[11]), sq_row(i[8] ? i[8] : i[3] + 1) {}
        __device__ inline parsed_instruction(::megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline kittens::semaphore &x_arrived(::megakernel::state<Config> &s) { return s.semaphores()[0]; }
    __device__ static inline kittens::semaphore &done(::megakernel::state<Config> &s)      { return s.semaphores()[1]; }

    struct controller {
        static __device__ int release_lid(const Globals &g,
                                          typename Config::instruction_t &instruction, int &query) {

            return (query + Config::NUM_PAGES - 1) % Config::NUM_PAGES;
        }
        static __device__ int init_semaphores(const Globals &g, ::megakernel::state<Config> &s) {
            kittens::init_semaphore(s.semaphores()[0], 1);
            kittens::init_semaphore(s.semaphores()[1], Config::NUM_CONSUMER_WARPS);
            return 2;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (kittens::laneid() == 0) {
                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
                int x_pid = s.pid(0);
                s.wait_page_ready(x_pid);
                x_vec *x_smem = reinterpret_cast<x_vec *>(s.pages[x_pid].data);
                kittens::tma::expect_bytes(x_arrived(s), inst.num_tokens * sizeof(x_vec));
                for (int i = 0; i < inst.num_tokens; i++)
                    kittens::tma::load_async(x_smem[i], g.vision_x, {inst.token_row + i, 0}, x_arrived(s));
            } else if (kittens::laneid() >= 1 && kittens::laneid() < Config::NUM_PAGES) {
                int pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            kittens::warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(x_arrived(s), 0);
            const int w = kittens::warpid();
            if (w >= inst.num_tokens) { kittens::warp::arrive(done(s)); return; }

            kittens::bf16 *x = reinterpret_cast<kittens::bf16 *>(s.pages[s.pid(0)].data) + w * vis::WIDTH;
            const int lane = kittens::laneid();
            float sx = 0.f, sxx = 0.f;
            for (int c = lane; c < vis::WIDTH; c += 32) {
                float v = __bfloat162float(x[c]);
                sx += v; sxx += v * v;
            }
            for (int off = 16; off > 0; off >>= 1) {
                sx  += __shfl_xor_sync(0xffffffff, sx,  off);
                sxx += __shfl_xor_sync(0xffffffff, sxx, off);
            }
            if (lane == 0) {
                const int tok = inst.token_row + w;
                g.ln_stats[kittens::coord<>{inst.stats_row,     tok}] = sx;
                g.ln_stats[kittens::coord<>{inst.sq_row, tok}] = sxx;
            }
            __syncwarp();
            kittens::warp::arrive(done(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(done(s), 0);
            if (kittens::laneid() == 0) {
                s.finish_page(s.pid(0), Config::NUM_CONSUMER_WARPS);

                mk_phase_release((uint32_t *)&g.Bar[kittens::coord<>{inst.layer, BarrierRow, 0}],
                                 inst.flags & 8);
            }
            kittens::warp::sync();
        }
    };
};

// row == opcode - 1
template <typename Config, typename Globals>
using siglip_ln_attn_op = siglip_ln<Config, Globals, OPCODE_SIGLIP_LN_ATTN, BARRIER_ROW(OPCODE_SIGLIP_LN_ATTN)>;
template <typename Config, typename Globals>
using siglip_ln_ffn_op = siglip_ln<Config, Globals, OPCODE_SIGLIP_LN_FFN, BARRIER_ROW(OPCODE_SIGLIP_LN_FFN)>;

template <typename Config, typename Globals>
using siglip_stats_attn_op = siglip_stats<Config, Globals, OPCODE_SIGLIP_LN_ATTN, BARRIER_ROW(OPCODE_SIGLIP_LN_ATTN)>;
template <typename Config, typename Globals>
using siglip_stats_ffn_op = siglip_stats<Config, Globals, OPCODE_SIGLIP_LN_FFN, BARRIER_ROW(OPCODE_SIGLIP_LN_FFN)>;

}  // namespace pi0
