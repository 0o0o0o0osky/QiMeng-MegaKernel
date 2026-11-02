#pragma once

#include "kittens.cuh"
#include "common/ae_globals.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"

namespace pi0 {

static constexpr float AE_RMS_EPS = 1e-6f;

template <typename Config, typename Globals, int Opcode, int NormSel>
struct ae_rmsn {
    static constexpr int opcode = Opcode;
    static constexpr int barrier_row = BARRIER_ROW(Opcode);

    using x_vec = kittens::sv_fl<dec::WIDTH>;                          // 1024 * 4B = 4096 B
    static constexpr int TOKENS_PER_INST = Config::NUM_CONSUMER_WARPS;  // 4
    static_assert(TOKENS_PER_INST * sizeof(x_vec) <= Config::PAGE_SIZE,
                  "The tokens processed by one fusion instruction must fit in one page");

    struct parsed_instruction {
        int opcode, token_row, dep_layer, dep_row, dep_expected, bar_layer, num_tokens;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), token_row(i[2]), dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]),
              bar_layer(i[7]), num_tokens(i[8]) {}
        __device__ inline parsed_instruction(::megakernel::state<Config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    __device__ static inline kittens::semaphore &x_arrived(::megakernel::state<Config> &s) { return s.semaphores()[0]; }
    __device__ static inline kittens::semaphore &done(::megakernel::state<Config> &s)      { return s.semaphores()[1]; }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &, int &query) {

            return (query + 1) % Config::NUM_PAGES;
        }
        static __device__ int init_semaphores(const Globals &g, ::megakernel::state<Config> &s) {
            for (int i = 0; i < 2; i++)
                kittens::init_semaphore(s.semaphores()[i], i == 1 ? Config::NUM_CONSUMER_WARPS : 1);
            return 2;
        }
    };

    struct loader {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            if (kittens::laneid() == 0) {
                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
                int pid = s.pid(0);
                s.wait_page_ready(pid);
                x_vec *x = reinterpret_cast<x_vec *>(s.pages[pid].data);

                kittens::tma::expect_bytes(x_arrived(s), inst.num_tokens * sizeof(x_vec));

                for (int i = 0; i < inst.num_tokens; i++)
                    kittens::tma::load_async(x[i], g.x_acc, {inst.token_row + i, 0}, x_arrived(s));
            } else if (kittens::laneid() >= 1 && kittens::laneid() < Config::NUM_PAGES) {
                int pid = s.pid(kittens::laneid());
                s.wait_page_ready(pid);
                s.finish_page(pid, Config::NUM_CONSUMER_WARPS);
            }
            kittens::warp::sync();
        }
    };

    struct launcher {
        static __device__ void run(const Globals &, ::megakernel::state<Config> &) {}
    };

    struct consumer {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(x_arrived(s), 0);

            const int w = kittens::warpid();
            if (w < inst.num_tokens) {
                const float *x = reinterpret_cast<float *>(s.pages[s.pid(0)].data) + w * dec::WIDTH;
                const int lane = kittens::laneid();
                const int row = inst.token_row + w;

                float sq = 0.0f;
                for (int c = lane; c < dec::WIDTH; c += 32) {
                    float v = x[c];
                    sq += v * v;
                    g.x[kittens::coord<>{row, c}] = __float2bfloat16(v);
                }
                for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, off);
                if (lane == 0) {

                    g.rms_factor[kittens::coord<>{NormSel, row}] =
                        rsqrtf(sq / float(dec::WIDTH) + AE_RMS_EPS);
                }
            }
            kittens::warp::arrive(done(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, ::megakernel::state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(done(s), 0);
            if (kittens::laneid() == 0) {
                s.finish_page(s.pid(0), Config::NUM_CONSUMER_WARPS);

                __threadfence();
                atomicAdd((uint32_t *)&g.Bar[kittens::coord<>{inst.bar_layer, barrier_row, 0}], 1);
            }
            kittens::warp::sync();
        }
    };
};

template <typename Config, typename Globals>
using ae_rmsn_attn_op = ae_rmsn<Config, Globals, OPCODE_AE_RMSN_ATTN, 0>;
template <typename Config, typename Globals>
using ae_rmsn_ffn_op  = ae_rmsn<Config, Globals, OPCODE_AE_RMSN_FFN, 1>;

#ifdef OPCODE_AE_RMSN_OUT
template <typename Config, typename Globals>
using ae_rmsn_out_op  = ae_rmsn<Config, Globals, OPCODE_AE_RMSN_OUT, 2>;
#endif

}  // namespace pi0
