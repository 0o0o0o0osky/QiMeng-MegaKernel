#pragma once

#include "kittens.cuh"
#include "include/megakernel.cuh"
#include "include/gmem_dep.cuh"
#include "include/pipelines/matmul_pipeline.cuh"
#include "common/instruction_list.cuh"   // mk_phase_release

namespace pi0 {

using namespace kittens;
using ::megakernel::state;

static constexpr float SIGLIP_LN_EPS_GEMM = 1.0e-5f;

// LayerNorm folding multiplies each weight column by gamma.

// Fold gamma into W and beta into the bias vector.

// out = rstd * (x @ W_folded^T - mean(x) * column_sum) + bias_folded.

struct EpiBias         { static constexpr bool bias=true;  static constexpr bool gelu=false; static constexpr bool residual=false; static constexpr bool row_scale=false; static constexpr bool gate_mul=false; static constexpr bool ln_fold=false; static constexpr bool resid_reg=false; static constexpr bool write_stats=false; };
struct EpiBiasGelu     { static constexpr bool bias=true;  static constexpr bool gelu=true;  static constexpr bool residual=false; static constexpr bool row_scale=false; static constexpr bool gate_mul=false; static constexpr bool ln_fold=false; static constexpr bool resid_reg=false; static constexpr bool write_stats=false; };
struct EpiBiasResidual { static constexpr bool bias=true;  static constexpr bool gelu=false; static constexpr bool residual=true;  static constexpr bool row_scale=false; static constexpr bool gate_mul=false; static constexpr bool ln_fold=false; static constexpr bool resid_reg=false; static constexpr bool write_stats=false; };

struct EpiLnFoldGelu   { static constexpr bool bias=false; static constexpr bool gelu=true;  static constexpr bool residual=false; static constexpr bool row_scale=false; static constexpr bool gate_mul=false; static constexpr bool ln_fold=true; static constexpr bool resid_reg=false; static constexpr bool write_stats=false;  };
struct EpiLnFold       { static constexpr bool bias=false; static constexpr bool gelu=false; static constexpr bool residual=false; static constexpr bool row_scale=false; static constexpr bool gate_mul=false; static constexpr bool ln_fold=true; static constexpr bool resid_reg=false; static constexpr bool write_stats=false;  };

struct EpiBiasResidStats { static constexpr bool bias=true; static constexpr bool gelu=false; static constexpr bool residual=false; static constexpr bool row_scale=false; static constexpr bool gate_mul=false; static constexpr bool ln_fold=false; static constexpr bool resid_reg=true; static constexpr bool write_stats=true; };

struct EpiRowScale     { static constexpr bool bias=false; static constexpr bool gelu=false; static constexpr bool residual=false; static constexpr bool row_scale=true;  static constexpr bool gate_mul=false; static constexpr bool ln_fold=false; static constexpr bool resid_reg=false; static constexpr bool write_stats=false; };
struct EpiRowScaleGelu { static constexpr bool bias=false; static constexpr bool gelu=true;  static constexpr bool residual=false; static constexpr bool row_scale=true;  static constexpr bool gate_mul=false; static constexpr bool ln_fold=false; static constexpr bool resid_reg=false; static constexpr bool write_stats=false; };
struct EpiRowScaleGate { static constexpr bool bias=false; static constexpr bool gelu=false; static constexpr bool residual=false; static constexpr bool row_scale=true;  static constexpr bool gate_mul=true;  static constexpr bool ln_fold=false; static constexpr bool resid_reg=false; static constexpr bool write_stats=false; };
struct EpiResidual     { static constexpr bool bias=false; static constexpr bool gelu=false; static constexpr bool residual=true;  static constexpr bool row_scale=false; static constexpr bool gate_mul=false; static constexpr bool ln_fold=false; static constexpr bool resid_reg=false; static constexpr bool write_stats=false; };

template <typename Config, typename Globals, typename Epi,
          auto A_Ptr, auto B_Ptr, auto Bias_Ptr, auto Out_Ptr,
          int K_ITERS, int Opcode, int BarrierRow, auto Aux_Ptr = A_Ptr,
          bool _ExplicitOut = false, auto CVec_Ptr = Bias_Ptr, int _StatsTiles = 1,
          int _TileN = 128>
struct siglip_gemm {
    static constexpr int opcode = Opcode;
    static constexpr int pipe_tile_n = _TileN;
    using bias_vec = sv_bf<pipe_tile_n>;

    struct parsed_instruction {
        int opcode, layer, local_row, local_col, dep_layer, dep_row, dep_expected, norm_sel;
        int out_row, out_col, bias_col;

        int flags;
        __device__ inline parsed_instruction(typename Config::instruction_t &i)
            : opcode(i[0]), layer(i[1]), local_row(i[2]), local_col(i[3]),
              dep_layer(i[4]), dep_row(i[5]), dep_expected(i[6]),
              norm_sel(i[7]),
              out_row (_ExplicitOut ? i[8]  : i[2]),
              out_col (_ExplicitOut ? i[9]  : i[3]),
              bias_col(_ExplicitOut ? i[10] : i[3]),
              flags(i[11]) {}
        __device__ inline parsed_instruction(state<Config> &s) : parsed_instruction(s.instruction()) {}

        __device__ inline bool wants_bias() const { return Epi::bias && bias_col >= 0; }
        __device__ inline bool pipelined()  const { return flags & 1; }
        __device__ inline bool lnf_early()  const { return flags & 2; }
        __device__ inline bool ln_self()    const { return flags & 4; }
        __device__ inline bool fused_rel()  const { return flags & 8; }
        __device__ inline bool no_stats()   const { return flags & 16; }
    };

    struct dep_waiter {
        __device__ static inline void gmem_wait(const Globals &g, state<Config> &s,
                                                const parsed_instruction &inst) {
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
        }
    };
    using pipe = matmul_pipeline<Config, Globals, parsed_instruction, dep_waiter, A_Ptr, B_Ptr,
                                 K_ITERS, _TileN>;

    __device__ static inline semaphore &output_arrived(state<Config> &s) { return s.semaphores()[pipe::SEM_COUNT]; }
    __device__ static inline semaphore &bias_arrived(state<Config> &s)   { return s.semaphores()[pipe::SEM_COUNT + 1]; }
    __device__ static inline semaphore &gate_arrived(state<Config> &s) {
        return s.semaphores()[pipe::SEM_COUNT + 1 + (Epi::bias ? 1 : 0)];
    }
    __device__ static inline semaphore &cvec_arrived(state<Config> &s) {
        return s.semaphores()[pipe::SEM_COUNT + 2 + (Epi::bias ? 1 : 0)];
    }

    __device__ static inline bias_vec &bias_smem(state<Config> &s) {
        return *reinterpret_cast<bias_vec *>(s.scratch());
    }

    static constexpr int SCALE_OFF_FLOATS = 1024;
    __device__ static inline float *row_scale_base(state<Config> &s) {
        return reinterpret_cast<float *>(s.scratch()) + SCALE_OFF_FLOATS;
    }
    __device__ static inline sv_fl<16> &row_scale_smem(state<Config> &s) {
        return *reinterpret_cast<sv_fl<16> *>(row_scale_base(s) + kittens::warpid() * 16);
    }

    static constexpr int MEAN_OFF_FLOATS = SCALE_OFF_FLOATS + 128;
    __device__ static inline float *row_mean_base(state<Config> &s) {
        return reinterpret_cast<float *>(s.scratch()) + MEAN_OFF_FLOATS;
    }
    __device__ static inline sv_fl<16> &row_mean_smem(state<Config> &s) {
        return *reinterpret_cast<sv_fl<16> *>(row_mean_base(s) + kittens::warpid() * 16);
    }

    static constexpr int STATS_OFF_FLOATS = 512;
    __device__ static inline sv_fl<16> &stats_smem(state<Config> &s) {
        return *reinterpret_cast<sv_fl<16> *>(
            reinterpret_cast<float *>(s.scratch()) + STATS_OFF_FLOATS + kittens::warpid() * 16);
    }

    static constexpr int CVEC_OFF_BYTES = 512;
    __device__ static inline bias_vec *cvec_smem(state<Config> &s) {
        return reinterpret_cast<bias_vec *>(reinterpret_cast<char *>(s.scratch()) + CVEC_OFF_BYTES);
    }

    struct controller {
        static __device__ int release_lid(const Globals &g, typename Config::instruction_t &inst, int &query) {
            return pipe::template release_lid<2>(g, inst, query);
        }
        static __device__ int init_semaphores(const Globals &g, state<Config> &s) {
            if (laneid() == 0) {
                init_semaphore(output_arrived(s), Config::NUM_CONSUMER_WARPS);
                if constexpr (Epi::bias)     init_semaphore(bias_arrived(s), 1);
                if constexpr (Epi::gate_mul || Epi::resid_reg) init_semaphore(gate_arrived(s), 1);
                if constexpr (Epi::ln_fold)  init_semaphore(cvec_arrived(s), 1);
            }
            return pipe::init_semaphores(s) + 1 + (Epi::bias ? 1 : 0) + ((Epi::gate_mul || Epi::resid_reg) ? 1 : 0)
                   + (Epi::ln_fold ? 1 : 0);
        }
    };

    __device__ static inline void ln_fold_prologue(const Globals &g, state<Config> &s,
                                                   const parsed_instruction &inst) {

            if (inst.ln_self()) {
                if (laneid() == 0) {
                    tma::expect_bytes(cvec_arrived(s), 2 * sizeof(bias_vec));
                    if constexpr (_TileN == 128) {
                        tma::load_async(cvec_smem(s)[0], g.*CVec_Ptr,
                                        {inst.layer, 2 * inst.local_col,     0}, cvec_arrived(s));
                        tma::load_async(cvec_smem(s)[1], g.*CVec_Ptr,
                                        {inst.layer, 2 * inst.local_col + 1, 0}, cvec_arrived(s));
                    } else {
                        static_assert(_TileN == 64);
                        const int parent = inst.local_col / 2;
                        const int offset = inst.local_col & 1;
                        tma::load_async(cvec_smem(s)[0], g.*CVec_Ptr,
                                        {inst.layer, 2 * parent,     offset}, cvec_arrived(s));
                        tma::load_async(cvec_smem(s)[1], g.*CVec_Ptr,
                                        {inst.layer, 2 * parent + 1, offset}, cvec_arrived(s));
                    }
                }
                return;
            }
            gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);
            const int sr   = inst.norm_sel;
            const int row0 = inst.local_row * pipe::TILE_M;
            float *rstd_dst = row_scale_base(s);
            float *mu_dst   = row_mean_base(s);

            static_assert(pipe::TILE_M == 128 || !Epi::ln_fold,
                          "The float4 LayerNorm-fold path requires TILE_M=128 with four rows per lane");
            const int c0 = row0 + 4 * (int)laneid();
            float4 sx4 = make_float4(0.f, 0.f, 0.f, 0.f);
            float4 sq4 = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int t = 0; t < _StatsTiles; t++) {
                const float4 a = *reinterpret_cast<const float4 *>(
                    &g.ln_stats[coord<>{sr + t, c0}]);
                const float4 b = *reinterpret_cast<const float4 *>(
                    &g.ln_stats[coord<>{sr + _StatsTiles + t, c0}]);
                sx4.x += a.x; sx4.y += a.y; sx4.z += a.z; sx4.w += a.w;
                sq4.x += b.x; sq4.y += b.y; sq4.z += b.z; sq4.w += b.w;
            }
            const float inv_w = 1.0f / float(vis::WIDTH);
            float4 mu4, rs4;
            {
                const float m = sx4.x * inv_w; mu4.x = m;
                rs4.x = rsqrtf(sq4.x * inv_w - m * m + SIGLIP_LN_EPS_GEMM);
            }
            {
                const float m = sx4.y * inv_w; mu4.y = m;
                rs4.y = rsqrtf(sq4.y * inv_w - m * m + SIGLIP_LN_EPS_GEMM);
            }
            {
                const float m = sx4.z * inv_w; mu4.z = m;
                rs4.z = rsqrtf(sq4.z * inv_w - m * m + SIGLIP_LN_EPS_GEMM);
            }
            {
                const float m = sx4.w * inv_w; mu4.w = m;
                rs4.w = rsqrtf(sq4.w * inv_w - m * m + SIGLIP_LN_EPS_GEMM);
            }
            *reinterpret_cast<float4 *>(mu_dst   + 4 * laneid()) = mu4;
            *reinterpret_cast<float4 *>(rstd_dst + 4 * laneid()) = rs4;
            warp::sync();
            if (laneid() == 0) {

                tma::expect_bytes(cvec_arrived(s), 2 * sizeof(bias_vec));
                if constexpr (_TileN == 128) {
                    tma::load_async(cvec_smem(s)[0], g.*CVec_Ptr,
                                    {inst.layer, 2 * inst.local_col,     0}, cvec_arrived(s));
                    tma::load_async(cvec_smem(s)[1], g.*CVec_Ptr,
                                    {inst.layer, 2 * inst.local_col + 1, 0}, cvec_arrived(s));
                } else {
                    static_assert(_TileN == 64);
                    const int parent = inst.local_col / 2;
                    const int offset = inst.local_col & 1;
                    tma::load_async(cvec_smem(s)[0], g.*CVec_Ptr,
                                    {inst.layer, 2 * parent,     offset}, cvec_arrived(s));
                    tma::load_async(cvec_smem(s)[1], g.*CVec_Ptr,
                                    {inst.layer, 2 * parent + 1, offset}, cvec_arrived(s));
                }
            }
    }

    struct loader {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            if constexpr (Epi::row_scale) {

                gmem_dep_wait<Config>(g, Opcode, inst.dep_layer, inst.dep_row, inst.dep_expected);

                const float *src = &g.rms_factor[coord<>{inst.norm_sel, inst.local_row * pipe::TILE_M}];
                float *dst = row_scale_base(s);
#pragma unroll
                for (int i = laneid(); i < pipe::TILE_M; i += 32) dst[i] = src[i];
            }
            if constexpr (Epi::ln_fold) {
                if (inst.lnf_early()) ln_fold_prologue(g, s, inst);
            }
            if constexpr (Epi::bias) {
                if (inst.wants_bias() && laneid() == 0) {
                    tma::expect(bias_arrived(s), bias_smem(s));

                    if constexpr (_TileN == 128) {
                        tma::load_async(bias_smem(s), g.*Bias_Ptr,
                                        {inst.layer, inst.bias_col, 0}, bias_arrived(s));
                    } else {
                        static_assert(_TileN == 64);
                        tma::load_async(bias_smem(s), g.*Bias_Ptr,
                                        {inst.layer, inst.bias_col / 2,
                                         inst.bias_col & 1}, bias_arrived(s));
                    }
                }
            }
            pipe::template loader_loop<dim::ROW>(s, g, inst.layer);

            if constexpr (Epi::ln_fold) {
                if (!inst.lnf_early()) ln_fold_prologue(g, s, inst);
            }

            if constexpr (Epi::gate_mul || Epi::resid_reg) {

                if (laneid() == 0) {
                    s.wait_page_ready(pipe::get_output_page(s, 0));
                    tma::expect_bytes(gate_arrived(s),
                                      pipe::MATMULS_PER_STAGE * sizeof(typename pipe::o_st));
#pragma unroll
                    for (int i = 0; i < pipe::MATMULS_PER_STAGE; i++)
                        tma::load_async(pipe::get_output_tile(s, i), g.*Aux_Ptr,
                                        {pipe::MATMULS_PER_STAGE * inst.local_row + i, inst.local_col},
                                        gate_arrived(s));
                }
                warp::sync();
            }
        }
    };

    struct launcher {
        static __device__ void run(const Globals &g, state<Config> &s) {}
    };

    template <int Fence, typename AT, typename BT>
    __device__ static inline void mma_step(rt_fl<16, pipe::TILE_N> &d, const AT &a, const BT &b) {
        warpgroup::mma_ABt<rt_fl<16, pipe::TILE_N>, AT, BT, Fence, 1>(d, a, b);
    }

    __device__ static __forceinline__ rt_fl<16, pipe::TILE_N> matmul_loop_pipelined(state<Config> &s) {
        static_assert(pipe::MATMULS_PER_STAGE == 2,
                      "This pipelined page layout requires backbone MPS=2; MPS=1 must use matmul_loop");
        static_assert(K_ITERS >= 5, "The i == _num_iters - 2 case must occur inside the i >= 1 loop");
        constexpr int NI = K_ITERS;
        constexpr int ST = pipe::INPUT_PIPELINE_STAGES;
        const int groupid = warpgroup::groupid();

        rt_fl<16, pipe::TILE_N> out;
        {
            ::megakernel::mk_semaphore_wait(pipe::inputs_arrived(s, 0), 0);
            warpgroup::mm_ABt(out, pipe::get_a_tile(s, 0)[groupid], pipe::get_b_tile(s, 0));
        }
        for (int i = 1; i < NI; i++) {
            const int st = i % ST;
            ::megakernel::mk_semaphore_wait(pipe::inputs_arrived(s, st), (i % (2 * ST)) >= ST);
            mma_step<0>(out, pipe::get_a_tile(s, st)[groupid], pipe::get_b_tile(s, st));

            warpgroup::mma_async_wait<1>();
            warpgroup::arrive(pipe::inputs_finished(s, (i - 1) % ST));

            if (i == NI - 2) {
                s.warp_finish_page(s.pid(2), 1);   // B stage 0,1
                s.warp_finish_page(s.pid(0), 1);   // A stage 0,1
            }
        }
        warpgroup::mma_async_wait<0>();
        warpgroup::arrive(pipe::inputs_finished(s, (NI - 1) % ST));
        s.warp_finish_page(s.pid(3), 1);                           // B stage 2,3
        s.warp_finish_page(s.pid(1), 1);                           // A stage 2,3
        return out;
    }

    __device__ static inline void accum_a_stats(state<Config> &s, int stage, int gid,
                                                kittens::col_vec<rt_fl<16, pipe::TILE_N>> &sx,
                                                kittens::col_vec<rt_fl<16, pipe::TILE_N>> &sxx) {
        rt_fl<16, PIPELINE_K_DIM> af;
        warpgroup::load(af, pipe::get_a_tile(s, stage)[gid]);
        warp::row_sum(sx, af, sx);
        warp::mul(af, af, af);
        warp::row_sum(sxx, af, sxx);
    }

    __device__ static __forceinline__ rt_fl<16, pipe::TILE_N>
    matmul_loop_stats(state<Config> &s,
                      kittens::col_vec<rt_fl<16, pipe::TILE_N>> &sx,
                      kittens::col_vec<rt_fl<16, pipe::TILE_N>> &sxx) {
        static_assert(pipe::MATMULS_PER_STAGE == 2, "The page layout requires backbone MPS=2");
        constexpr int NI = K_ITERS;
        constexpr int ST = pipe::INPUT_PIPELINE_STAGES;
        const int groupid = warpgroup::groupid();

        rt_fl<16, pipe::TILE_N> out;
        warp::zero(sx);
        warp::zero(sxx);

        int input_stage = 0;
        {
            ::megakernel::mk_semaphore_wait(pipe::inputs_arrived(s, input_stage), 0);
            warpgroup::mm_ABt(out, pipe::get_a_tile(s, input_stage)[groupid],
                              pipe::get_b_tile(s, input_stage));
            accum_a_stats(s, input_stage, groupid, sx, sxx);
            warpgroup::mma_async_wait();
            warpgroup::arrive(pipe::inputs_finished(s, input_stage));
            input_stage = (input_stage + 1) % ST;
        }
        for (int i = 1; i < NI; i++) {
            ::megakernel::mk_semaphore_wait(pipe::inputs_arrived(s, input_stage), (i % (2 * ST)) >= ST);
            warpgroup::mma_ABt(out, pipe::get_a_tile(s, input_stage)[groupid],
                               pipe::get_b_tile(s, input_stage));
            accum_a_stats(s, input_stage, groupid, sx, sxx);
            warpgroup::mma_async_wait();
            warpgroup::arrive(pipe::inputs_finished(s, input_stage));

            if (i == NI - 3) {
                s.warp_finish_page(s.pid(2), 1);   // B stage 0,1
                s.warp_finish_page(s.pid(0), 1);   // A stage 0,1
            } else if (i == NI - 1) {
                s.warp_finish_page(s.pid(3), 1);   // B stage 2,3
                s.warp_finish_page(s.pid(1), 1);   // A stage 2,3
            }
            input_stage = (input_stage + 1) % ST;
        }
        return out;
    }

    struct consumer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            rt_fl<16, pipe::TILE_N> acc;

            kittens::col_vec<rt_fl<16, pipe::TILE_N>> self_sx, self_sxx;
            const bool ln_self = parsed_instruction{s}.ln_self();
            {
                parsed_instruction pi{s};
                if constexpr (Epi::ln_fold) {
                    if (ln_self) acc = matmul_loop_stats(s, self_sx, self_sxx);
                    else if (pi.pipelined()) acc = matmul_loop_pipelined(s);
                    else acc = pipe::template matmul_loop<2>(s, g);
                } else {
                    if (pi.pipelined()) acc = matmul_loop_pipelined(s);
                    else                acc = pipe::template matmul_loop<2>(s, g);
                }
            }

            if constexpr (Epi::row_scale) {

                kittens::col_vec<rt_fl<16, pipe::TILE_N>> sc;
                warp::load(sc, row_scale_smem(s));
                warp::mul_row(acc, acc, sc);
            }

            if constexpr (Epi::ln_fold) {
                ::megakernel::mk_semaphore_wait(cvec_arrived(s), 0);

                kittens::col_vec<rt_fl<16, pipe::TILE_N>> mu, rstd;
                if (ln_self) {

                    static_assert(std::is_same_v<typename decltype(mu)::dtype, float2>,
                                  "Update the .x/.y expansion when the col_vec element type changes");
                    constexpr float inv_w = 1.0f / float(vis::WIDTH);
#pragma unroll
                    for (int i = 0; i < decltype(mu)::outer_dim; i++)
#pragma unroll
                        for (int j = 0; j < decltype(mu)::inner_dim; j++) {
                            const float2 a = self_sx.data[i][j];
                            const float2 q = self_sxx.data[i][j];
                            const float m0 = a.x * inv_w, m1 = a.y * inv_w;
                            mu.data[i][j] = make_float2(m0, m1);
                            rstd.data[i][j] = make_float2(
                                rsqrtf(q.x * inv_w - m0 * m0 + SIGLIP_LN_EPS_GEMM),
                                rsqrtf(q.y * inv_w - m1 * m1 + SIGLIP_LN_EPS_GEMM));
                        }
                } else {
                    warp::load(mu, row_mean_smem(s));
                    warp::load(rstd, row_scale_smem(s));
                }
                {
                    rt_fl<16, pipe::TILE_N> t;
                    kittens::row_vec<rt_fl<16, pipe::TILE_N>> c_rv;
                    warp::load(c_rv, cvec_smem(s)[0]);
                    warp::zero(t);
                    warp::add_col(t, t, c_rv);          // t[m][n] = c[n]
                    warp::mul_row(t, t, mu);            // t[m][n] = mean[m] * column_sum[n]
                    warp::sub(acc, acc, t);
                }
                {
                    warp::mul_row(acc, acc, rstd);      // acc = rstd * (x @ W_folded^T - mean * column_sum)
                }
                {
                    kittens::row_vec<rt_fl<16, pipe::TILE_N>> bp;
                    warp::load(bp, cvec_smem(s)[1]);
                    warp::add_col(acc, acc, bp);        // Add the folded bias.
                }
            }
            if constexpr (Epi::bias) {

                parsed_instruction inst{s};
                if (inst.wants_bias()) {
                    ::megakernel::mk_semaphore_wait(bias_arrived(s), 0);
                    kittens::row_vec<rt_fl<16, pipe::TILE_N>> bias_rv;
                    warp::load(bias_rv, bias_smem(s));
                    warp::add_col(acc, acc, bias_rv);
                }
            }

            if constexpr (Epi::gelu) {
                // gelu_tanh(x) = x * sigmoid(1.5957691216 * (x + 0.044715 x^3))

                rt_fl<16, pipe::TILE_N> t;
                warp::mul(t, acc, acc);          // x^2
                warp::mul(t, t, acc);            // x^3
                warp::mul(t, t, 0.044715f);
                warp::add(t, t, acc);            // x + 0.044715 x^3
                warp::mul(t, t, 1.5957691216f);  // u
                warp::mul(t, t, -1.0f);          // -u
                warp::exp(t, t);                 // e = exp(-u)
                warp::add(t, t, 1.0f);           // 1+e
                warp::div(acc, acc, t);          // x/(1+e)
            }

            if constexpr (Epi::resid_reg) {
                ::megakernel::mk_semaphore_wait(gate_arrived(s), 0);
                rt_fl<16, pipe::TILE_N> r;
                warpgroup::load(r, pipe::get_output_tile(s, warpgroup::groupid()));
                warp::add(acc, acc, r);

                if constexpr (Epi::write_stats) {
                    parsed_instruction si{s};

                    if (!si.ln_self() && !si.no_stats()) {
                    const int base = si.local_row * pipe::TILE_M + kittens::warpid() * 16;
                    kittens::col_vec<rt_fl<16, pipe::TILE_N>> v;
                    warp::zero(v);
                    warp::row_sum(v, acc, v);
                    warp::store(stats_smem(s), v);
                    warp::sync();
                    if (laneid() < 16)
                        g.ln_stats[coord<>{si.local_col, base + laneid()}] = stats_smem(s)[laneid()];

                    warp::mul(r, acc, acc);
                    warp::zero(v);
                    warp::row_sum(v, r, v);                          // sum(x^2)
                    warp::store(stats_smem(s), v);
                    warp::sync();
                    if (laneid() < 16)
                        g.ln_stats[coord<>{_StatsTiles + si.local_col, base + laneid()}]
                            = stats_smem(s)[laneid()];
                    }
                }
            }

            if constexpr (Epi::gate_mul) {

                ::megakernel::mk_semaphore_wait(gate_arrived(s), 0);
                rt_fl<16, pipe::TILE_N> gate;
                warpgroup::load(gate, pipe::get_output_tile(s, warpgroup::groupid()));
                warp::mul(acc, acc, gate);
            }

            warpgroup::store(pipe::get_output_tile(s, warpgroup::groupid()), acc);
            warp::sync();
            warp::arrive(output_arrived(s));
        }
    };

    struct storer {
        static __device__ void run(const Globals &g, state<Config> &s) {
            parsed_instruction inst{s};
            ::megakernel::mk_semaphore_wait(output_arrived(s), 0);
            auto &Out = g.*Out_Ptr;

            if (laneid() == 0) {
#pragma unroll
                for (int i = 0; i < pipe::MATMULS_PER_STAGE; i++) {
                    auto &out_smem = pipe::get_output_tile(s, i);

                    if constexpr (Epi::residual) {

                        tma::store_add_async(Out, out_smem,
                                             {pipe::MATMULS_PER_STAGE * inst.out_row + i, inst.out_col});
                    } else {
                        tma::store_async(Out, out_smem,
                                         {pipe::MATMULS_PER_STAGE * inst.out_row + i, inst.out_col});
                    }
                }
                tma::store_async_read_wait();

                s.finish_page(pipe::get_output_page(s, 0), Config::NUM_CONSUMER_WARPS);
            }

            tma::store_async_wait();
            warp::sync();
            if (laneid() == 0) {
                mk_phase_release((uint32_t *)&g.Bar[coord<>{inst.layer, BarrierRow, 0}],
                                 inst.fused_rel());
            }
        }
    };
};

}  // namespace pi0
