// Pi0 Gemma kernels for A100/A800.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/gemma_globals.cuh"
#include "common/host_utils.cuh"
#include "common/instruction_list.cuh"
#include "common/matmul_pipeline.cuh"
#include "common/runtime.cuh"

#include "backbone/gemma/gemma_rmsn.cu"
#include "backbone/gemma/gemma_qkv_rope.cu"
#include "backbone/gemma/gemma_attn.cu"
#include "backbone/gemma/gemma_resid_gemm.cu"
#include "backbone/gemma/gemma_ffn_dual.cu"
#include "backbone/gemma/gemma_down_splitk.cu"

namespace py = pybind11;
using namespace pi0_a100;

static constexpr int kSmemNone = 16;
static constexpr int kSmemGemm = GemmaGemm64::kShmSize;
static constexpr int kSmemDual = mk_gemm::DualSmem<GemmaGemm64>::kShmSizeDual;
static constexpr int kSmemQk = gemma_attn_mat::QkConfig::kShmSize;
static constexpr int kSmemPv = gemma_attn_mat::PvConfig::kShmSize;

static_assert(gemma_attn_mat::PvConfig::BM == GemmaOprojWideN::BM);
static_assert(gemma_attn_mat::PvConfig::BN == GemmaOprojWideN::BN);
static_assert(gemma_attn_mat::PvConfig::BK == GemmaOprojWideN::BK);
static_assert(gemma_attn_mat::PvConfig::kStage == GemmaOprojWideN::kStage);

using GemmaDownView1EightWarp =
    mk_gemm::GemmConfig<64, 128, 64, 3, 2, 4, 2>;
using GemmaOprojView1 = mk_gemm::GemmConfig<32, 128, 64, 3, 2, 2, 2>;
using GemmaQkvQueryView2 = mk_gemm::GemmConfig<64, 128, 32, 4, 2, 2, 2>;
static_assert(GemmaQkvQueryView2::kPhysicalS2gVec == 8);
static constexpr int kSmemQkvView2 =
    GemmaQkvQueryView2::kShmSize > GemmaQkvGemm::kShmSize
        ? GemmaQkvQueryView2::kShmSize
        : GemmaQkvGemm::kShmSize;
static constexpr int kSmemQkvQkView2Base =
    kSmemQkvView2 > kSmemQk ? kSmemQkvView2 : kSmemQk;
static constexpr int kSmemQkvQkView2 =
    kSmemQkvQkView2Base + kInstructionWidth * sizeof(int32_t);
static constexpr int kSmemPvOprojView1 =
    GemmaOprojView1::kShmSize > kSmemPv
        ? GemmaOprojView1::kShmSize
        : kSmemPv;

__device__ __noinline__ void gemma_qkv_rope_view2(
    const GemmaGlobals& g, const int32_t* inst, unsigned char* smem) {
    // Two-view Q uses 128 columns per CTA; K/V keep 64-column tiles.
    constexpr int kInstructionColumns = GemmaQkvGemm::BN;
    const int layer = inst[1];
    const int logical_col0 = inst[3] * kInstructionColumns;
    const bool query_section = logical_col0 < gemma::kHeads * gemma::kHeadDim;
    const int section_col0 = query_section
        ? logical_col0
        : logical_col0 - gemma::kHeads * gemma::kHeadDim;
    const int section_width = query_section
        ? gemma::kHeads * gemma::kHeadDim
        : 2 * gemma::kHeadDim;
    const nv_bfloat16* weight = g.qkv_w +
        static_cast<int64_t>(layer) * gemma::kQkvWidth * gemma::kWidth +
        (query_section
            ? 0
            : static_cast<int64_t>(gemma::kHeads * gemma::kHeadDim) *
                  gemma::kWidth);
    const int row_tile = inst[2];
    const float* factor =
        g.rms_factor + static_cast<int64_t>(inst[7]) * gemma::kMaxSeq;
    GemmaQkvFragmentEpilogue epilogue{
        factor, g.rope, g.k_cache, g.v_cache, layer,
        query_section ? 0 : gemma::kHeads * gemma::kHeadDim,
    };

    if (query_section) {
        const int col_tile = section_col0 / GemmaQkvQueryView2::BN;
        mk_gemm::gemm_unified_tile<GemmaQkvQueryView2>(
            g.encoder_x, weight, g.qkv_buf, g.seq, section_width,
            gemma::kWidth, row_tile, col_tile, col_tile + 1, epilogue,
            smem, false, gemma::kWidth, gemma::kWidth, gemma::kQkvWidth);
    } else {
        const int col_tile = section_col0 / GemmaQkvGemm::BN;
        mk_gemm::gemm_unified_tile<GemmaQkvGemm>(
            g.encoder_x, weight, g.qkv_buf + epilogue.output_col_offset,
            g.seq, section_width, gemma::kWidth, row_tile, col_tile,
            col_tile + 1, epilogue, smem, false, gemma::kWidth,
            gemma::kWidth, gemma::kQkvWidth);
    }
}

__global__ __launch_bounds__(GemmaOprojView1::kNumThreads, 1)
void gemma_final_oproj_view1(GemmaGlobals g, int layer) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kNTiles = gemma::kWidth / GemmaOprojView1::BN;
    const int task = static_cast<int>(blockIdx.x);
    const int tile_m = task / kNTiles;
    const int tile_n = task - tile_m * kNTiles;
    const nv_bfloat16* weight =
        g.oproj_w + static_cast<int64_t>(layer) * gemma::kWidth * gemma::kWidth;
    mk_gemm::EpilogueResidual<nv_bfloat16> epilogue{g.encoder_x, gemma::kWidth};
    mk_gemm::gemm_unified_tile<GemmaOprojView1>(
        g.attn_out, weight, g.encoder_x,
        g.seq, gemma::kWidth, gemma::kWidth,
        tile_m, tile_n, tile_n + 1, epilogue, smem);
}

__global__ __launch_bounds__(GemmaDownView1EightWarp::kNumThreads, 1)
void gemma_final_down_view1_eight_warp(GemmaGlobals g, int layer) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kNTiles = gemma::kWidth / GemmaDownView1EightWarp::BN;
    const int task = static_cast<int>(blockIdx.x);
    const int tile_m = task / kNTiles;
    const int tile_n = task - tile_m * kNTiles;
    const nv_bfloat16* weight =
        g.down_w + static_cast<int64_t>(layer) * gemma::kWidth * gemma::kMlpWidth;
    mk_gemm::EpilogueResidual<nv_bfloat16> epilogue{g.encoder_x, gemma::kWidth};
    mk_gemm::gemm_unified_tile<GemmaDownView1EightWarp>(
        g.up_buf, weight, g.encoder_x,
        g.seq, gemma::kWidth, gemma::kMlpWidth,
        tile_m, tile_n, tile_n + 1, epilogue, smem);
}

// Three-view PV -> OProj subgraph.  Both sides use the same 64x128x32/s4
// execution tile and independently host two CTAs/SM.  There are exactly 208
// tasks on each side: PV is 8 heads * 13 row tiles * 2 N tiles, while OProj is
// 13 row tiles * 16 N tiles.  A row counter lets OProj start as soon as all PV
// heads for that row tile are visible; its dependency-free weight stages are
// prefetched before the wait, following the Llama single-batch pattern.
__global__ __launch_bounds__(gemma_attn_mat::PvConfig::kNumThreads, 2)
void gemma_final_pv_oproj_v3_subgraph(
    GemmaGlobals g, int32_t* ready, int layer, bool prefetch_oproj) {
    extern __shared__ __align__(16) unsigned char smem[];
    using Pv = gemma_attn_mat::PvConfig;
    using Oproj = GemmaOprojWideN;
    constexpr int kRows = 832;
    constexpr int kMTiles = kRows / Pv::BM;
    constexpr int kPvNTiles = gemma::kHeadDim / Pv::BN;
    constexpr int kOprojNTiles = gemma::kWidth / Oproj::BN;
    constexpr int kGrid = gemma::kHeads * kMTiles * kPvNTiles;
    static_assert(kMTiles == 13);
    static_assert(kGrid == 208);
    static_assert(kMTiles * kOprojNTiles == kGrid);

    const int task = static_cast<int>(blockIdx.x);
    const int pv_head = task / (kMTiles * kPvNTiles);
    const int pv_rem = task % (kMTiles * kPvNTiles);
    const int pv_m = pv_rem / kPvNTiles;
    const int pv_n = pv_rem % kPvNTiles;
    mk_gemm::EpilogueIdentity pv_epi{};
    mk_gemm::gemm_unified_tile<Pv>(
        g.attn_scores + static_cast<int64_t>(pv_head) * kRows * kRows,
        g.attn_vt,
        g.attn_out + pv_head * gemma::kHeadDim,
        kRows, gemma::kHeadDim, kRows,
        pv_m, pv_n, pv_n + 1, pv_epi, smem, false,
        kRows, kRows, gemma::kWidth);
    __syncthreads();
    if (threadIdx.x == 0) {
        ampere_barrier::publish(ready + pv_m);
    }

    const int oproj_m = task / kOprojNTiles;
    const int oproj_n = task % kOprojNTiles;
    const nv_bfloat16* weight = g.oproj_w +
        static_cast<int64_t>(layer) * gemma::kWidth * gemma::kWidth;
    if (prefetch_oproj) {
        mk_gemm::gemm_unified_preload_b_stages<Oproj>(
            weight, gemma::kWidth, gemma::kWidth, oproj_n, smem);
    }
    if (threadIdx.x == 0) {
        ampere_barrier::wait(
            ready + oproj_m,
            static_cast<uint32_t>((layer + 1) * gemma::kHeads * kPvNTiles),
#ifndef NDEBUG
            task, 0, GEMMA_OPROJ, oproj_m * sizeof(int32_t), 4000000000ULL,
#endif
            0);
    }
    __syncthreads();

    mk_gemm::EpilogueResidual<nv_bfloat16> oproj_epi{
        g.encoder_x, gemma::kWidth};
    mk_gemm::gemm_unified_tile<Oproj>(
        g.attn_out, weight, g.encoder_x,
        kRows, gemma::kWidth, gemma::kWidth,
        oproj_m, oproj_n, oproj_n + 1, oproj_epi, smem,
        prefetch_oproj);
}

// Two-view PV and OProj each expose 144 two-CTA/SM tasks. The first OProj
// weight stage is prefetched while its CTA waits for the matching PV row.
__global__ __launch_bounds__(gemma_attn_mat::PvConfig::kNumThreads, 2)
void gemma_final_pv_oproj_v2_subgraph(
    GemmaGlobals g, int32_t* ready, int layer, bool prefetch_oproj) {
    extern __shared__ __align__(16) unsigned char smem[];
    using Pv = gemma_attn_mat::PvConfig;
    using Oproj = GemmaGemm64;
    constexpr int kRows = 576;
    constexpr int kMTiles = kRows / Pv::BM;
    constexpr int kPvNTiles = gemma::kHeadDim / Pv::BN;
    constexpr int kOprojLogicalNTiles =
        gemma::kWidth / kGemmaInstructionCols;
    constexpr int kOprojPhysicalPerLogical =
        kGemmaInstructionCols / Oproj::BN;
    constexpr int kGrid = gemma::kHeads * kMTiles * kPvNTiles;
    static_assert(kMTiles == 9);
    static_assert(kGrid == 144);
    static_assert(kMTiles * kOprojLogicalNTiles == kGrid);
    static_assert(kOprojPhysicalPerLogical == 2);

    const int task = static_cast<int>(blockIdx.x);
    const int pv_head = task / (kMTiles * kPvNTiles);
    const int pv_rem = task % (kMTiles * kPvNTiles);
    const int pv_m = pv_rem / kPvNTiles;
    const int pv_n = pv_rem % kPvNTiles;
    mk_gemm::EpilogueIdentity pv_epi{};
    mk_gemm::gemm_unified_tile<Pv>(
        g.attn_scores + static_cast<int64_t>(pv_head) * kRows * kRows,
        g.attn_vt,
        g.attn_out + pv_head * gemma::kHeadDim,
        kRows, gemma::kHeadDim, kRows,
        pv_m, pv_n, pv_n + 1, pv_epi, smem, false,
        kRows, kRows, gemma::kWidth);
    __syncthreads();
    if (threadIdx.x == 0) {
        ampere_barrier::publish(ready + pv_m);
    }

    const int oproj_m = task / kOprojLogicalNTiles;
    const int oproj_n = task % kOprojLogicalNTiles;
    const int physical_n = oproj_n * kOprojPhysicalPerLogical;
    const nv_bfloat16* weight = g.oproj_w +
        static_cast<int64_t>(layer) * gemma::kWidth * gemma::kWidth;
    if (prefetch_oproj) {
        mk_gemm::gemm_unified_preload_b_stages<Oproj>(
            weight, gemma::kWidth, gemma::kWidth, physical_n, smem);
    }
    if (threadIdx.x == 0) {
        ampere_barrier::wait(
            ready + oproj_m,
            static_cast<uint32_t>((layer + 1) * gemma::kHeads * kPvNTiles),
#ifndef NDEBUG
            task, 0, GEMMA_OPROJ, oproj_m * sizeof(int32_t), 4000000000ULL,
#endif
            0);
    }
    __syncthreads();

    mk_gemm::EpilogueResidual<nv_bfloat16> oproj_epi{
        g.encoder_x, gemma::kWidth};
    mk_gemm::gemm_unified_tile<Oproj>(
        g.attn_out, weight, g.encoder_x,
        kRows, gemma::kWidth, gemma::kWidth,
        oproj_m, physical_n, physical_n + kOprojPhysicalPerLogical,
        oproj_epi, smem, prefetch_oproj);
}

// One-view PV -> OProj.  PV has 80 tasks at seq_pad=320 while the selected
// BM32 OProj has 160.  Both use 128 threads and retain two CTA/SM in this
// subgraph.  A PV 64-row tile publishes the two BM32 OProj rows it produces.
__global__ __launch_bounds__(gemma_attn_mat::PvConfig::kNumThreads, 2)
void gemma_final_pv_oproj_v1_subgraph(
    GemmaGlobals g, int32_t* ready, int layer, bool prefetch_oproj) {
    extern __shared__ __align__(16) unsigned char smem[];
    using Pv = gemma_attn_mat::PvConfig;
    using Oproj = GemmaOprojView1;
    constexpr int kRows = 320;
    constexpr int kPvMTiles = kRows / Pv::BM;
    constexpr int kPvNTiles = gemma::kHeadDim / Pv::BN;
    constexpr int kOprojMTiles = kRows / Oproj::BM;
    constexpr int kOprojNTiles = gemma::kWidth / Oproj::BN;
    constexpr int kPvTasks = gemma::kHeads * kPvMTiles * kPvNTiles;
    constexpr int kGrid = kOprojMTiles * kOprojNTiles;
    static_assert(kPvTasks == 80);
    static_assert(kGrid == 160);
    static_assert(Pv::kNumThreads == Oproj::kNumThreads);
    static_assert(Pv::BM == 2 * Oproj::BM);

    const int task = static_cast<int>(blockIdx.x);
    if (task < kPvTasks) {
        const int pv_head = task / (kPvMTiles * kPvNTiles);
        const int pv_rem = task % (kPvMTiles * kPvNTiles);
        const int pv_m = pv_rem / kPvNTiles;
        const int pv_n = pv_rem % kPvNTiles;
        mk_gemm::EpilogueIdentity pv_epi{};
        mk_gemm::gemm_unified_tile<Pv>(
            g.attn_scores + static_cast<int64_t>(pv_head) * kRows * kRows,
            g.attn_vt,
            g.attn_out + pv_head * gemma::kHeadDim,
            kRows, gemma::kHeadDim, kRows,
            pv_m, pv_n, pv_n + 1, pv_epi, smem, false,
            kRows, kRows, gemma::kWidth);
        __syncthreads();
        if (threadIdx.x == 0) {
            ampere_barrier::publish(ready + 2 * pv_m);
            ampere_barrier::publish(ready + 2 * pv_m + 1);
        }
    }

    const int oproj_m = task / kOprojNTiles;
    const int oproj_n = task % kOprojNTiles;
    const nv_bfloat16* weight = g.oproj_w +
        static_cast<int64_t>(layer) * gemma::kWidth * gemma::kWidth;
    if (prefetch_oproj) {
        mk_gemm::gemm_unified_preload_b_stages<Oproj>(
            weight, gemma::kWidth, gemma::kWidth, oproj_n, smem);
    }
    if (threadIdx.x == 0) {
        ampere_barrier::wait(
            ready + oproj_m,
            static_cast<uint32_t>((layer + 1) * gemma::kHeads * kPvNTiles),
#ifndef NDEBUG
            task, 0, GEMMA_OPROJ, oproj_m * sizeof(int32_t), 4000000000ULL,
#endif
            0);
    }
    __syncthreads();

    mk_gemm::EpilogueResidual<nv_bfloat16> oproj_epi{
        g.encoder_x, gemma::kWidth};
    mk_gemm::gemm_unified_tile<Oproj>(
        g.attn_out, weight, g.encoder_x,
        kRows, gemma::kWidth, gemma::kWidth,
        oproj_m, oproj_n, oproj_n + 1, oproj_epi, smem,
        prefetch_oproj);
}

__device__ __forceinline__ void gemma_subgraph_grid_barrier(
    int32_t* words, int participants) {
    __syncthreads();
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned epoch = atomicAdd(
            reinterpret_cast<unsigned*>(words + 1), 0u);
        const unsigned ticket = atomicAdd(
            reinterpret_cast<unsigned*>(words), 1u);
        if (ticket == static_cast<unsigned>(participants - 1)) {
            atomicExch(reinterpret_cast<unsigned*>(words), 0u);
            __threadfence();
            atomicAdd(reinterpret_cast<unsigned*>(words + 1), 1u);
        } else {
            while (atomicAdd(
                       reinterpret_cast<unsigned*>(words + 1), 0u) == epoch) {
                __nanosleep(32);
            }
        }
    }
    __syncthreads();
}

// Two-view mixed-QKV and materialized QK share a fully resident grid.
__global__ __launch_bounds__(kThreads, 2)
void gemma_final_qkv_qk_v2_subgraph(
    GemmaGlobals g, int32_t* barrier, int layer, int seq_len) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kRows = 576;
    constexpr int kQkvMTiles = 9;
    constexpr int kQkvCompactNTiles = 24;
    constexpr int kGrid = kQkvMTiles * kQkvCompactNTiles;
    constexpr int kQkMTiles = (kRows + gemma_attn_mat::QkConfig::BM - 1) /
        gemma_attn_mat::QkConfig::BM;
    constexpr int kQkNTiles = (kRows + gemma_attn_mat::QkConfig::BN - 1) /
        gemma_attn_mat::QkConfig::BN;
    constexpr int kQkTasks = gemma::kHeads * kQkMTiles * kQkNTiles;
    constexpr int kVtTasks =
        (gemma::kHeadDim / gemma_attn_mat::kVtTile) *
        (kRows / gemma_attn_mat::kVtTile);
    constexpr int kAttentionTasks = kQkTasks + kVtTasks;
    static_assert(kGrid == 216);
    static_assert(kQkTasks == 200);
    static_assert(kVtTasks == 144);
    static_assert(kAttentionTasks == 344);

    int32_t* static_inst = reinterpret_cast<int32_t*>(
        smem + kSmemQkvQkView2Base);
    const int qkv_task = static_cast<int>(blockIdx.x);
    const int qkv_m = qkv_task / kQkvCompactNTiles;
    const int compact_n = qkv_task % kQkvCompactNTiles;
    const int logical_n = compact_n < 16 ? 2 * compact_n : compact_n + 16;
    if (threadIdx.x == 0) {
        static_inst[1] = layer;
        static_inst[2] = qkv_m;
        static_inst[3] = logical_n;
        static_inst[7] = 0;
    }
    __syncthreads();
    gemma_qkv_rope_view2(g, static_inst, smem);
    gemma_subgraph_grid_barrier(barrier, kGrid);

    for (int task = static_cast<int>(blockIdx.x);
         task < kAttentionTasks; task += kGrid) {
        if (threadIdx.x == 0) {
            static_inst[1] = layer;
            static_inst[8] = seq_len;
            if (task < kQkTasks) {
                const int per_head = kQkMTiles * kQkNTiles;
                const int head = task / per_head;
                const int within_head = task % per_head;
                static_inst[2] = within_head / kQkNTiles;
                static_inst[3] = within_head % kQkNTiles;
                static_inst[7] = head;
                static_inst[9] = 0;
            } else {
                static_inst[2] = task - kQkTasks;
                static_inst[3] = 0;
                static_inst[7] = 0;
                static_inst[9] = 1;
            }
        }
        __syncthreads();
        gemma_attn_qk(g, static_inst, smem);
        __syncthreads();
    }
}

// Three-view mixed-QKV and materialized QK share a two-CTA/SM grid.
__global__ __launch_bounds__(kThreads, 2)
void gemma_final_qkv_qk_v3_subgraph(
    GemmaGlobals g, int32_t* barrier, int layer, int seq_len) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kRows = 832;
    constexpr int kQkvMTiles = 13;
    constexpr int kQkvCompactNTiles = 24;
    constexpr int kQkvTasks = kQkvMTiles * kQkvCompactNTiles;
    constexpr int kQkMTiles =
        (kRows + gemma_attn_mat::QkConfig::BM - 1) /
        gemma_attn_mat::QkConfig::BM;
    constexpr int kQkNTiles =
        (kRows + gemma_attn_mat::QkConfig::BN - 1) /
        gemma_attn_mat::QkConfig::BN;
    constexpr int kQkTasks = gemma::kHeads * kQkMTiles * kQkNTiles;
    constexpr int kVtTasks =
        (gemma::kHeadDim / gemma_attn_mat::kVtTile) *
        (kRows / gemma_attn_mat::kVtTile);
    constexpr int kAttentionTasks = kQkTasks + kVtTasks;
    constexpr int kGrid = kVtTasks;
    static_assert(kQkvTasks == 312);
    static_assert(kQkTasks == 392);
    static_assert(kVtTasks == 208);
    static_assert(kAttentionTasks == 600);
    static_assert(kGrid <= 2 * kSmCount);

    int32_t* static_inst = reinterpret_cast<int32_t*>(
        smem + kSmemQkvQkView2Base);
    for (int task = static_cast<int>(blockIdx.x);
         task < kQkvTasks; task += kGrid) {
        const int qkv_m = task / kQkvCompactNTiles;
        const int compact_n = task % kQkvCompactNTiles;
        const int logical_n = compact_n < 16 ? 2 * compact_n : compact_n + 16;
        if (threadIdx.x == 0) {
            static_inst[1] = layer;
            static_inst[2] = qkv_m;
            static_inst[3] = logical_n;
            static_inst[7] = 0;
        }
        __syncthreads();
        gemma_qkv_rope_view2(g, static_inst, smem);
        __syncthreads();
    }
    gemma_subgraph_grid_barrier(barrier, kGrid);

    for (int task = static_cast<int>(blockIdx.x);
         task < kAttentionTasks; task += kGrid) {
        if (threadIdx.x == 0) {
            static_inst[1] = layer;
            static_inst[8] = seq_len;
            if (task < kQkTasks) {
                const int per_head = kQkMTiles * kQkNTiles;
                const int head = task / per_head;
                const int within_head = task % per_head;
                static_inst[2] = within_head / kQkNTiles;
                static_inst[3] = within_head % kQkNTiles;
                static_inst[7] = head;
                static_inst[9] = 0;
            } else {
                static_inst[2] = task - kQkTasks;
                static_inst[3] = 0;
                static_inst[7] = 0;
                static_inst[9] = 1;
            }
        }
        __syncthreads();
        gemma_attn_qk(g, static_inst, smem);
        __syncthreads();
    }
}

#define GEMMA_PHASE_KERNEL(NAME, BODY)                                               \
    __global__ __launch_bounds__(kThreads, 1)                                       \
    void gemma_final_##NAME(GemmaGlobals g, const int32_t* insts, int count) {       \
        extern __shared__ __align__(16) unsigned char smem[];                        \
        const int index = static_cast<int>(blockIdx.x);                              \
        if (index >= count) return;                                                  \
        const int32_t* inst =                                                        \
            insts + static_cast<int64_t>(index) * kInstructionWidth;                 \
        BODY;                                                                        \
    }

GEMMA_PHASE_KERNEL(rmsn_attn, gemma_rms(g, inst, 0))
GEMMA_PHASE_KERNEL(rmsn_ffn, gemma_rms(g, inst, 1))
GEMMA_PHASE_KERNEL(qkv_rope, gemma_qkv_rope(g, inst, smem))
GEMMA_PHASE_KERNEL(qkv_rope_view2, gemma_qkv_rope_view2(g, inst, smem))
GEMMA_PHASE_KERNEL(attn_qk, gemma_attn_qk(g, inst, smem))
GEMMA_PHASE_KERNEL(attn_softmax, gemma_attn_softmax(g, inst))
GEMMA_PHASE_KERNEL(attn_pv, gemma_attn_pv(g, inst, smem))
GEMMA_PHASE_KERNEL(oproj, gemma_oproj(g, inst, smem))
GEMMA_PHASE_KERNEL(gate, gemma_gate(g, inst, smem))
GEMMA_PHASE_KERNEL(up_mul, gemma_up_mul(g, inst, smem))
GEMMA_PHASE_KERNEL(down, gemma_down(g, inst, smem))
GEMMA_PHASE_KERNEL(down_splitk, gemma_down_splitk(g, inst, smem))

namespace {

using Kernel = void (*)(GemmaGlobals, const int32_t*, int);
struct PhaseKernel { Kernel kernel; int smem; };

PhaseKernel phase_kernel_for(int opcode) {
    switch (opcode) {
        case GEMMA_RMSN_ATTN: return {gemma_final_rmsn_attn, kSmemNone};
        case GEMMA_RMSN_FFN: return {gemma_final_rmsn_ffn, kSmemNone};
        case GEMMA_QKV_ROPE: return {gemma_final_qkv_rope, kSmemGemm};
        case GEMMA_ATTN_QK: return {gemma_final_attn_qk, kSmemQk};
        case GEMMA_ATTN_SOFTMAX: return {gemma_final_attn_softmax, kSmemNone};
        case GEMMA_ATTN_PV: return {gemma_final_attn_pv, kSmemPv};
        case GEMMA_OPROJ: return {gemma_final_oproj, kSmemGemm};
        case GEMMA_GATE: return {gemma_final_gate, kSmemDual};
        case GEMMA_UP_MUL: return {gemma_final_up_mul, kSmemNone};
        case GEMMA_DOWN: return {gemma_final_down, kSmemGemm};
        case GEMMA_DOWN_SPLITK: return {gemma_final_down_splitk, kSmemGemm};
        default: return {nullptr, 0};
    }
}

constexpr int kOpcodes[] = {
    GEMMA_RMSN_ATTN, GEMMA_RMSN_FFN, GEMMA_QKV_ROPE, GEMMA_ATTN_QK,
    GEMMA_ATTN_SOFTMAX, GEMMA_ATTN_PV, GEMMA_OPROJ, GEMMA_GATE,
    GEMMA_UP_MUL, GEMMA_DOWN, GEMMA_DOWN_SPLITK,
};
GemmaGlobals globals{};
bool bound = false;
int pv_oproj_v3_ctas_per_sm = 0;
int pv_oproj_v2_ctas_per_sm = 0;
int pv_oproj_v1_ctas_per_sm = 0;
int qkv_qk_v2_ctas_per_sm = 0;
int qkv_qk_v3_ctas_per_sm = 0;

}  // namespace

void gemma_bind(
    torch::Tensor encoder_x, torch::Tensor rms_factor,
    torch::Tensor qkv_w, torch::Tensor qkv_buf, torch::Tensor rope,
    torch::Tensor k_cache, torch::Tensor v_cache,
    torch::Tensor attn_scores, torch::Tensor attn_vt, torch::Tensor attn_out,
    torch::Tensor oproj_w, torch::Tensor gate_w, torch::Tensor up_w,
    torch::Tensor gate_buf, torch::Tensor up_buf, torch::Tensor down_w,
    torch::Tensor barriers, torch::Tensor instructions) {
    check_cuda_contiguous(encoder_x, "encoder_x");
    globals = GemmaGlobals{
        ptr<nv_bfloat16>(encoder_x), ptr<float>(rms_factor),
        ptr<nv_bfloat16>(qkv_w), ptr<nv_bfloat16>(qkv_buf),
        ptr<nv_bfloat16>(rope), ptr<nv_bfloat16>(k_cache),
        ptr<nv_bfloat16>(v_cache), ptr<nv_bfloat16>(attn_scores),
        ptr<nv_bfloat16>(attn_vt), ptr<nv_bfloat16>(attn_out),
        ptr<nv_bfloat16>(oproj_w), ptr<nv_bfloat16>(gate_w),
        ptr<nv_bfloat16>(up_w), ptr<nv_bfloat16>(gate_buf),
        ptr<nv_bfloat16>(up_buf), ptr<nv_bfloat16>(down_w),
        ptr<int32_t>(barriers), ptr<int32_t>(instructions),
        static_cast<int>(instructions.size(2)),
        static_cast<int>(encoder_x.size(-2)),
    };
    for (int opcode : kOpcodes) {
        const PhaseKernel phase_kernel = phase_kernel_for(opcode);
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            phase_kernel.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, phase_kernel.smem));
    }
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_down_view1_eight_warp,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        GemmaDownView1EightWarp::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_oproj_view1,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        GemmaOprojView1::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_qkv_rope_view2,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemQkvView2));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_pv_oproj_v3_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        GemmaOprojWideN::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_pv_oproj_v2_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemPv));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_pv_oproj_v1_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemPvOprojView1));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_qkv_qk_v2_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemQkvQkView2));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        gemma_final_qkv_qk_v3_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSmemQkvQkView2));
    int pv_active = 0;
    int oproj_active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &pv_active, gemma_final_attn_pv, kThreads, kSmemPv));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &oproj_active, gemma_final_oproj, kThreads, kSmemGemm));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &pv_oproj_v3_ctas_per_sm, gemma_final_pv_oproj_v3_subgraph,
        kThreads, GemmaOprojWideN::kShmSize));
    TORCH_CHECK(pv_active == 2 && oproj_active == 2 &&
                    pv_oproj_v3_ctas_per_sm == 2,
                "three-view PV-OProj residency changed: PV=", pv_active,
                " OProj=", oproj_active,
                " fused=", pv_oproj_v3_ctas_per_sm);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &pv_oproj_v2_ctas_per_sm, gemma_final_pv_oproj_v2_subgraph,
        kThreads, kSmemPv));
    TORCH_CHECK(pv_active == 2 && oproj_active == 2 &&
                    pv_oproj_v2_ctas_per_sm >= 2,
                "two-view PV-OProj residency changed: PV=", pv_active,
                " OProj=", oproj_active,
                " fused=", pv_oproj_v2_ctas_per_sm);
    int oproj_v1_active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &oproj_v1_active, gemma_final_oproj_view1,
        GemmaOprojView1::kNumThreads, GemmaOprojView1::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &pv_oproj_v1_ctas_per_sm, gemma_final_pv_oproj_v1_subgraph,
        gemma_attn_mat::PvConfig::kNumThreads, kSmemPvOprojView1));
    TORCH_CHECK(pv_active == 2 && oproj_v1_active == 2 &&
                    pv_oproj_v1_ctas_per_sm == 2,
                "one-view PV-OProj residency changed: PV=", pv_active,
                " OProj=", oproj_v1_active,
                " fused=", pv_oproj_v1_ctas_per_sm);
    int qkv_v2_active = 0;
    int qk_active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &qkv_v2_active, gemma_final_qkv_rope_view2,
        kThreads, kSmemQkvView2));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &qk_active, gemma_final_attn_qk, kThreads, kSmemQk));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &qkv_qk_v2_ctas_per_sm, gemma_final_qkv_qk_v2_subgraph,
        kThreads, kSmemQkvQkView2));
    TORCH_CHECK(qkv_v2_active == 2 && qk_active == 2 &&
                    qkv_qk_v2_ctas_per_sm == 2,
                "two-view QKV-QK residency changed: QKV=", qkv_v2_active,
                " QK=", qk_active, " fused=", qkv_qk_v2_ctas_per_sm);
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &qkv_qk_v3_ctas_per_sm, gemma_final_qkv_qk_v3_subgraph,
        kThreads, kSmemQkvQkView2));
    TORCH_CHECK(qkv_v2_active == 2 && qk_active == 2 &&
                    qkv_qk_v3_ctas_per_sm == 2,
                "three-view QKV-QK residency changed: mixed QKV=",
                qkv_v2_active, " QK=", qk_active,
                " fused=", qkv_qk_v3_ctas_per_sm);
    bound = true;
}

void gemma_run_phase(torch::Tensor phase, int64_t opcode64) {
    TORCH_CHECK(bound, "call bind() before run_phase()");
    check_cuda_contiguous(phase, "phase");
    TORCH_CHECK(phase.scalar_type() == torch::kInt32, "phase must be int32");
    TORCH_CHECK(phase.dim() == 2 && phase.size(1) == kInstructionWidth,
                "phase must be [count, ", kInstructionWidth, "]");
    const PhaseKernel phase_kernel = phase_kernel_for(static_cast<int>(opcode64));
    TORCH_CHECK(phase_kernel.kernel != nullptr, "opcode has no final Gemma phase_kernel");
    const int count = static_cast<int>(phase.size(0));
    if (count == 0) return;
    phase_kernel.kernel<<<count, kThreads, phase_kernel.smem, current_stream()>>>(
        globals, ptr<int32_t>(phase), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemma_run_down_view1_eight_warp(int64_t layer64) {
    TORCH_CHECK(bound, "call bind() before run_down_view1_eight_warp()");
    TORCH_CHECK(globals.seq == 320,
                "one-view down kernel requires seq_pad=320, got ", globals.seq);
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < gemma::kLayers, "invalid layer ", layer);
    constexpr int kTasks =
        (320 / GemmaDownView1EightWarp::BM) *
        (gemma::kWidth / GemmaDownView1EightWarp::BN);
    gemma_final_down_view1_eight_warp<<<
        kTasks, GemmaDownView1EightWarp::kNumThreads,
        GemmaDownView1EightWarp::kShmSize,
        current_stream()>>>(globals, layer);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemma_run_qkv_view2(torch::Tensor phase) {
    TORCH_CHECK(bound, "call bind() before run_qkv_view2()");
    TORCH_CHECK(globals.seq == 576,
                "two-view QKV kernel requires seq_pad=576, got ", globals.seq);
    check_cuda_contiguous(phase, "phase");
    TORCH_CHECK(phase.scalar_type() == torch::kInt32, "phase must be int32");
    TORCH_CHECK(phase.dim() == 2 && phase.size(1) == kInstructionWidth,
                "phase must be [count, ", kInstructionWidth, "]");
    const int count = static_cast<int>(phase.size(0));
    gemma_final_qkv_rope_view2<<<
        count, kThreads, kSmemQkvView2, current_stream()>>>(
            globals, ptr<int32_t>(phase), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemma_run_pv_oproj_v3_subgraph(
    torch::Tensor ready, int64_t layer64, bool prefetch_oproj) {
    TORCH_CHECK(bound, "call bind() before run_pv_oproj_v3_subgraph()");
    TORCH_CHECK(globals.seq == 832,
                "PV-OProj subgraph is specialized for three-view seq_pad=832; got ",
                globals.seq);
    check_cuda_contiguous(ready, "pv_oproj_ready");
    TORCH_CHECK(ready.scalar_type() == torch::kInt32 && ready.numel() >= 13,
                "pv_oproj_ready must contain at least 13 int32 counters");
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < gemma::kLayers - 1,
                "invalid full Gemma layer ", layer);
    constexpr int kGrid =
        gemma::kHeads * (832 / gemma_attn_mat::PvConfig::BM) *
        (gemma::kHeadDim / gemma_attn_mat::PvConfig::BN);
    TORCH_CHECK(kGrid <= pv_oproj_v3_ctas_per_sm * kSmCount,
                "PV-OProj subgraph grid is not fully resident");
    gemma_final_pv_oproj_v3_subgraph<<<
        kGrid, kThreads, GemmaOprojWideN::kShmSize, current_stream()>>>(
            globals, ptr<int32_t>(ready), layer, prefetch_oproj);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemma_run_pv_oproj_v2_subgraph(
    torch::Tensor ready, int64_t layer64, bool prefetch_oproj) {
    TORCH_CHECK(bound, "call bind() before run_pv_oproj_v2_subgraph()");
    TORCH_CHECK(globals.seq == 576,
                "PV-OProj subgraph is specialized for two-view seq_pad=576; got ",
                globals.seq);
    check_cuda_contiguous(ready, "pv_oproj_ready");
    TORCH_CHECK(ready.scalar_type() == torch::kInt32 && ready.numel() >= 9,
                "pv_oproj_ready must contain at least 9 int32 counters");
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < gemma::kLayers - 1,
                "invalid full Gemma layer ", layer);
    constexpr int kGrid =
        gemma::kHeads * (576 / gemma_attn_mat::PvConfig::BM) *
        (gemma::kHeadDim / gemma_attn_mat::PvConfig::BN);
    TORCH_CHECK(kGrid <= pv_oproj_v2_ctas_per_sm * kSmCount,
                "PV-OProj subgraph grid is not fully resident");
    gemma_final_pv_oproj_v2_subgraph<<<
        kGrid, kThreads, kSmemPv, current_stream()>>>(
            globals, ptr<int32_t>(ready), layer, prefetch_oproj);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemma_run_pv_oproj_v1_subgraph(
    torch::Tensor ready, int64_t layer64, bool prefetch_oproj) {
    TORCH_CHECK(bound, "call bind() before run_pv_oproj_v1_subgraph()");
    TORCH_CHECK(globals.seq == 320,
                "PV-OProj subgraph requires seq_pad=320; got ", globals.seq);
    check_cuda_contiguous(ready, "pv_oproj_ready");
    TORCH_CHECK(ready.scalar_type() == torch::kInt32 && ready.numel() >= 10,
                "pv_oproj_ready must contain at least 10 int32 counters");
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < gemma::kLayers - 1,
                "invalid full Gemma layer ", layer);
    constexpr int kGrid =
        (320 / GemmaOprojView1::BM) *
        (gemma::kWidth / GemmaOprojView1::BN);
    TORCH_CHECK(kGrid <= pv_oproj_v1_ctas_per_sm * kSmCount,
                "one-view PV-OProj grid is not fully resident");
    gemma_final_pv_oproj_v1_subgraph<<<
        kGrid, gemma_attn_mat::PvConfig::kNumThreads,
        kSmemPvOprojView1, current_stream()>>>(
            globals, ptr<int32_t>(ready), layer, prefetch_oproj);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemma_run_qkv_qk_v2_subgraph(
    torch::Tensor barrier, int64_t layer64, int64_t seq_len64) {
    TORCH_CHECK(bound, "call bind() before run_qkv_qk_v2_subgraph()");
    TORCH_CHECK(globals.seq == 576,
                "QKV-QK subgraph is specialized for two-view seq_pad=576; got ",
                globals.seq);
    check_cuda_contiguous(barrier, "gemma_subgraph_barrier");
    TORCH_CHECK(barrier.scalar_type() == torch::kInt32 && barrier.numel() >= 2,
                "gemma_subgraph_barrier must contain at least two int32 words");
    const int layer = static_cast<int>(layer64);
    const int seq_len = static_cast<int>(seq_len64);
    TORCH_CHECK(layer >= 0 && layer < gemma::kLayers - 1,
                "invalid full Gemma layer ", layer);
    TORCH_CHECK(seq_len > 0 && seq_len <= globals.seq,
                "invalid logical encoder length ", seq_len);
    constexpr int kGrid = 216;
    TORCH_CHECK(kGrid <= qkv_qk_v2_ctas_per_sm * kSmCount,
                "QKV-QK subgraph grid is not fully resident");
    gemma_final_qkv_qk_v2_subgraph<<<
        kGrid, kThreads, kSmemQkvQkView2, current_stream()>>>(
            globals, ptr<int32_t>(barrier), layer, seq_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gemma_run_qkv_qk_v3_subgraph(
    torch::Tensor barrier, int64_t layer64, int64_t seq_len64) {
    TORCH_CHECK(bound, "call bind() before run_qkv_qk_v3_subgraph()");
    TORCH_CHECK(globals.seq == 832,
                "QKV-QK subgraph requires seq_pad=832; got ", globals.seq);
    check_cuda_contiguous(barrier, "gemma_subgraph_barrier");
    TORCH_CHECK(barrier.scalar_type() == torch::kInt32 && barrier.numel() >= 2,
                "gemma_subgraph_barrier must contain at least two int32 words");
    const int layer = static_cast<int>(layer64);
    const int seq_len = static_cast<int>(seq_len64);
    TORCH_CHECK(layer >= 0 && layer < gemma::kLayers - 1,
                "invalid full Gemma layer ", layer);
    TORCH_CHECK(seq_len > 0 && seq_len <= globals.seq,
                "invalid logical encoder length ", seq_len);
    constexpr int kGrid = 208;
    TORCH_CHECK(kGrid <= qkv_qk_v3_ctas_per_sm * kSmCount,
                "three-view QKV-QK grid is not fully resident");
    gemma_final_qkv_qk_v3_subgraph<<<
        kGrid, kThreads, kSmemQkvQkView2, current_stream()>>>(
            globals, ptr<int32_t>(barrier), layer, seq_len);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bind_module(py::module_& module) {
    module.doc() = "Pi0 A100 production Gemma kernels";
    module.def("bind", &gemma_bind);
    module.def("run_phase", &gemma_run_phase);
    module.def("run_down_view1",
               &gemma_run_down_view1_eight_warp);
    module.def("run_qkv_view2", &gemma_run_qkv_view2);
    module.def("run_pv_oproj_view3",
               &gemma_run_pv_oproj_v3_subgraph,
               py::arg("ready"), py::arg("layer"),
               py::arg("prefetch_oproj") = true);
    module.def("run_pv_oproj_view2",
               &gemma_run_pv_oproj_v2_subgraph,
               py::arg("ready"), py::arg("layer"),
               py::arg("prefetch_oproj") = true);
    module.def("run_pv_oproj_view1",
               &gemma_run_pv_oproj_v1_subgraph,
               py::arg("ready"), py::arg("layer"),
               py::arg("prefetch_oproj") = false);
    module.def("run_qkv_qk_view2",
               &gemma_run_qkv_qk_v2_subgraph,
               py::arg("barrier"), py::arg("layer"), py::arg("seq_len"));
    module.def("run_qkv_qk_view3",
               &gemma_run_qkv_qk_v3_subgraph,
               py::arg("barrier"), py::arg("layer"), py::arg("seq_len"));
}

#ifndef PI0_MK_MODULE
#error "PI0_MK_MODULE must be provided by the route Makefile"
#endif

PYBIND11_MODULE(PI0_MK_MODULE, m) {
    bind_module(m);
}
