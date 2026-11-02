// Pi0 SigLIP kernels for A100/A800.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/host_utils.cuh"
#include "common/instruction_list.cuh"
#include "common/matmul_pipeline.cuh"
#include "common/runtime.cuh"
#include "common/siglip_globals.cuh"

#include "backbone/siglip/siglip_ln.cu"
#include "backbone/siglip/siglip_gemm.cu"
#include "backbone/siglip/siglip_attn.cu"

namespace py = pybind11;
using namespace pi0_a100;

static constexpr int kSmemNone = 16;
static constexpr int kSmemGemm = SiglipGemm64::kShmSize;
static constexpr int kSmemAttn = sizeof(siglip_attn_tc::AttentionSmem);

#define SIGLIP_PHASE_KERNEL(NAME, BODY)                                      \
    extern "C" __global__ __launch_bounds__(kThreads, 1)                     \
    void siglip_phase_##NAME(                                                 \
        SiglipGlobals g, const int32_t* insts, int count) {                   \
        extern __shared__ __align__(16) unsigned char smem[];                 \
        (void)smem;                                                           \
        const int index = static_cast<int>(blockIdx.x);                       \
        if (index >= count) return;                                           \
        const int32_t* inst =                                                 \
            insts + static_cast<int64_t>(index) * kInstructionWidth;          \
        BODY;                                                                 \
    }

SIGLIP_PHASE_KERNEL(ln_attn, siglip_stats(g, inst))
SIGLIP_PHASE_KERNEL(ln_ffn, siglip_stats(g, inst))
SIGLIP_PHASE_KERNEL(attn, siglip_attention(g, inst, smem))
SIGLIP_PHASE_KERNEL(patch, siglip_patch(g, inst, smem))
SIGLIP_PHASE_KERNEL(final_norm, siglip_final_norm(g, inst))
SIGLIP_PHASE_KERNEL(projector, siglip_projector(g, inst, smem))

template <typename Config, int Which, bool NMajor = false>
__device__ __forceinline__ void siglip_gemm_body(
    SiglipGlobals g, int layer, int m_tiles, int n_tiles,
    unsigned char* smem, bool b_stages_preloaded = false) {
    const int tasks = m_tiles * n_tiles;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int tm = NMajor ? task % m_tiles : task / n_tiles;
        const int tn = NMajor ? task / m_tiles : task % n_tiles;
        if constexpr (Which == SIGLIP_QKV) {
            SiglipLnFoldPackedEpilogue epi{
                g.ln_stats, g.ln_fold_qkv, g.seq, siglip::kQkvWidth,
                layer, g.seq == 256 ? 40 : 20,
                g.seq == 256 ? 41 : 21, 0, false};
            const nv_bfloat16* weight = g.qkv_w +
                static_cast<int64_t>(layer) * siglip::kQkvWidth * siglip::kWidthPad;
            mk_gemm::gemm_unified_tile<Config>(
                g.vision_x, weight, g.qkv_buf, g.seq, siglip::kQkvWidth,
                siglip::kWidth, tm, tn, tn + 1, epi, smem, false,
                siglip::kWidthPad, siglip::kWidthPad, siglip::kQkvWidth);
        } else if constexpr (Which == SIGLIP_OPROJ) {
            const nv_bfloat16* weight = g.oproj_w +
                static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kWidthPad;
            const nv_bfloat16* bias =
                g.oproj_b + static_cast<int64_t>(layer) * siglip::kWidthPad;
            mk_gemm::EpilogueBiasResidual<nv_bfloat16> epi{
                bias, g.vision_x, siglip::kWidthPad};
            mk_gemm::gemm_unified_tile<Config>(
                g.attn_out, weight, g.vision_x, g.seq, siglip::kWidthPad,
                siglip::kWidthPad, tm, tn, tn + 1, epi, smem);
        } else if constexpr (Which == SIGLIP_FC1) {
            const nv_bfloat16* weight = g.fc1_w +
                static_cast<int64_t>(layer) * siglip::kMlpWidth * siglip::kWidthPad;
            SiglipLnFoldPackedEpilogue epi{
                g.ln_stats, g.ln_fold_fc1, g.seq, siglip::kMlpWidth,
                layer, 0, 10, 0, true};
            mk_gemm::gemm_unified_tile<Config>(
                g.vision_x, weight, g.fc1_buf, g.seq, siglip::kMlp,
                siglip::kWidth, tm, tn, tn + 1, epi, smem, false,
                siglip::kWidthPad, siglip::kWidthPad, siglip::kMlpWidth);
        } else if constexpr (Which == SIGLIP_FC2) {
            const nv_bfloat16* bias =
                g.fc2_b + static_cast<int64_t>(layer) * siglip::kWidthPad;
            const nv_bfloat16* weight = g.fc2_w +
                static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kMlpWidth;
            mk_gemm::EpilogueBiasResidual<nv_bfloat16> epi{
                bias, g.vision_x, siglip::kWidthPad};
            mk_gemm::gemm_unified_tile<Config>(
                g.fc1_buf, weight, g.vision_x, g.seq, siglip::kWidth,
                siglip::kMlp, tm, tn, tn + 1, epi, smem,
                b_stages_preloaded,
                siglip::kMlpWidth, siglip::kMlpWidth, siglip::kWidthPad);
        }
    }
}

template <typename Config, int Which, bool NMajor = false>
__global__ __launch_bounds__(Config::kNumThreads, 1) void siglip_gemm_kernel(
    SiglipGlobals g, int layer, int m_tiles, int n_tiles) {
    extern __shared__ __align__(16) unsigned char smem[];
    siglip_gemm_body<Config, Which, NMajor>(
        g, layer, m_tiles, n_tiles, smem);
}

using SiglipQkvView1 = mk_gemm::GemmConfig<64, 128, 32, 4, 2, 2, 2>;
using SiglipQkvView2 = mk_gemm::GemmConfig<64, 160, 32, 4, 2, 2, 2>;
using SiglipQkvView3 = mk_gemm::GemmConfig<128, 128, 32, 3, 2, 2, 2>;
using SiglipFc1View2 = mk_gemm::GemmConfig<64, 64, 64, 3, 2, 2, 1>;
using SiglipFc1View3 = mk_gemm::GemmConfig<128, 128, 32, 4, 2, 2, 2>;
using SiglipFc2View2 = mk_gemm::GemmConfig<64, 64, 64, 3, 2, 2, 1>;
using SiglipFc2View3 = mk_gemm::GemmConfig<128, 64, 64, 3, 2, 2, 1>;
using SiglipFc1View1 = mk_gemm::GemmConfig<64, 128, 32, 4, 2, 2, 2>;
using SiglipFc2View1 = mk_gemm::GemmConfig<64, 64, 64, 4, 2, 2, 1>;
using SiglipOprojView1 = mk_gemm::GemmConfig<64, 64, 64, 4, 2, 2, 1>;
using SiglipOprojView2 = mk_gemm::GemmConfig<64, 64, 64, 5, 2, 2, 1>;
using SiglipOprojView3 = mk_gemm::GemmConfig<96, 64, 64, 3, 2, 2, 1>;

template <int A, int B>
struct SiglipConstMax {
    static constexpr int value = A > B ? A : B;
};

__device__ __forceinline__ void siglip_subgraph_grid_barrier(
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
            while (atomicAdd(reinterpret_cast<unsigned*>(words + 1), 0u) == epoch) {
                __nanosleep(32);
            }
        }
    }
    __syncthreads();
}

__global__ __launch_bounds__(SiglipFc1View3::kNumThreads, 2)
void siglip_mlp_v3_subgraph(
    SiglipGlobals g, int32_t* barrier, int layer, bool prefetch_fc2) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kRows = 768;
    constexpr int kMTiles = kRows / SiglipFc1View3::BM;
    constexpr int kFc1NTiles =
        (siglip::kMlp + SiglipFc1View3::BN - 1) / SiglipFc1View3::BN;
    constexpr int kFc2NTiles = siglip::kWidth / SiglipFc2View3::BN;
    constexpr int kGrid = kMTiles * kFc1NTiles;
    static_assert(kGrid == 204);
    static_assert(SiglipFc1View3::kNumThreads == SiglipFc2View3::kNumThreads);

    siglip_gemm_body<SiglipFc1View3, SIGLIP_FC1>(
        g, layer, kMTiles, kFc1NTiles, smem);
    if (prefetch_fc2 &&
        static_cast<int>(blockIdx.x) < kMTiles * kFc2NTiles) {
        const int fc2_tile_n = static_cast<int>(blockIdx.x) % kFc2NTiles;
        const nv_bfloat16* fc2_weight = g.fc2_w +
            static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kMlpWidth;
        mk_gemm::gemm_unified_preload_b_stages<SiglipFc2View3>(
            fc2_weight, siglip::kWidth, siglip::kMlp, fc2_tile_n, smem,
            siglip::kMlpWidth);
    }
    siglip_subgraph_grid_barrier(barrier, kGrid);
    siglip_gemm_body<SiglipFc2View3, SIGLIP_FC2>(
        g, layer, kMTiles, kFc2NTiles, smem, prefetch_fc2);
}

__global__ __launch_bounds__(SiglipFc1View1::kNumThreads, 2)
void siglip_mlp_v1_subgraph(
    SiglipGlobals g, int32_t* barrier, int layer, bool prefetch_fc2) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kRows = 256;
    constexpr int kMTiles = kRows / SiglipFc1View1::BM;
    constexpr int kFc1NTiles =
        (siglip::kMlp + SiglipFc1View1::BN - 1) / SiglipFc1View1::BN;
    constexpr int kFc2NTiles = siglip::kWidth / SiglipFc2View1::BN;
    constexpr int kGrid = kMTiles * kFc1NTiles;
    static_assert(kGrid == 136);
    static_assert(SiglipFc1View1::kNumThreads == SiglipFc2View1::kNumThreads);

    siglip_gemm_body<SiglipFc1View1, SIGLIP_FC1>(
        g, layer, kMTiles, kFc1NTiles, smem);
    if (prefetch_fc2 &&
        static_cast<int>(blockIdx.x) < kMTiles * kFc2NTiles) {
        const int fc2_tile_n = static_cast<int>(blockIdx.x) % kFc2NTiles;
        const nv_bfloat16* fc2_weight = g.fc2_w +
            static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kMlpWidth;
        mk_gemm::gemm_unified_preload_b_stages<SiglipFc2View1>(
            fc2_weight, siglip::kWidth, siglip::kMlp, fc2_tile_n, smem,
            siglip::kMlpWidth);
    }
    siglip_subgraph_grid_barrier(barrier, kGrid);
    siglip_gemm_body<SiglipFc2View1, SIGLIP_FC2>(
        g, layer, kMTiles, kFc2NTiles, smem, prefetch_fc2);
}

__global__ __launch_bounds__(SiglipFc1View2::kNumThreads, 3)
void siglip_mlp_v2_subgraph(
    SiglipGlobals g, int32_t* barrier, int layer, bool prefetch_fc2) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kRows = 512;
    constexpr int kMTiles = kRows / SiglipFc1View2::BM;
    constexpr int kFc1NTiles =
        (siglip::kMlp + SiglipFc1View2::BN - 1) / SiglipFc1View2::BN;
    constexpr int kFc2NTiles = siglip::kWidth / SiglipFc2View2::BN;
    constexpr int kGrid = (kMTiles * kFc1NTiles) / 2;
    static_assert(kMTiles * kFc1NTiles == 544);
    static_assert(kGrid == 272);
    static_assert(kMTiles * kFc2NTiles == 144);
    static_assert(SiglipFc1View2::kNumThreads == SiglipFc2View2::kNumThreads);

    siglip_gemm_body<SiglipFc1View2, SIGLIP_FC1, true>(
        g, layer, kMTiles, kFc1NTiles, smem);
    if (prefetch_fc2 &&
        static_cast<int>(blockIdx.x) < kMTiles * kFc2NTiles) {
        const int fc2_tile_n = static_cast<int>(blockIdx.x) % kFc2NTiles;
        const nv_bfloat16* fc2_weight = g.fc2_w +
            static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kMlpWidth;
        mk_gemm::gemm_unified_preload_b_stages<SiglipFc2View2>(
            fc2_weight, siglip::kWidth, siglip::kMlp, fc2_tile_n, smem,
            siglip::kMlpWidth);
    }
    siglip_subgraph_grid_barrier(barrier, kGrid);
    siglip_gemm_body<SiglipFc2View2, SIGLIP_FC2>(
        g, layer, kMTiles, kFc2NTiles, smem, prefetch_fc2);
}

template <int Rows, int Views, typename Qkv, typename Oproj>
__global__ __launch_bounds__(Qkv::kNumThreads, 2)
void siglip_qkv_attn_oproj_subgraph(
    SiglipGlobals g, int32_t* barrier, int32_t* oproj_ready,
    int layer, bool prefetch_oproj) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kQkvMTiles = Rows / Qkv::BM;
    constexpr int kQkvNTiles =
        (siglip::kQkvWidth + Qkv::BN - 1) / Qkv::BN;
    constexpr int kQkvTasks = kQkvMTiles * kQkvNTiles;
    constexpr int kQueryRows = siglip_attn_tc::kBlockM;
    constexpr int kQueryTilesPerView = 256 / kQueryRows;
    constexpr int kAttnTasks = Views * kQueryTilesPerView * siglip::kHeads;
    constexpr int kOprojMTiles = (Rows + Oproj::BM - 1) / Oproj::BM;
    constexpr int kOprojNTiles = siglip::kWidthPad / Oproj::BN;
    constexpr int kOprojTasks = kOprojMTiles * kOprojNTiles;
    constexpr int kGrid0 = SiglipConstMax<kQkvTasks, kAttnTasks>::value;
    constexpr int kGrid = SiglipConstMax<kGrid0, kOprojTasks>::value;
    static_assert(Rows == Views * 256);
    static_assert(Qkv::kNumThreads == siglip_attn_tc::kThreadsTc);
    static_assert(Qkv::kNumThreads == Oproj::kNumThreads);
    static_assert(kGrid <= 2 * kSmCount);

    siglip_gemm_body<Qkv, SIGLIP_QKV>(
        g, layer, kQkvMTiles, kQkvNTiles, smem);
    siglip_subgraph_grid_barrier(barrier, kGrid);

    const int task = static_cast<int>(blockIdx.x);
    if (task < kAttnTasks) {
        const int per_view = kQueryTilesPerView * siglip::kHeads;
        const int view = task / per_view;
        const int within_view = task % per_view;
        const int query_tile = within_view / siglip::kHeads;
        const int head = within_view % siglip::kHeads;
        int32_t* static_inst = reinterpret_cast<int32_t*>(
            smem + sizeof(siglip_attn_tc::AttentionSmem));
        if (threadIdx.x == 0) {
            static_inst[2] = view;
            static_inst[3] = query_tile;
            static_inst[4] = head;
        }
        __syncthreads();
        siglip_attention(g, static_inst, smem);
        __syncthreads();
        if (threadIdx.x == 0) {
            const int row0 = view * 256 + query_tile * kQueryRows;
            const int first_m = row0 / Oproj::BM;
            const int last_m = (row0 + kQueryRows - 1) / Oproj::BM;
            for (int m = first_m; m <= last_m; ++m) {
                ampere_barrier::publish(oproj_ready + m);
            }
        }
    }

    if (task < kOprojTasks) {
        const int oproj_m = task / kOprojNTiles;
        const int oproj_n = task % kOprojNTiles;
        const nv_bfloat16* weight = g.oproj_w +
            static_cast<int64_t>(layer) * siglip::kWidthPad * siglip::kWidthPad;
        if (prefetch_oproj) {
            mk_gemm::gemm_unified_preload_b_stages<Oproj>(
                weight, siglip::kWidthPad, siglip::kWidthPad, oproj_n, smem);
        }
        const int row0 = oproj_m * Oproj::BM;
        const int first_query = row0 / kQueryRows;
        const int last_query =
            (min(Rows, row0 + Oproj::BM) - 1) / kQueryRows;
        const int query_count = last_query - first_query + 1;
        if (threadIdx.x == 0) {
            ampere_barrier::wait(
                oproj_ready + oproj_m,
                static_cast<uint32_t>(
                    (layer + 1) * siglip::kHeads * query_count),
#ifndef NDEBUG
                task, 0, SIGLIP_OPROJ, oproj_m * sizeof(int32_t),
                4000000000ULL,
#endif
                0);
        }
        __syncthreads();

        mk_gemm::EpilogueBiasResidual<nv_bfloat16> epi{
            g.oproj_b + static_cast<int64_t>(layer) * siglip::kWidthPad,
            g.vision_x, siglip::kWidthPad};
        mk_gemm::gemm_unified_tile<Oproj>(
            g.attn_out, weight, g.vision_x, Rows, siglip::kWidthPad,
            siglip::kWidthPad, oproj_m, oproj_n, oproj_n + 1, epi, smem,
            prefetch_oproj);
    }
}

namespace {

using PhaseKernelFn = void (*)(SiglipGlobals, const int32_t*, int);

struct PhaseKernel {
    PhaseKernelFn kernel;
    int smem;
};

PhaseKernel phase_kernel_for(int opcode) {
    switch (opcode) {
        case SIGLIP_LN_ATTN: return {siglip_phase_ln_attn, kSmemNone};
        case SIGLIP_LN_FFN: return {siglip_phase_ln_ffn, kSmemNone};
        case SIGLIP_ATTN: return {siglip_phase_attn, kSmemAttn};
        case SIGLIP_PATCH: return {siglip_phase_patch, kSmemGemm};
        case SIGLIP_FINAL_NORM: return {siglip_phase_final_norm, kSmemNone};
        case SIGLIP_PROJECTOR: return {siglip_phase_projector, kSmemGemm};
        default: return {nullptr, 0};
    }
}

constexpr int kOpcodes[] = {
    SIGLIP_LN_ATTN, SIGLIP_LN_FFN, SIGLIP_ATTN, SIGLIP_PATCH,
    SIGLIP_FINAL_NORM, SIGLIP_PROJECTOR,
};

SiglipGlobals g_globals{};
bool g_bound = false;
int g_mlp_v3_ctas_per_sm = 0;
int g_mlp_v1_ctas_per_sm = 0;
int g_mlp_v2_ctas_per_sm = 0;
int g_attention_chain_ctas_per_sm[3] = {0, 0, 0};

template <typename Config, int Which, bool NMajor = false>
void configure_gemm_kernel() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        siglip_gemm_kernel<Config, Which, NMajor>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, Config::kShmSize));
}

template <typename Qkv, typename Oproj>
constexpr int attention_chain_smem() {
    constexpr int attention_bytes =
        sizeof(siglip_attn_tc::AttentionSmem) +
        kInstructionWidth * sizeof(int32_t);
    constexpr int gemm_bytes =
        Qkv::kShmSize > Oproj::kShmSize ? Qkv::kShmSize : Oproj::kShmSize;
    return gemm_bytes > attention_bytes ? gemm_bytes : attention_bytes;
}

template <int Rows, int Views, typename Qkv, typename Oproj>
void configure_attention_chain(int slot) {
    auto fused =
        siglip_qkv_attn_oproj_subgraph<Rows, Views, Qkv, Oproj>;
    constexpr int smem_bytes = attention_chain_smem<Qkv, Oproj>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        fused, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    int qkv_active = 0;
    int attn_active = 0;
    int oproj_active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &qkv_active, siglip_gemm_kernel<Qkv, SIGLIP_QKV>,
        Qkv::kNumThreads, Qkv::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &attn_active, siglip_phase_attn, kThreads, kSmemAttn));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &oproj_active, siglip_gemm_kernel<Oproj, SIGLIP_OPROJ>,
        Oproj::kNumThreads, Oproj::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &g_attention_chain_ctas_per_sm[slot], fused,
        Qkv::kNumThreads, smem_bytes));
    TORCH_CHECK(
        qkv_active == 2 && attn_active == 2 && oproj_active == 2 &&
            g_attention_chain_ctas_per_sm[slot] == 2,
        Views, "-view attention chain residency changed: QKV=", qkv_active,
        " attention=", attn_active, " OProj=", oproj_active,
        " fused=", g_attention_chain_ctas_per_sm[slot]);
}

}  // namespace

void siglip_bind(
    torch::Tensor vision_x,
    torch::Tensor siglip_x_normed,
    torch::Tensor ln_gamma,
    torch::Tensor ln_beta,
    torch::Tensor qkv_w,
    torch::Tensor qkv_b,
    torch::Tensor qkv_buf,
    torch::Tensor attn_out,
    torch::Tensor oproj_w,
    torch::Tensor oproj_b,
    torch::Tensor fc1_w,
    torch::Tensor fc1_b,
    torch::Tensor fc1_buf,
    torch::Tensor fc2_w,
    torch::Tensor fc2_b,
    torch::Tensor patch_in,
    torch::Tensor patch_w,
    torch::Tensor patch_b,
    torch::Tensor proj_w,
    torch::Tensor proj_b,
    torch::Tensor proj_out,
    torch::Tensor ln_stats,
    torch::Tensor ln_fold_qkv,
    torch::Tensor ln_fold_fc1,
    torch::Tensor Bar,
    torch::Tensor instructions) {
    check_cuda_contiguous(vision_x, "vision_x");
    g_globals = SiglipGlobals{
        ptr<nv_bfloat16>(vision_x), ptr<nv_bfloat16>(siglip_x_normed),
        ptr<nv_bfloat16>(ln_gamma), ptr<nv_bfloat16>(ln_beta),
        ptr<nv_bfloat16>(qkv_w), ptr<nv_bfloat16>(qkv_b),
        ptr<nv_bfloat16>(qkv_buf), ptr<nv_bfloat16>(attn_out),
        ptr<nv_bfloat16>(oproj_w), ptr<nv_bfloat16>(oproj_b),
        ptr<nv_bfloat16>(fc1_w), ptr<nv_bfloat16>(fc1_b),
        ptr<nv_bfloat16>(fc1_buf), ptr<nv_bfloat16>(fc2_w),
        ptr<nv_bfloat16>(fc2_b), ptr<nv_bfloat16>(patch_in),
        ptr<nv_bfloat16>(patch_w), ptr<nv_bfloat16>(patch_b),
        ptr<nv_bfloat16>(proj_w), ptr<nv_bfloat16>(proj_b),
        ptr<nv_bfloat16>(proj_out), ptr<float>(ln_stats),
        ptr<nv_bfloat16>(ln_fold_qkv), ptr<nv_bfloat16>(ln_fold_fc1),
        ptr<int32_t>(Bar), ptr<int32_t>(instructions),
        static_cast<int>(instructions.size(2)),
        static_cast<int>(vision_x.size(-2)),
    };
    for (int opcode : kOpcodes) {
        const PhaseKernel phase_kernel = phase_kernel_for(opcode);
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            phase_kernel.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, phase_kernel.smem));
    }
    configure_gemm_kernel<SiglipQkvView1, SIGLIP_QKV>();
    configure_gemm_kernel<SiglipQkvView2, SIGLIP_QKV>();
    configure_gemm_kernel<SiglipQkvView3, SIGLIP_QKV>();
    configure_gemm_kernel<SiglipOprojView1, SIGLIP_OPROJ>();
    configure_gemm_kernel<SiglipOprojView2, SIGLIP_OPROJ>();
    configure_gemm_kernel<SiglipOprojView3, SIGLIP_OPROJ>();
    configure_gemm_kernel<SiglipFc1View2, SIGLIP_FC1, true>();
    configure_gemm_kernel<SiglipFc1View3, SIGLIP_FC1>();
    configure_gemm_kernel<SiglipFc2View2, SIGLIP_FC2>();
    configure_gemm_kernel<SiglipFc2View3, SIGLIP_FC2>();
    configure_gemm_kernel<SiglipFc1View1, SIGLIP_FC1>();
    configure_gemm_kernel<SiglipFc2View1, SIGLIP_FC2>();
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        siglip_mlp_v3_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SiglipFc2View3::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        siglip_mlp_v1_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SiglipFc2View1::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        siglip_mlp_v2_subgraph,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SiglipFc2View2::kShmSize));
    configure_attention_chain<256, 1, SiglipQkvView1, SiglipOprojView1>(0);
    configure_attention_chain<512, 2, SiglipQkvView2, SiglipOprojView2>(1);
    configure_attention_chain<768, 3, SiglipQkvView3, SiglipOprojView3>(2);
    int fc1_active = 0;
    int fc2_active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fc1_active, siglip_gemm_kernel<SiglipFc1View3, SIGLIP_FC1>,
        SiglipFc1View3::kNumThreads, SiglipFc1View3::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fc2_active, siglip_gemm_kernel<SiglipFc2View3, SIGLIP_FC2>,
        SiglipFc2View3::kNumThreads, SiglipFc2View3::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &g_mlp_v3_ctas_per_sm, siglip_mlp_v3_subgraph,
        SiglipFc1View3::kNumThreads, SiglipFc2View3::kShmSize));
    TORCH_CHECK(fc1_active == 2 && fc2_active == 2 &&
                    g_mlp_v3_ctas_per_sm == 2,
                "three-view MLP residency changed: FC1=", fc1_active,
                " FC2=", fc2_active, " fused=", g_mlp_v3_ctas_per_sm);
    int fc1_v1_fuse_active = 0;
    int fc2_v1_fuse_active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fc1_v1_fuse_active,
        siglip_gemm_kernel<SiglipFc1View1, SIGLIP_FC1>,
        SiglipFc1View1::kNumThreads, SiglipFc1View1::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fc2_v1_fuse_active,
        siglip_gemm_kernel<SiglipFc2View1, SIGLIP_FC2>,
        SiglipFc2View1::kNumThreads, SiglipFc2View1::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &g_mlp_v1_ctas_per_sm, siglip_mlp_v1_subgraph,
        SiglipFc1View1::kNumThreads, SiglipFc2View1::kShmSize));
    TORCH_CHECK(fc1_v1_fuse_active == 2 && fc2_v1_fuse_active == 2 &&
                    g_mlp_v1_ctas_per_sm == 2,
                "one-view selected MLP residency changed: FC1=",
                fc1_v1_fuse_active, " FC2=", fc2_v1_fuse_active,
                " fused=", g_mlp_v1_ctas_per_sm);
    int fc1_v2_active = 0;
    int fc2_v2_active = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fc1_v2_active,
        siglip_gemm_kernel<SiglipFc1View2, SIGLIP_FC1, true>,
        SiglipFc1View2::kNumThreads, SiglipFc1View2::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &fc2_v2_active,
        siglip_gemm_kernel<SiglipFc2View2, SIGLIP_FC2>,
        SiglipFc2View2::kNumThreads, SiglipFc2View2::kShmSize));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &g_mlp_v2_ctas_per_sm, siglip_mlp_v2_subgraph,
        SiglipFc1View2::kNumThreads, SiglipFc2View2::kShmSize));
    TORCH_CHECK(fc1_v2_active == 3 && fc2_v2_active == 3 &&
                    g_mlp_v2_ctas_per_sm >= 3,
                "two-view MLP residency changed: FC1=", fc1_v2_active,
                " FC2=", fc2_v2_active,
                " fused=", g_mlp_v2_ctas_per_sm);
    g_bound = true;
}

void siglip_run_phase(torch::Tensor phase, int64_t opcode) {
    TORCH_CHECK(g_bound, "call bind() before run_phase()");
    check_cuda_contiguous(phase, "phase");
    TORCH_CHECK(phase.scalar_type() == torch::kInt32, "phase must be int32");
    TORCH_CHECK(
        phase.dim() == 2 && phase.size(1) == kInstructionWidth,
        "phase must be [count, ", kInstructionWidth, "]");
    const PhaseKernel phase_kernel = phase_kernel_for(static_cast<int>(opcode));
    TORCH_CHECK(phase_kernel.kernel != nullptr, "opcode ", opcode, " has no phase kernel");
    const int count = static_cast<int>(phase.size(0));
    if (count == 0) return;
    phase_kernel.kernel<<<count, kThreads, phase_kernel.smem, current_stream()>>>(
        g_globals, ptr<int32_t>(phase), count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void siglip_run_mlp_v3_subgraph(
    torch::Tensor barrier, int64_t layer64, bool prefetch_fc2) {
    TORCH_CHECK(g_bound, "call bind() before run_mlp_v3_subgraph()");
    TORCH_CHECK(g_globals.seq == 768,
                "MLP subgraph is specialized for three-view seq=768; got ",
                g_globals.seq);
    check_cuda_contiguous(barrier, "subgraph_barrier");
    TORCH_CHECK(barrier.scalar_type() == torch::kInt32 && barrier.numel() >= 2,
                "subgraph_barrier must contain at least two int32 words");
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < siglip::kLayers,
                "invalid SigLIP layer ", layer);
    constexpr int kGrid =
        (768 / SiglipFc1View3::BM) *
        ((siglip::kMlp + SiglipFc1View3::BN - 1) / SiglipFc1View3::BN);
    TORCH_CHECK(kGrid <= g_mlp_v3_ctas_per_sm * kSmCount,
                "MLP subgraph grid is not fully resident");
    siglip_mlp_v3_subgraph<<<
        kGrid, SiglipFc1View3::kNumThreads, SiglipFc2View3::kShmSize, current_stream()>>>(
            g_globals, ptr<int32_t>(barrier), layer, prefetch_fc2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void siglip_run_mlp_v1_subgraph(
    torch::Tensor barrier, int64_t layer64, bool prefetch_fc2) {
    TORCH_CHECK(g_bound, "call bind() before run_mlp_v1_subgraph()");
    TORCH_CHECK(g_globals.seq == 256,
                "selected MLP subgraph is specialized for one-view seq=256; got ",
                g_globals.seq);
    check_cuda_contiguous(barrier, "subgraph_barrier");
    TORCH_CHECK(barrier.scalar_type() == torch::kInt32 && barrier.numel() >= 2,
                "subgraph_barrier must contain at least two int32 words");
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < siglip::kLayers,
                "invalid SigLIP layer ", layer);
    constexpr int kGrid =
        (256 / SiglipFc1View1::BM) *
        ((siglip::kMlp + SiglipFc1View1::BN - 1) / SiglipFc1View1::BN);
    TORCH_CHECK(kGrid <= g_mlp_v1_ctas_per_sm * kSmCount,
                "selected MLP subgraph grid is not fully resident");
    siglip_mlp_v1_subgraph<<<
        kGrid, SiglipFc1View1::kNumThreads, SiglipFc2View1::kShmSize,
        current_stream()>>>(
            g_globals, ptr<int32_t>(barrier), layer, prefetch_fc2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void siglip_run_mlp_v2_subgraph(
    torch::Tensor barrier, int64_t layer64, bool prefetch_fc2) {
    TORCH_CHECK(g_bound, "call bind() before run_mlp_v2_subgraph()");
    TORCH_CHECK(g_globals.seq == 512,
                "MLP subgraph is specialized for two views; got seq=",
                g_globals.seq);
    check_cuda_contiguous(barrier, "subgraph_barrier");
    TORCH_CHECK(barrier.scalar_type() == torch::kInt32 && barrier.numel() >= 2,
                "subgraph_barrier must contain at least two int32 words");
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < siglip::kLayers,
                "invalid SigLIP layer ", layer);
    constexpr int kGrid = 272;
    TORCH_CHECK(kGrid <= g_mlp_v2_ctas_per_sm * kSmCount,
                "two-view MLP subgraph grid is not fully resident");
    siglip_mlp_v2_subgraph<<<
        kGrid, SiglipFc1View2::kNumThreads, SiglipFc2View2::kShmSize, current_stream()>>>(
            g_globals, ptr<int32_t>(barrier), layer, prefetch_fc2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int Rows, int Views, typename Qkv, typename Oproj>
void launch_siglip_attention_chain(
    torch::Tensor barrier, torch::Tensor ready, int layer,
    bool prefetch_oproj, int slot) {
    constexpr int kQkvTasks =
        (Rows / Qkv::BM) *
        ((siglip::kQkvWidth + Qkv::BN - 1) / Qkv::BN);
    constexpr int kAttnTasks =
        Views * (256 / siglip_attn_tc::kBlockM) * siglip::kHeads;
    constexpr int kOprojMTiles = (Rows + Oproj::BM - 1) / Oproj::BM;
    constexpr int kOprojTasks =
        kOprojMTiles * (siglip::kWidthPad / Oproj::BN);
    constexpr int kGrid0 = SiglipConstMax<kQkvTasks, kAttnTasks>::value;
    constexpr int kGrid = SiglipConstMax<kGrid0, kOprojTasks>::value;
    constexpr int smem_bytes = attention_chain_smem<Qkv, Oproj>();
    TORCH_CHECK(kGrid <= g_attention_chain_ctas_per_sm[slot] * kSmCount,
                Views, "-view attention-chain grid is not fully resident");
    TORCH_CHECK(ready.numel() >= kOprojMTiles,
                "attention_oproj_ready must contain at least ",
                kOprojMTiles, " int32 counters");
    siglip_qkv_attn_oproj_subgraph<Rows, Views, Qkv, Oproj><<<
        kGrid, Qkv::kNumThreads, smem_bytes, current_stream()>>>(
            g_globals, ptr<int32_t>(barrier), ptr<int32_t>(ready), layer,
            prefetch_oproj);
}

void siglip_run_attention_chain(
    torch::Tensor barrier, torch::Tensor ready, int64_t layer64,
    bool prefetch_oproj) {
    TORCH_CHECK(g_bound, "call bind() before run_attention_chain()");
    check_cuda_contiguous(barrier, "subgraph_barrier");
    TORCH_CHECK(barrier.scalar_type() == torch::kInt32 && barrier.numel() >= 2,
                "subgraph_barrier must contain at least two int32 words");
    check_cuda_contiguous(ready, "attention_oproj_ready");
    TORCH_CHECK(ready.scalar_type() == torch::kInt32,
                "attention_oproj_ready must be int32");
    const int layer = static_cast<int>(layer64);
    TORCH_CHECK(layer >= 0 && layer < siglip::kLayers,
                "invalid SigLIP layer ", layer);
    if (g_globals.seq == 256) {
        launch_siglip_attention_chain<256, 1, SiglipQkvView1, SiglipOprojView1>(
            barrier, ready, layer, prefetch_oproj, 0);
    } else if (g_globals.seq == 512) {
        launch_siglip_attention_chain<512, 2, SiglipQkvView2, SiglipOprojView2>(
            barrier, ready, layer, prefetch_oproj, 1);
    } else if (g_globals.seq == 768) {
        launch_siglip_attention_chain<768, 3, SiglipQkvView3, SiglipOprojView3>(
            barrier, ready, layer, prefetch_oproj, 2);
    } else {
        TORCH_CHECK(false, "attention chain requires seq=256/512/768; got ",
                    g_globals.seq);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void bind_module(py::module_& m) {
    m.doc() = "Pi0 A100 production SigLIP kernels";
    m.def(
        "bind", &siglip_bind,
        py::arg("vision_x"), py::arg("siglip_x_normed"),
        py::arg("ln_gamma"), py::arg("ln_beta"),
        py::arg("qkv_w"), py::arg("qkv_b"), py::arg("qkv_buf"),
        py::arg("attn_out"), py::arg("oproj_w"), py::arg("oproj_b"),
        py::arg("fc1_w"), py::arg("fc1_b"), py::arg("fc1_buf"),
        py::arg("fc2_w"), py::arg("fc2_b"),
        py::arg("patch_in"), py::arg("patch_w"), py::arg("patch_b"),
        py::arg("proj_w"), py::arg("proj_b"), py::arg("proj_out"),
        py::arg("ln_stats"), py::arg("ln_fold_qkv"), py::arg("ln_fold_fc1"),
        py::arg("Bar"), py::arg("instructions"));
    m.def("run_phase", &siglip_run_phase, py::arg("phase"), py::arg("opcode"));
    m.def("run_mlp_view3", &siglip_run_mlp_v3_subgraph,
          py::arg("barrier"), py::arg("layer"),
          py::arg("prefetch_fc2") = true);
    m.def("run_mlp_view1", &siglip_run_mlp_v1_subgraph,
          py::arg("barrier"), py::arg("layer"),
          py::arg("prefetch_fc2") = true);
    m.def("run_mlp_view2", &siglip_run_mlp_v2_subgraph,
          py::arg("barrier"), py::arg("layer"),
          py::arg("prefetch_fc2") = true);
    m.def("run_attention", &siglip_run_attention_chain,
          py::arg("barrier"), py::arg("ready"), py::arg("layer"),
          py::arg("prefetch_oproj") = false);
}

#ifndef PI0_MK_MODULE
#error "PI0_MK_MODULE must be provided by the route Makefile"
#endif

PYBIND11_MODULE(PI0_MK_MODULE, m) {
    bind_module(m);
}
