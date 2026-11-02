// Pi0 Action Expert kernels for A100/A800.

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/ae_globals.cuh"
#include "common/host_utils.cuh"
#include "common/instruction_list.cuh"
#include "common/runtime.cuh"

#include "action_expert/ae_rmsn.cu"
#include "action_expert/ae_qkv_rope.cu"
#include "action_expert/ae_attn.cu"
#include "action_expert/ae_attn_materialized.cu"
#include "action_expert/ae_resid_gemm.cu"
#include "action_expert/ae_ffn_dual.cu"
#include "action_expert/ae_action.cu"

namespace py = pybind11;
using namespace pi0_a100;

static constexpr int kSmemScalar = 256;

#define AE_PHASE_KERNEL(NAME, BODY)                                          \
    extern "C" __global__ __launch_bounds__(kThreads, 1)                     \
    void ae_phase_##NAME(                                                     \
        AeGlobals g, const int32_t* insts, int count) {                       \
        extern __shared__ __align__(16) unsigned char smem[];                 \
        const int index = static_cast<int>(blockIdx.x);                       \
        if (index >= count) return;                                           \
        const int32_t* inst =                                                 \
            insts + static_cast<int64_t>(index) * kInstructionWidth;          \
        BODY;                                                                 \
    }

AE_PHASE_KERNEL(rmsn_out, ae_rmsn(g, inst, 2, smem))
__global__ __launch_bounds__(kThreads, 2) void ae_warp_rows_rmsn(
    AeGlobals g, int norm_row, int query_rows) {
    constexpr int kWarps = kThreads / 32;
    constexpr int kOldPasses = 2;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    if (row >= query_rows) return;

    const float* src = g.x_acc + static_cast<int64_t>(row) * ae::kWidth;
    float4 held[kWarps][kOldPasses];
    float partial[kWarps] = {};
    #pragma unroll
    for (int old_warp = 0; old_warp < kWarps; ++old_warp) {
        #pragma unroll
        for (int pass = 0; pass < kOldPasses; ++pass) {
            const int element = pass * 512 + (old_warp * 32 + lane) * 4;
            held[old_warp][pass] =
                *reinterpret_cast<const float4*>(src + element);
            const float4 value = held[old_warp][pass];
            partial[old_warp] = fmaf(value.x, value.x, partial[old_warp]);
            partial[old_warp] = fmaf(value.y, value.y, partial[old_warp]);
            partial[old_warp] = fmaf(value.z, value.z, partial[old_warp]);
            partial[old_warp] = fmaf(value.w, value.w, partial[old_warp]);
        }
        partial[old_warp] = ae_warp_sum(partial[old_warp]);
    }
    if (lane == 0) {
        const float total =
            (partial[0] + partial[2]) + (partial[1] + partial[3]);
        g.rms_factor[static_cast<int64_t>(norm_row) * ae::kRows + row] =
            rsqrtf(total / static_cast<float>(ae::kWidth) + ae::kRmsEps);
    }

    nv_bfloat16* dst = g.x + static_cast<int64_t>(row) * ae::kWidth;
    #pragma unroll
    for (int old_warp = 0; old_warp < kWarps; ++old_warp) {
        #pragma unroll
        for (int pass = 0; pass < kOldPasses; ++pass) {
            const int element = pass * 512 + (old_warp * 32 + lane) * 4;
            const float4 value = held[old_warp][pass];
            alignas(16) nv_bfloat16 out[4];
            out[0] = __float2bfloat16(value.x);
            out[1] = __float2bfloat16(value.y);
            out[2] = __float2bfloat16(value.z);
            out[3] = __float2bfloat16(value.w);
            *reinterpret_cast<uint2*>(dst + element) =
                *reinterpret_cast<const uint2*>(out);
        }
    }
}

// Consume the four deterministic residual planes while the row is already
// resident for RMSNorm.  This keeps the OProj/Down producer grid unchanged and
// removes the standalone 32-task reduction without any in-grid dependency.
__global__ __launch_bounds__(kThreads, 2) void ae_warp_rows_residual_rmsn(
    AeGlobals g, int norm_row, int query_rows) {
    constexpr int kWarps = kThreads / 32;
    constexpr int kPerLane = 4;
    constexpr int kPerPass = kThreads * kPerLane;
    constexpr int kPasses = ae::kWidth / kPerPass;
    constexpr int kPlane = ae::kRows * ae::kWidth;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int row = static_cast<int>(blockIdx.x);
    if (row >= query_rows) return;

    float* residual = g.x_acc + static_cast<int64_t>(row) * ae::kWidth;
    const float* scratch = reinterpret_cast<const float*>(g.attn_partial);
    float4 held[kPasses];
    float partial = 0.0f;
    #pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        const int element = pass * kPerPass + threadIdx.x * kPerLane;
        const int index = row * ae::kWidth + element;
        float4 left = *reinterpret_cast<const float4*>(scratch + index);
        const float4 p1 =
            *reinterpret_cast<const float4*>(scratch + kPlane + index);
        left.x += p1.x;
        left.y += p1.y;
        left.z += p1.z;
        left.w += p1.w;
        float4 right =
            *reinterpret_cast<const float4*>(scratch + 2 * kPlane + index);
        const float4 p3 =
            *reinterpret_cast<const float4*>(scratch + 3 * kPlane + index);
        right.x += p3.x;
        right.y += p3.y;
        right.z += p3.z;
        right.w += p3.w;
        left.x += right.x;
        left.y += right.y;
        left.z += right.z;
        left.w += right.w;
        float4 value = *reinterpret_cast<const float4*>(residual + element);
        value.x += left.x;
        value.y += left.y;
        value.z += left.z;
        value.w += left.w;
        held[pass] = value;
        *reinterpret_cast<float4*>(residual + element) = value;
        partial = fmaf(value.x, value.x, partial);
        partial = fmaf(value.y, value.y, partial);
        partial = fmaf(value.z, value.z, partial);
        partial = fmaf(value.w, value.w, partial);
    }
    partial = ae_warp_sum(partial);
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[warp] = partial;
    __syncthreads();
    if (threadIdx.x == 0) {
        const float total =
            (warp_sums[0] + warp_sums[2]) +
            (warp_sums[1] + warp_sums[3]);
        g.rms_factor[static_cast<int64_t>(norm_row) * ae::kRows + row] =
            rsqrtf(total / static_cast<float>(ae::kWidth) + ae::kRmsEps);
    }

    nv_bfloat16* dst = g.x + static_cast<int64_t>(row) * ae::kWidth;
    #pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        const int element = pass * kPerPass + threadIdx.x * kPerLane;
        const float4 value = held[pass];
        alignas(16) nv_bfloat16 out[4];
        out[0] = __float2bfloat16(value.x);
        out[1] = __float2bfloat16(value.y);
        out[2] = __float2bfloat16(value.z);
        out[3] = __float2bfloat16(value.w);
        *reinterpret_cast<uint2*>(dst + element) =
            *reinterpret_cast<const uint2*>(out);
    }
}

template <typename Config>
__global__ __launch_bounds__(Config::kNumThreads, 1) void ae_logical_qkv(
    AeGlobals g, int layer, int query_rows, int m_tiles, int gemm_rows,
    int state_key) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kNTiles = ae::kQkvWidth / Config::BN;
    for (int task = blockIdx.x; task < m_tiles * kNTiles; task += gridDim.x) {
        const int tile_m = task / kNTiles;
        const int tile_n = task - tile_m * kNTiles;
        const int logical_col0 = tile_n * Config::BN;
        const int query_width = ae::kHeads * ae::kHeadDim;
        const bool query_section = logical_col0 < query_width;
        const int section_col0 = query_section ? logical_col0 : logical_col0 - query_width;
        const int section_width = query_section ? query_width : 2 * ae::kHeadDim;
        const nv_bfloat16* weight = g.qkv_w +
            static_cast<int64_t>(layer) * ae::kQkvWidth * ae::kWidth +
            (query_section ? 0 : static_cast<int64_t>(query_width) * ae::kWidth);
        nv_bfloat16* section_out = g.qkv_buf + (query_section ? 0 : query_width);
        mk_gemm::gemm_unified_tile<Config>(
            g.x, weight, section_out, gemm_rows, section_width, ae::kWidth,
            tile_m, section_col0 / Config::BN,
            section_col0 / Config::BN + 1,
            AeRowScale{g.rms_factor}, smem, false,
            ae::kWidth, ae::kWidth, ae::kQkvWidth);
        __syncthreads();
        for (int item = threadIdx.x;
             item < Config::BM * (Config::BN / 2); item += blockDim.x) {
            const int row = tile_m * Config::BM + item / (Config::BN / 2);
            if (row >= gemm_rows) continue;
            const int pair = item % (Config::BN / 2);
            const int col = logical_col0 + 2 * pair;
            nv_bfloat16* dst =
                g.qkv_buf + static_cast<int64_t>(row) * ae::kQkvWidth + col;
            float even = __bfloat162float(dst[0]);
            float odd = __bfloat162float(dst[1]);
            if (col < query_width + ae::kHeadDim) {
                const int rope_col = col % ae::kHeadDim;
                const nv_bfloat16* rope =
                    g.rope + static_cast<int64_t>(row) * ae::kHeadDim + rope_col;
                const float cosine = __bfloat162float(rope[0]);
                const float sine = __bfloat162float(rope[1]);
                const float rotated_even = even * cosine - odd * sine;
                odd = even * sine + odd * cosine;
                even = rotated_even;
            }
            dst[0] = __float2bfloat16(even);
            dst[1] = __float2bfloat16(odd);
            if (!query_section) {
                const int section = (col - query_width) / ae::kHeadDim;
                const int cache_col = (col - query_width) % ae::kHeadDim;
                nv_bfloat16* cache = (section == 0 ? g.k_cache : g.v_cache) +
                    (static_cast<int64_t>(layer) * g.kv_rows + state_key + row) *
                        ae::kHeadDim + cache_col;
                cache[0] = dst[0];
                cache[1] = dst[1];
            }
        }
    }
}

using AeLogical32 = mk_gemm::GemmConfig<32, 32, 64, 3, 2, 2, 1>;

struct AeActMlpFp32Store {
    static constexpr bool kUseFp32TerminalStore = true;
    float* output;
    const float* bias;
    const float* state;

    __device__ __forceinline__ float transform(float value, int row, int col) const {
        return row == 0 ? state[col] : value + bias[col];
    }
    __device__ __forceinline__ void store(float value, int row, int col) const {
        output[static_cast<int64_t>(row) * ae::kWidth + col] = value;
    }
};

struct AeActOutFp32Store {
    static constexpr bool kUseFp32TerminalStore = true;
    nv_bfloat16* noise;
    const float* bias;
    const float* factor;

    __device__ __forceinline__ float transform(float value, int, int) const {
        return value;
    }
    __device__ __forceinline__ void store(float value, int row, int col) const {
        const nv_bfloat16 delta =
            __float2bfloat16(value * factor[row] + bias[col]);
        nv_bfloat16* destination =
            noise + static_cast<int64_t>(row) * ae::kKSplit + col;
        *destination = __float2bfloat16(
            __bfloat162float(*destination) + __bfloat162float(delta));
    }
};

template <typename Config>
__global__ __launch_bounds__(Config::kNumThreads, 1) void ae_logical_act_in(
    AeGlobals g, int bias_row, int query_rows) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kNTiles = ae::kWidth / Config::BN;
    const int task = static_cast<int>(blockIdx.x);
    const int tile_m = task / kNTiles;
    const int tile_n = task - tile_m * kNTiles;
    mk_gemm::gemm_unified_tile<Config>(
        g.noise, g.act_w_in, g.act_h,
        query_rows, ae::kWidth, ae::kKSplit,
        tile_m, tile_n, tile_n + 1,
        AeActInEpilogue{g.act_bias + static_cast<int64_t>(bias_row) * ae::kWidth},
        smem);
}

template <typename Config>
__global__ __launch_bounds__(Config::kNumThreads, 1) void ae_logical_act_mlp(
    AeGlobals g, int bias_row, int query_rows) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kNTiles = ae::kWidth / Config::BN;
    const int task = static_cast<int>(blockIdx.x);
    const int tile_m = task / kNTiles;
    const int tile_n = task - tile_m * kNTiles;
    mk_gemm::gemm_unified_tile<Config>(
        g.act_h, g.act_w_mlp, g.act_h,
        query_rows, ae::kWidth, ae::kWidth,
        tile_m, tile_n, tile_n + 1,
        AeActMlpFp32Store{
            g.x_acc,
            g.act_bias + static_cast<int64_t>(bias_row) * ae::kWidth,
            g.act_bias + static_cast<int64_t>(12) * ae::kWidth},
        smem);
}

template <typename Config>
__global__ __launch_bounds__(Config::kNumThreads, 1) void ae_logical_act_out(
    AeGlobals g, int bias_row, int norm_row, int query_rows) {
    extern __shared__ __align__(16) unsigned char smem[];
    mk_gemm::gemm_unified_tile<Config>(
        g.x, g.act_w_out, g.x,
        query_rows, ae::kActionDim, ae::kWidth,
        static_cast<int>(blockIdx.x), 0, 1,
        AeActOutFp32Store{
            g.noise,
            g.act_bias + static_cast<int64_t>(bias_row) * ae::kWidth,
            g.rms_factor + static_cast<int64_t>(norm_row) * ae::kRows},
        smem);
}

template <typename Config>
__global__ __launch_bounds__(Config::kNumThreads, 1) void ae_logical_ffn(
    AeGlobals g, int layer, int query_rows, int m_tiles) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kTilesPerSplit = ae::kKSplit / Config::BN;
    const int per_split = m_tiles * kTilesPerSplit;
    const int tasks = (ae::kMlpWidth / ae::kKSplit) * per_split;
    for (int task = blockIdx.x; task < tasks; task += gridDim.x) {
        const int split = task / per_split;
        const int local = task - split * per_split;
        const int tile_m = local / kTilesPerSplit;
        const int tile_n = local - tile_m * kTilesPerSplit;
        const int64_t weight_offset =
            static_cast<int64_t>(layer) * ae::kMlpWidth * ae::kWidth +
            static_cast<int64_t>(split) * ae::kKSplit * ae::kWidth;
        nv_bfloat16* output =
            g.ffn_buf + static_cast<int64_t>(split) * ae::kRows * ae::kKSplit;
        mk_gemm::gemm_unified_dual_tile<Config>(
            g.x, g.gate_w + weight_offset, g.up_w + weight_offset,
            output, query_rows, ae::kKSplit, ae::kWidth,
            tile_m, tile_n, tile_n + 1,
            AeFfnMerge{g.rms_factor + ae::kRows},
            mk_gemm::EpilogueIdentity{}, smem);
    }
}

template <typename Config, int Segments, bool IsDown>
__global__ __launch_bounds__(Config::kNumThreads, 1) void ae_logical_residual(
    AeGlobals g, int layer, int query_rows, bool ordered) {
    extern __shared__ __align__(16) unsigned char smem[];
    constexpr int kNTiles = ae::kWidth / Config::BN;
    constexpr int kTotalSegments = IsDown ? 16 : 8;
    constexpr int kGroups = kTotalSegments / Segments;
    const int task = static_cast<int>(blockIdx.x);
    const int group = task / kNTiles;
    const int tile_n = task - group * kNTiles;
    const nv_bfloat16* input = (IsDown ? g.ffn_buf : g.attn_out) +
        static_cast<int64_t>(group * Segments) * ae::kRows * ae::kKSplit;
    const nv_bfloat16* weights = (IsDown ? g.down_w : g.oproj_w) +
        static_cast<int64_t>(layer * kTotalSegments + group * Segments) *
            ae::kWidth * ae::kKSplit;
    if constexpr (kGroups == 4) {
        if (ordered) {
            constexpr int64_t kPlane =
                static_cast<int64_t>(ae::kRows) * ae::kWidth;
            ae_segmented_partial<Config, Segments>(
                input, weights,
                reinterpret_cast<float*>(g.attn_partial) + group * kPlane,
                ae::kWidth, tile_n, smem, query_rows, true);
        } else {
            ae_segmented_partial<Config, Segments>(
                input, weights, g.x_acc, ae::kWidth, tile_n, smem, query_rows);
        }
    } else {
        ae_segmented_partial<Config, Segments>(
            input, weights, g.x_acc, ae::kWidth, tile_n, smem, query_rows);
    }
}

namespace {

using PhaseKernelFn = void (*)(AeGlobals, const int32_t*, int);

struct PhaseKernel {
    PhaseKernelFn kernel;
    int smem;
};

PhaseKernel phase_kernel_for(int opcode) {
    switch (opcode) {
        case AE_RMSN_OUT: return {ae_phase_rmsn_out, kSmemScalar};
        default: return {nullptr, 0};
    }
}

constexpr int kOpcodes[] = {AE_RMSN_OUT};

AeGlobals g_globals{};
bool g_bound = false;
int64_t g_attn_scratch_elements = 0;

}  // namespace

void ae_bind(
    torch::Tensor x,
    torch::Tensor x_acc,
    torch::Tensor rms_factor,
    torch::Tensor qkv_w,
    torch::Tensor qkv_buf,
    torch::Tensor rope,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor attn_partial,
    torch::Tensor attn_stats,
    torch::Tensor attn_out,
    torch::Tensor oproj_w,
    torch::Tensor gate_w,
    torch::Tensor up_w,
    torch::Tensor gate_buf,
    torch::Tensor ffn_buf,
    torch::Tensor down_w,
    torch::Tensor noise,
    torch::Tensor act_h,
    torch::Tensor act_w_in,
    torch::Tensor act_w_mlp,
    torch::Tensor act_w_out,
    torch::Tensor act_bias,
    torch::Tensor Bar,
    torch::Tensor instructions) {
    check_cuda_contiguous(x, "x");
    const int prefix_len = static_cast<int>(k_cache.size(-2)) - ae::kRows;
    g_globals = AeGlobals{
        ptr<nv_bfloat16>(x), ptr<float>(x_acc), ptr<float>(rms_factor),
        ptr<nv_bfloat16>(qkv_w), ptr<nv_bfloat16>(qkv_buf),
        ptr<nv_bfloat16>(rope), ptr<nv_bfloat16>(k_cache),
        ptr<nv_bfloat16>(v_cache), ptr<nv_bfloat16>(attn_partial),
        ptr<float>(attn_stats), ptr<nv_bfloat16>(attn_out),
        ptr<nv_bfloat16>(oproj_w), ptr<nv_bfloat16>(gate_w),
        ptr<nv_bfloat16>(up_w), ptr<nv_bfloat16>(gate_buf),
        ptr<nv_bfloat16>(ffn_buf), ptr<nv_bfloat16>(down_w),
        ptr<nv_bfloat16>(noise), ptr<nv_bfloat16>(act_h),
        ptr<nv_bfloat16>(act_w_in), ptr<nv_bfloat16>(act_w_mlp),
        ptr<nv_bfloat16>(act_w_out), ptr<float>(act_bias),
        ptr<int32_t>(Bar), ptr<int32_t>(instructions),
        static_cast<int>(instructions.size(2)), prefix_len,
        ((prefix_len + 63) / 64) * 64, ae::kRows,
        static_cast<int>(k_cache.size(-2)),
        static_cast<int>(attn_partial.size(1)),
    };
    g_attn_scratch_elements = attn_partial.numel();
    for (int opcode : kOpcodes) {
        const PhaseKernel phase_kernel = phase_kernel_for(opcode);
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            phase_kernel.kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, phase_kernel.smem));
    }
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_materialized::qk_vt<ae_materialized::QConfig>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        ae_materialized::QConfig::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_materialized::pv<ae_materialized::PvConfig>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        ae_materialized::PvConfig::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_qkv<AeLogical32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeLogical32::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_ffn<AeLogical32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mk_gemm::DualSmem<AeLogical32>::kShmSizeDual));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_ffn<AeDecoderGemm>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        mk_gemm::DualSmem<AeDecoderGemm>::kShmSizeDual));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_residual<AeLogical32, 2, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeLogical32::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_residual<AeLogical32, 4, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeLogical32::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_residual<AeDecoderGemm, 2, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeDecoderGemm::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_residual<AeDecoderGemm, 4, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeDecoderGemm::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_act_in<AeLogical32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeLogical32::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_act_mlp<AeLogical32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeLogical32::kShmSize));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        ae_logical_act_out<AeLogical32>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        AeLogical32::kShmSize));
    g_bound = true;
}

void ae_run_logical(
    int64_t opcode64, int64_t layer64, int64_t query_rows64,
    bool ordered) {
    TORCH_CHECK(g_bound, "call bind() before run_logical()");
    const int opcode = static_cast<int>(opcode64);
    const int layer = static_cast<int>(layer64);
    const int query_rows = static_cast<int>(query_rows64);
    if (opcode == AE_QKV_ROPE) {
        const int m_tiles = (query_rows + AeLogical32::BM - 1) / AeLogical32::BM;
        const int tasks = m_tiles * (ae::kQkvWidth / AeLogical32::BN);
        ae_logical_qkv<AeLogical32><<<
            tasks, AeLogical32::kNumThreads, AeLogical32::kShmSize,
            current_stream()>>>(
                g_globals, layer, query_rows, m_tiles, query_rows,
                g_globals.state_key);
    } else if (opcode == AE_FFN_DUAL) {
        if (query_rows <= 32) {
            constexpr int per_m = ae::kMlpWidth / AeLogical32::BN;
            ae_logical_ffn<AeLogical32><<<
                per_m, AeLogical32::kNumThreads,
                mk_gemm::DualSmem<AeLogical32>::kShmSizeDual,
                current_stream()>>>(g_globals, layer, query_rows, 1);
        } else {
            constexpr int per_m = ae::kMlpWidth / AeDecoderGemm::BN;
            ae_logical_ffn<AeDecoderGemm><<<
                per_m, AeDecoderGemm::kNumThreads,
                mk_gemm::DualSmem<AeDecoderGemm>::kShmSizeDual,
            current_stream()>>>(g_globals, layer, query_rows, 1);
        }
    } else if (opcode == AE_OPROJ) {
        if (query_rows <= 32) {
            ae_logical_residual<AeLogical32, 2, false><<<
                4 * (ae::kWidth / AeLogical32::BN),
                AeLogical32::kNumThreads, AeLogical32::kShmSize,
                current_stream()>>>(
                    g_globals, layer / ae::kOprojSplits, query_rows, ordered);
        } else {
            ae_logical_residual<AeDecoderGemm, 2, false><<<
                4 * (ae::kWidth / AeDecoderGemm::BN),
                AeDecoderGemm::kNumThreads, AeDecoderGemm::kShmSize,
                current_stream()>>>(
                    g_globals, layer / ae::kOprojSplits, query_rows, ordered);
        }
    } else if (opcode == AE_DOWN) {
        if (query_rows <= 32) {
            ae_logical_residual<AeLogical32, 4, true><<<
                4 * (ae::kWidth / AeLogical32::BN),
                AeLogical32::kNumThreads, AeLogical32::kShmSize,
                current_stream()>>>(
                    g_globals, layer / ae::kDownSplits, query_rows, ordered);
        } else {
            ae_logical_residual<AeDecoderGemm, 4, true><<<
                4 * (ae::kWidth / AeDecoderGemm::BN),
                AeDecoderGemm::kNumThreads, AeDecoderGemm::kShmSize,
                current_stream()>>>(
                    g_globals, layer / ae::kDownSplits, query_rows, ordered);
        }
    } else {
        TORCH_CHECK(false, "opcode ", opcode, " has no logical-row AE kernel");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ae_run_warp_rmsn(int64_t norm_row64, int64_t query_rows64) {
    TORCH_CHECK(g_bound, "call bind() before run_warp_rmsn()");
    const int norm_row = static_cast<int>(norm_row64);
    const int query_rows = static_cast<int>(query_rows64);
    ae_warp_rows_rmsn<<<(query_rows + 3) / 4, kThreads, 0, current_stream()>>>(
        g_globals, norm_row, query_rows);
}

void ae_run_residual_rmsn(int64_t norm_row64, int64_t query_rows64) {
    TORCH_CHECK(g_bound, "call bind() before run_residual_rmsn()");
    const int norm_row = static_cast<int>(norm_row64);
    const int query_rows = static_cast<int>(query_rows64);
    TORCH_CHECK(norm_row >= 0 && norm_row < 3, "invalid AE norm row ", norm_row);
    TORCH_CHECK(query_rows > 0 && query_rows <= ae::kRows,
                "query_rows must be in [1,64]");
    ae_warp_rows_residual_rmsn<<<
        query_rows, kThreads, 0, current_stream()>>>(
            g_globals, norm_row, query_rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ae_run_qkv_ragged(
    int64_t layer64, int64_t query_rows64, int64_t state_key64) {
    TORCH_CHECK(g_bound, "call bind() before run_qkv_ragged()");
    const int layer = static_cast<int>(layer64);
    const int query_rows = static_cast<int>(query_rows64);
    const int state_key = static_cast<int>(state_key64);
    TORCH_CHECK(layer >= 0 && layer < ae::kLayers, "invalid AE layer ", layer);
    TORCH_CHECK(query_rows > 0 && query_rows <= ae::kRows,
                "query_rows must be in [1,64]");
    const int m_tiles = (query_rows + AeLogical32::BM - 1) / AeLogical32::BM;
    const int tasks = m_tiles * (ae::kQkvWidth / AeLogical32::BN);
    ae_logical_qkv<AeLogical32><<<
        tasks, AeLogical32::kNumThreads, AeLogical32::kShmSize,
        current_stream()>>>(
            g_globals, layer, query_rows, m_tiles, query_rows, state_key);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ae_run_action(
    int64_t opcode64, int64_t bias_row64, int64_t norm_row64,
    int64_t query_rows64) {
    TORCH_CHECK(g_bound, "call bind() before run_action()");
    const int opcode = static_cast<int>(opcode64);
    const int bias_row = static_cast<int>(bias_row64);
    const int norm_row = static_cast<int>(norm_row64);
    const int query_rows = static_cast<int>(query_rows64);
    TORCH_CHECK(query_rows > 0 && query_rows <= 64,
                "logical action rows must be in [1,64]");
    const int m_tiles =
        (query_rows + AeLogical32::BM - 1) / AeLogical32::BM;
    if (opcode == AE_ACT_IN) {
        ae_logical_act_in<AeLogical32><<<
            m_tiles * (ae::kWidth / AeLogical32::BN),
            AeLogical32::kNumThreads, AeLogical32::kShmSize,
            current_stream()>>>(g_globals, bias_row, query_rows);
    } else if (opcode == AE_ACT_MLP) {
        ae_logical_act_mlp<AeLogical32><<<
            m_tiles * (ae::kWidth / AeLogical32::BN),
            AeLogical32::kNumThreads, AeLogical32::kShmSize,
            current_stream()>>>(g_globals, bias_row, query_rows);
    } else if (opcode == AE_ACT_OUT) {
        ae_logical_act_out<AeLogical32><<<
            m_tiles, AeLogical32::kNumThreads, AeLogical32::kShmSize,
            current_stream()>>>(g_globals, bias_row, norm_row, query_rows);
    } else {
        TORCH_CHECK(false, "opcode ", opcode, " has no logical action kernel");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ae_run_materialized(
    int64_t layer, int64_t query_rows, int64_t prefix_len, int64_t state_key) {
    TORCH_CHECK(g_bound, "call bind() before run_materialized()");
    TORCH_CHECK(layer >= 0 && layer < ae::kLayers, "invalid AE layer ", layer);
    TORCH_CHECK(query_rows > 0 && query_rows <= 64,
                "query_rows must be in [1,64]");
    TORCH_CHECK(prefix_len >= 0 && prefix_len <= state_key,
                "invalid prefix/state positions");
    AeGlobals globals = g_globals;
    globals.prefix_len = static_cast<int>(prefix_len);
    globals.state_key = static_cast<int>(state_key);
    const int self_slot = ((globals.prefix_len + 31) / 32) * 32;
    const int stride = ((self_slot + query_rows + 31) / 32) * 32;
    const int64_t required =
        static_cast<int64_t>(2 * 8 * query_rows + 256) * stride;
    TORCH_CHECK(required <= g_attn_scratch_elements,
                "materialized AE attention needs ", required,
                " BF16 scratch elements, has ", g_attn_scratch_elements);
    ae_materialized::launch(
        globals, static_cast<int>(layer), static_cast<int>(query_rows),
        current_stream());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ae_run_phase(torch::Tensor phase, int64_t opcode) {
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

void bind_module(py::module_& m) {
    m.doc() = "Pi0 A100 production AE kernels";
    m.def(
        "bind", &ae_bind,
        py::arg("x"), py::arg("x_acc"), py::arg("rms_factor"),
        py::arg("qkv_w"), py::arg("qkv_buf"), py::arg("rope"),
        py::arg("k_cache"), py::arg("v_cache"),
        py::arg("attn_partial"), py::arg("attn_stats"), py::arg("attn_out"),
        py::arg("oproj_w"), py::arg("gate_w"), py::arg("up_w"),
        py::arg("gate_buf"), py::arg("ffn_buf"), py::arg("down_w"),
        py::arg("noise"), py::arg("act_h"),
        py::arg("act_w_in"), py::arg("act_w_mlp"), py::arg("act_w_out"),
        py::arg("act_bias"), py::arg("Bar"), py::arg("instructions"));
    m.def("run_phase", &ae_run_phase, py::arg("phase"), py::arg("opcode"));
    m.def("run_materialized", &ae_run_materialized,
          py::arg("layer"), py::arg("query_rows"),
          py::arg("prefix_len"), py::arg("state_key"));
    m.def("run_logical", &ae_run_logical,
          py::arg("opcode"), py::arg("layer"), py::arg("query_rows"),
          py::arg("ordered") = true);
    m.def("run_rmsn", &ae_run_warp_rmsn,
          py::arg("norm_row"), py::arg("query_rows"));
    m.def("run_residual_rmsn", &ae_run_residual_rmsn,
          py::arg("norm_row"), py::arg("query_rows"));
    m.def("run_qkv", &ae_run_qkv_ragged,
          py::arg("layer"), py::arg("query_rows"), py::arg("state_key"));
    m.def("run_action", &ae_run_action,
          py::arg("opcode"), py::arg("bias_row"), py::arg("norm_row"),
          py::arg("query_rows"));
}

#ifndef PI0_MK_MODULE
#error "PI0_MK_MODULE must be provided by the route Makefile"
#endif

PYBIND11_MODULE(PI0_MK_MODULE, m) {
    bind_module(m);
}
