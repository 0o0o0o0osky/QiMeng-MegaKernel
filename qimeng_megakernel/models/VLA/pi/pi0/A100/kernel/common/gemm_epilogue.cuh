// Composable epilogues selected by the Pi0 Action Expert GEMM instances.
//
// Each functor implements
//   __device__ float transform(float acc, int m_global, int n_global) const;
//
// The unified GEMM kernel applies transform() on fp32 MMA accumulators *before*
// bf16 conversion and vectorized S2G store.  Plain GEMM (no bias/residual/fusion)
// uses EpilogueIdentity — a no-op passthrough.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace mk_gemm {

__device__ __forceinline__ float gelu_tanh(float x) {
    float x3    = x * x * x;
    float inner = 0.79788456f * (x + 0.044715f * x3);
    return x * 0.5f * (1.0f + tanhf(inner));
}

// No-op: raw GEMM output (cuBLAS-equivalent when no bias).
struct EpilogueIdentity {
    __device__ __forceinline__
    float transform(float acc, int /*m*/, int /*n*/) const {
        return acc;
    }
};

template <typename T = nv_bfloat16>
struct EpilogueBias {
    T const* bias_ptr = nullptr;

    __device__ __forceinline__
    float transform(float acc, int /*m*/, int n_global) const {
        return acc + float(bias_ptr[n_global]);
    }
};

template <typename T = nv_bfloat16>
struct EpilogueBiasResidual {
    T const* bias_ptr     = nullptr;
    T const* residual_ptr = nullptr;
    int      ld           = 0;

    __device__ __forceinline__
    float transform(float acc, int m_global, int n_global) const {
        float biased   = acc + float(bias_ptr[n_global]);
        float residual = float(residual_ptr[(int64_t)m_global * ld + n_global]);
        return residual + biased;
    }
};

// Bias-free residual used by this route's OProj and Down post slots.
template <typename T = nv_bfloat16>
struct EpilogueResidual {
    T const* residual_ptr = nullptr;
    int      ld           = 0;

    __device__ __forceinline__
    float transform(float acc, int m_global, int n_global) const {
        return acc + float(residual_ptr[(int64_t)m_global * ld + n_global]);
    }
};

}  // namespace mk_gemm
