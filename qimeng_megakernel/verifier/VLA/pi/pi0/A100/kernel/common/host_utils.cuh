#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "common/pi0_config.cuh"

namespace pi0_a100 {

inline void check_cuda_contiguous(const torch::Tensor& tensor, const char* name) {
    TORCH_CHECK(tensor.defined(), name, " must be defined");
    TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

inline void check_a100_hardware() {
    static thread_local int validated_device = -1;
    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    if (device == validated_device) {
        return;
    }
    cudaDeviceProp props{};
    C10_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    TORCH_CHECK(
        props.major == 8 && props.minor == 0 &&
            props.multiProcessorCount == kSmCount,
        "Pi0 A100 route requires a 108-SM compute-8.0 A100/A800");
    validated_device = device;
}

template <typename T>
inline T* ptr(const torch::Tensor& tensor) {
    return reinterpret_cast<T*>(tensor.data_ptr());
}

inline cudaStream_t current_stream() {
    return at::cuda::getCurrentCUDAStream();
}

}  // namespace pi0_a100
