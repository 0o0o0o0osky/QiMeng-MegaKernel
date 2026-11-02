# Llama-3.2-1B H100 instruction-queue multi-batch decode kernel (B=2..16).
# Build explicitly with the repository's Python 3.12 environment:
#   make QIMENG_PYTHON=/path/to/python

GPU ?= H100

ifndef THUNDERKITTENS_ROOT
  QIMENG_ROOT ?= ../../../../../../..
  THUNDERKITTENS_ROOT := $(QIMENG_ROOT)/third_party/ThunderKittens
endif
ifeq ($(wildcard $(THUNDERKITTENS_ROOT)/include/kittens.cuh),)
  $(error ThunderKittens not found; set THUNDERKITTENS_ROOT)
endif

CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
PYTHON ?= python3
QIMENG_PYTHON ?= $(PYTHON)

ifeq ($(GPU),H100)
  GPU_FLAGS := -DKITTENS_HOPPER -arch=sm_90a
else
  $(error This kernel supports GPU=H100 only)
endif

PYBIND_INC := $(shell $(QIMENG_PYTHON) -m pybind11 --includes)
PY_LDFLAGS := $(shell $(QIMENG_PYTHON) -c 'import sysconfig; print("-L" + sysconfig.get_config_var("LIBDIR") + " " + (sysconfig.get_config_var("LIBS") or ""))')
EXT_SUFFIX := $(shell $(QIMENG_PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

MK_MODULE_NAME ?= qimeng_llama3_2_1b_h100_multi_batch_kernel
MODULE := $(MK_MODULE_NAME)
TARGET := $(MODULE)$(EXT_SUFFIX)
SRC := mk.cu

LIBS := -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl
NVCCFLAGS := -std=c++20 -O3 -DNDEBUG -DMK_MODULE_NAME=$(MK_MODULE_NAME) $(GPU_FLAGS) \
             --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr \
             -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
             -Xptxas=--verbose -Xptxas=--warn-on-spills -Xptxas=-O3 \
             -Xnvlink=--verbose -lineinfo \
             -I$(THUNDERKITTENS_ROOT)/include -I. \
             $(PYBIND_INC) \
             -shared -Xcompiler=-fPIC $(PY_LDFLAGS) $(LIBS)
NVCCFLAGS += $(NVCCFLAGS_EXTRA)

DEPS := Makefile globals.cuh kernel_config.cuh llama_config.cuh \
        instruction_list.cuh utils.cuh rms_qkv_gemm_rope.cu \
        decode_attention.cu oproj_gemm_residual.cu \
        rms_upgate_gemm_silu.cu down_proj_gemm_residual.cu \
        rms_lmhead_gemm.cu include/gemm/matmul_pipeline.cuh \
        include/gemm/wgmma_n8.cuh

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC) $(DEPS)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $@

clean:
	rm -f $(MODULE)*.so
