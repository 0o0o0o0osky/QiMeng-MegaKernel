# Pi0 A100/A800 production kernels.

CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc
CXX ?= g++
PYTHON ?= python3

QIMENG_ROOT ?= ../../../../../..
CUTLASS_ROOT ?= $(QIMENG_ROOT)/third_party/cutlass

ifeq ($(wildcard $(CUTLASS_ROOT)/include/cute/tensor.hpp),)
  $(error CUTLASS/CuTe not found at $(CUTLASS_ROOT))
endif

PYTHON_INCLUDE := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("include"))')
TORCH_ROOT := $(shell $(PYTHON) -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).resolve().parent)')
TORCH_ABI := $(shell $(PYTHON) -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

COMMON := -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=$(TORCH_ABI)
INCLUDES := -I$(CUTLASS_ROOT)/include -I$(CUDA_HOME)/include -I$(PYTHON_INCLUDE) \
            -I$(TORCH_ROOT)/include -I$(TORCH_ROOT)/include/torch/csrc/api/include -I.
NVCCFLAGS := $(COMMON) -O3 -DNDEBUG --use_fast_math --expt-extended-lambda \
             --expt-relaxed-constexpr --generate-code=arch=compute_80,code=sm_80 \
             -Xcompiler=-fPIC -Xptxas=--verbose -Xptxas=--warn-on-spills -lineinfo \
             $(INCLUDES)
NVCCFLAGS_EXTRA ?=
NVCCFLAGS += $(NVCCFLAGS_EXTRA)
CXXFLAGS := $(COMMON) -O3 -DNDEBUG -fPIC $(INCLUDES)
LDFLAGS := -shared -L$(TORCH_ROOT)/lib -Wl,-rpath,$(TORCH_ROOT)/lib \
           -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_cuda \
           -L$(CUDA_HOME)/lib64 -lcudart

SIGLIP_MODULE := qimeng_pi0_a100_siglip_s3_kernel
GEMMA_MODULE := qimeng_pi0_a100_gemma_s3_kernel
AE_MODULE := qimeng_pi0_a100_ae_s3_kernel
TARGETS := $(SIGLIP_MODULE)$(EXT_SUFFIX) $(GEMMA_MODULE)$(EXT_SUFFIX) \
           $(AE_MODULE)$(EXT_SUFFIX)
COMMON_HEADERS := $(wildcard common/*.cuh)

.PHONY: all clean
all: $(TARGETS)

define build_kernel
.$(1).o: mk_$(1).cu $(COMMON_HEADERS)
	$$(NVCC) $$(NVCCFLAGS) -DPI0_MK_MODULE=$(2) -MMD -MP -c $$< -o $$@

$(2)$$(EXT_SUFFIX): .$(1).o
	$$(CXX) $$< $$(LDFLAGS) -o $$@
endef

$(eval $(call build_kernel,siglip,$(SIGLIP_MODULE)))
$(eval $(call build_kernel,gemma,$(GEMMA_MODULE)))
$(eval $(call build_kernel,ae,$(AE_MODULE)))

clean:
	rm -f .siglip.o .siglip.d .gemma.o .gemma.d .ae.o .ae.d $(TARGETS)

-include .siglip.d .gemma.d .ae.d
