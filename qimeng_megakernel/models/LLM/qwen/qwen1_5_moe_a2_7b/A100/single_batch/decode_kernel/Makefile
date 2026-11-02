# Qwen1.5-MoE-A2.7B A100 decode kernel.
# S3 may override MK_MODULE_NAME for its verifier-owned build.

-include Makefile.local
BUILD_MODE ?= RELEASE
$(info [INFO] Building in $(BUILD_MODE) mode...)

CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH := ../../../../../../../third_party/cutlass
PYTHON ?= python3
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_PATH := $(shell $(PYTHON) -c "import torch; import os; print(os.path.dirname(torch.__file__))")
MK_MODULE_NAME ?= qimeng_qwen1_5_moe_a2_7b_a100_single_batch_kernel
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

NVCC := $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS := -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr \
              --generate-code=arch=compute_80,code=[compute_80,sm_80] \
              -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
              -DMK_MODULE_NAME=$(MK_MODULE_NAME)

CXX := g++
CXX_FLAGS := -std=c++17 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
             -DMK_MODULE_NAME=$(MK_MODULE_NAME)

ifeq ($(BUILD_MODE),DEBUG)
    NVCC_FLAGS += -g -G -O0 -lineinfo
    CXX_FLAGS += -g -O0
else
    NVCC_FLAGS += -O3 --use_fast_math -DNDEBUG
    CXX_FLAGS += -O3 -DNDEBUG
endif

NVCC_FLAGS += -Xptxas=--verbose -Xptxas=--warn-on-spills

INCLUDES := -I$(CUTLASS_PATH)/include \
            -I$(CUTLASS_PATH)/tools/util/include \
            -I$(CUDA_PATH)/include \
            -I$(PYTHON_INCLUDE) \
            -I$(TORCH_PATH)/include \
            -I$(TORCH_PATH)/include/torch/csrc/api/include \
            -I.

LDFLAGS := -L$(TORCH_PATH)/lib \
           -Wl,-rpath,$(TORCH_PATH)/lib \
           -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_cuda \
           -L$(CUDA_PATH)/lib64 -lcudart -lcublas

DEPFLAGS := -MMD -MP

TARGET := $(MK_MODULE_NAME)$(EXT_SUFFIX)
OBJS := $(MK_MODULE_NAME).mk.o $(MK_MODULE_NAME).pybind.o
DEPS := $(OBJS:.o=.d)
CUDA_SOURCES := mk.cu $(filter-out mk.cu,$(wildcard *.cu)) \
                $(wildcard *.cuh) $(wildcard include/debug/*.cuh)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXX_FLAGS) $(OBJS) $(LDFLAGS) -o $@

$(MK_MODULE_NAME).mk.o: $(CUDA_SOURCES)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

$(MK_MODULE_NAME).pybind.o: pybind.cpp mk.h
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)

-include $(DEPS)
