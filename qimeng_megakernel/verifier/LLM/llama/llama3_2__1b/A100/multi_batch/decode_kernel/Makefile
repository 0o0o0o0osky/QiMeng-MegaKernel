# Llama 3.2 1B Ampere multi-batch decode kernel.

BUILD_MODE ?= RELEASE
$(info [INFO] Building in $(BUILD_MODE) mode...)

-include Makefile.local
CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH := ../../../../../../../third_party/cutlass
PYTHON ?= python3
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_PATH := $(shell $(PYTHON) -c "import torch; import os; print(os.path.dirname(torch.__file__))")
MK_MODULE_NAME ?= qimeng_llama3_2_1b_a100_multi_batch_kernel
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')
DECODE_GEMM_STAGE_COUNT ?= 5
DECODE_QKV_STAGED_A ?= 1
DECODE_OPROJ_STAGED_A ?= 1
DECODE_UPGATE_STAGED_A ?= 1
DECODE_LMHEAD_STAGED_A ?= 1

NVCC := $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS := -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr \
              --generate-code=arch=compute_80,code=[compute_80,sm_80] \
              -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
              -DMK_MODULE_NAME=$(MK_MODULE_NAME) \
              -DDECODE_GEMM_STAGE_COUNT=$(DECODE_GEMM_STAGE_COUNT) \
              -DDECODE_QKV_STAGED_A=$(DECODE_QKV_STAGED_A) \
              -DDECODE_OPROJ_STAGED_A=$(DECODE_OPROJ_STAGED_A) \
              -DDECODE_UPGATE_STAGED_A=$(DECODE_UPGATE_STAGED_A) \
              -DDECODE_LMHEAD_STAGED_A=$(DECODE_LMHEAD_STAGED_A)

CXX := g++
CXX_FLAGS := -std=c++17 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
             -DMK_MODULE_NAME=$(MK_MODULE_NAME)

ifeq ($(BUILD_MODE),DEBUG)
    NVCC_FLAGS += -g -G -O0 -lineinfo -DDEBUG_PRINT=1
    CXX_FLAGS += -g -O0
else
    NVCC_FLAGS += -O3 --use_fast_math -DNDEBUG -DDEBUG_PRINT=0
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

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXX_FLAGS) $(OBJS) $(LDFLAGS) -o $@

$(MK_MODULE_NAME).mk.o: mk.cu *.cuh
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

$(MK_MODULE_NAME).pybind.o: pybind.cpp mk.h
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(DEPFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)

-include $(DEPS)
