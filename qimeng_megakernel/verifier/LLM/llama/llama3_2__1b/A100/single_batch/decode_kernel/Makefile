# Llama 3.2 1B Ampere decode kernel.
# S3 may override MK_MODULE_NAME for its verifier-owned build.

# Detect build mode from environment or default to RELEASE
BUILD_MODE ?= RELEASE
$(info [INFO] Building in $(BUILD_MODE) mode...)

# Paths (optional local overrides belong in uncommitted Makefile.local)
-include Makefile.local
CUDA_PATH ?= /usr/local/cuda
CUTLASS_PATH := ../../../../../../../third_party/cutlass
PYTHON ?= python3
PYTHON_INCLUDE := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_PATH := $(shell $(PYTHON) -c "import torch; import os; print(os.path.dirname(torch.__file__))")
MK_MODULE_NAME ?= qimeng_llama3_2_1b_a100_single_batch_kernel
EXT_SUFFIX := $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# Compiler flags
NVCC := $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS := -std=c++17 --expt-extended-lambda --expt-relaxed-constexpr \
              --generate-code=arch=compute_80,code=[compute_80,sm_80] \
              -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
              -DMK_MODULE_NAME=$(MK_MODULE_NAME)

CXX := g++
CXX_FLAGS := -std=c++17 -fPIC -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
             -DMK_MODULE_NAME=$(MK_MODULE_NAME)

# Build mode specific flags
ifeq ($(BUILD_MODE),DEBUG)
    NVCC_FLAGS += -g -G -O0 -lineinfo
    CXX_FLAGS += -g -O0
else
    NVCC_FLAGS += -O3 --use_fast_math -DNDEBUG
    CXX_FLAGS += -O3 -DNDEBUG
endif

# Always add verbose flags
NVCC_FLAGS += -Xptxas=--verbose -Xptxas=--warn-on-spills

# Include paths
INCLUDES := -I$(CUTLASS_PATH)/include \
            -I$(CUTLASS_PATH)/tools/util/include \
            -I$(CUDA_PATH)/include \
            -I$(PYTHON_INCLUDE) \
            -I$(TORCH_PATH)/include \
            -I$(TORCH_PATH)/include/torch/csrc/api/include \
            -I.

# Library paths
LDFLAGS := -L$(TORCH_PATH)/lib \
           -Wl,-rpath,$(TORCH_PATH)/lib \
           -ltorch -ltorch_cpu -ltorch_python -lc10 -lc10_cuda \
           -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# Dependency generation
DEPFLAGS := -MMD -MP

# Targets
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

# Include generated dependencies
-include $(DEPS)
