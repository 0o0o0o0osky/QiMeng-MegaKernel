# Qwen2.5 3B H100 decode_kernel: build mk.so
#   conda activate qimeng_megakernel && make
#   or: make build-with-env
# Verifier decode_kernel: from this dir, 7 levels up to qimeng_megakernel package root.

CONDA_ENV ?= qimeng_megakernel
NVCC ?= nvcc
PYTHON ?= python3
PYTHON_CONFIG ?= $(shell $(PYTHON) -c "import sys; from pathlib import Path; print(Path(sys.executable).with_name('python3-config'))")

ifndef GPU
GPU=H100
endif

ifndef THUNDERKITTENS_ROOT
# 7 levels up from verifier/.../decode_kernel to qimeng_megakernel package root
QIMENG_ROOT ?= ../../../../../../..
THUNDERKITTENS_ROOT := $(QIMENG_ROOT)/third_party/ThunderKittens
endif

ifndef MEGAKERNELS_ROOT
MEGAKERNELS_ROOT := .
endif

ifndef PYTHON_VERSION
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
endif

MK_MODULE_NAME ?= qimeng_qwen2_5_3b_h100_single_batch_kernel
EXT_SUFFIX := $(shell $(PYTHON_CONFIG) --extension-suffix)
TARGET=$(MK_MODULE_NAME)$(EXT_SUFFIX)
SRC=mk.cu

NVCCFLAGS=-DNDEBUG -DMK_MODULE_NAME=$(MK_MODULE_NAME) -Xcompiler=-fPIE --expt-extended-lambda --expt-relaxed-constexpr -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing --use_fast_math -forward-unknown-to-host-compiler -O3 -Xnvlink=--verbose -Xptxas=--verbose -Xptxas=--warn-on-spills -Xptxas=-O3 -std=c++20 -x cu -lrt -lpthread -ldl -lcuda -lcudadevrt -lcudart_static -lcublas -lineinfo
NVCCFLAGS+= -I$(THUNDERKITTENS_ROOT)/include -I$(MEGAKERNELS_ROOT)/include $(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON_CONFIG) --ldflags) -shared -fPIC -lpython$(PYTHON_VERSION)

ifeq ($(GPU),4090)
NVCCFLAGS+= -DKITTENS_4090 -arch=sm_89
else ifeq ($(GPU),A100)
NVCCFLAGS+= -DKITTENS_A100 -arch=sm_80
else ifeq ($(GPU),H100)
NVCCFLAGS+= -DKITTENS_HOPPER -arch=sm_90a
else
NVCCFLAGS+= -DKITTENS_HOPPER -DKITTENS_BLACKWELL -arch=sm_100a
endif

# Release uses the acquire-form GMEM wait and hardware-parking semaphore wait.
# Arm timeout diagnostics without forking source via:
#   make clean && make NVCCFLAGS_EXTRA=-UNDEBUG
# Keep this as the final NVCCFLAGS append so command-line -UNDEBUG wins.
NVCCFLAGS+= $(NVCCFLAGS_EXTRA)

all: $(TARGET)

build-with-env:
	bash -c 'eval "$$(conda shell.bash hook 2>/dev/null)" && conda activate $(CONDA_ENV) && make all'

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)

.PHONY: all build-with-env clean
