# Llama-3.2-1B H100 single-batch MegaKernel build.
#   conda activate qimeng_megakernel
#   make
# QIMENG_ROOT is seven levels above this directory.
#   decode_kernel -> single_batch -> H100 -> llama3_2__1b -> llama -> LLM -> models -> qimeng_megakernel
# Keep this file byte-identical in the model and verifier trees.
# Source identity check: verifier/.../single_batch/pyvm_to_megakernel/test_kernel_source_identity.py

NVCC ?= nvcc
PYTHON ?= python3
PYTHON_CONFIG ?= $(shell $(PYTHON) -c "import sys; from pathlib import Path; print(Path(sys.executable).with_name('python3-config'))")

ifndef GPU
GPU=H100
endif

ifndef THUNDERKITTENS_ROOT
# The model and verifier copies have the same depth from the package root.
QIMENG_ROOT ?= ../../../../../../..
THUNDERKITTENS_ROOT := $(QIMENG_ROOT)/third_party/ThunderKittens
endif

ifndef MEGAKERNELS_ROOT
MEGAKERNELS_ROOT := .
endif

ifndef PYTHON_VERSION
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
endif

MK_MODULE_NAME ?= qimeng_llama3_2_1b_h100_single_batch_kernel
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

# Release builds use -DNDEBUG and compile the pure-spin barrier/semaphore path
# without timeout diagnostics. Enable the watchdog with:
#     make clean && make NVCCFLAGS_EXTRA=-UNDEBUG
# The trailing -U overrides -DNDEBUG for this diagnostic build.
NVCCFLAGS+= $(NVCCFLAGS_EXTRA)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
