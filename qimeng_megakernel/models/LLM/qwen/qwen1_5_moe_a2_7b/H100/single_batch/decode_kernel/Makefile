# Qwen1.5-MoE-A2.7B H100 single-batch MegaKernel build.
#   conda activate qimeng_megakernel
#   make
# QIMENG_ROOT is seven levels above this directory.
#   decode_kernel -> single_batch -> H100 -> qwen1_5_moe_a2_7b -> qwen -> LLM -> models -> qimeng_megakernel
# This build follows the shared H100 single-batch optimization skeleton.

NVCC ?= nvcc
PYTHON ?= python3
PYTHON_CONFIG ?= $(shell $(PYTHON) -c "import sys; from pathlib import Path; print(Path(sys.executable).with_name('python3-config'))")

ifndef GPU
GPU=H100
endif

ifndef THUNDERKITTENS_ROOT
# Resolve the package root from the model-local kernel directory.
QIMENG_ROOT ?= ../../../../../../..
THUNDERKITTENS_ROOT := $(QIMENG_ROOT)/third_party/ThunderKittens
endif

ifndef MEGAKERNELS_ROOT
MEGAKERNELS_ROOT := .
endif

ifndef PYTHON_VERSION
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
endif

MK_MODULE_NAME ?= qimeng_qwen1_5_moe_a2_7b_h100_single_batch_kernel
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

# Escape hatch for the barrier/semaphore watchdog.  The flags above contain -DNDEBUG,
# under which `barrier.cuh` compiles the plain waits only -- `debug/barrier_watchdog.cuh`
# is not even included, so the shipped binary has no timeout, no __nanosleep, no printf
# and no __trap.  To arm the watchdog for hang hunting:
#     make clean && make NVCCFLAGS_EXTRA=-UNDEBUG
# (-U wins because it comes last on the command line).
NVCCFLAGS+= $(NVCCFLAGS_EXTRA)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(SRC) $(NVCCFLAGS) -o $@

clean:
	rm -f $(TARGET)
