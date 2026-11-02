# Pi0 H100 MegaKernel build.
# Usage: make, make clean, or make GPU=H100.
# The Python environment must provide pybind11 and python3-config.

GPU ?= H100

# ── ThunderKittens ──────────────────────────────────────────────────────────
# The package root is six levels above kernel/.
ifndef THUNDERKITTENS_ROOT
  QIMENG_ROOT ?= ../../../../../..
  THUNDERKITTENS_ROOT := $(QIMENG_ROOT)/third_party/ThunderKittens
endif
ifeq ($(wildcard $(THUNDERKITTENS_ROOT)/include/kittens.cuh),)
  $(error ThunderKittens not found. Export THUNDERKITTENS_ROOT=<repository-path>)
endif

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc

ifeq ($(GPU),H100)
  GPU_FLAGS := -DKITTENS_HOPPER -arch=sm_90a
else
  $(error Only GPU=H100 is supported)
endif

QIMENG_PYTHON ?= python3
PYBIND_INC := $(shell $(QIMENG_PYTHON) -m pybind11 --includes)
PY_LDFLAGS := $(shell $(QIMENG_PYTHON) -c 'import sysconfig; print("-L" + sysconfig.get_config_var("LIBDIR") + " " + (sysconfig.get_config_var("LIBS") or ""))')
EXT_SUFFIX := $(shell $(QIMENG_PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

# ThunderKittens TMA descriptors use the CUDA Driver API; keep -lcuda.
LIBS := -lcuda -lcudadevrt -lcudart_static -lrt -lpthread -ldl

NVCCFLAGS := -std=c++20 -O3 -DNDEBUG $(GPU_FLAGS) \
             --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr \
             -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing \
             -Xptxas=--verbose -Xptxas=--warn-on-spills -Xnvlink=--verbose -lineinfo \
             -I$(THUNDERKITTENS_ROOT)/include -I. \
             $(PYBIND_INC) -shared -Xcompiler=-fPIC $(PY_LDFLAGS) $(LIBS)

# Enable the barrier watchdog for hang diagnosis with:
#     make clean && make NVCCFLAGS_EXTRA=-UNDEBUG
NVCCFLAGS += $(NVCCFLAGS_EXTRA)

# Keep explicit dependencies exhaustive so source changes rebuild the DSO.
DEPS := Makefile \
        common/pi0_config.cuh common/hardware_config.cuh \
        common/globals.cuh common/instruction_list.cuh \
        include/config.cuh include/util.cuh include/megakernel.cuh \
        include/consumer.cuh include/loader.cuh include/storer.cuh \
        include/launcher.cuh include/noop.cuh include/barrier.cuh include/debug/barrier_watchdog.cuh \
        include/controller/controller.cuh include/controller/instruction_fetch.cuh \
        include/controller/page_allocator.cuh include/controller/semaphore_constructor.cuh \
        include/gmem_dep.cuh \
        include/pipelines/matmul_pipeline.cuh include/pipelines/wgmma_compat.cuh \
        backbone/siglip/siglip_ln.cu backbone/siglip/siglip_gemm.cu backbone/siglip/siglip_attn.cu \
        backbone/siglip/variants/view1/config.cuh backbone/siglip/variants/view2/config.cuh \
        backbone/siglip/variants/view3/config.cuh

TARGET_SIGLIP_VIEW1 := mk_siglip_view1$(EXT_SUFFIX)
TARGET_SIGLIP_VIEW2 := mk_siglip_view2$(EXT_SUFFIX)
TARGET_SIGLIP_VIEW3 := mk_siglip_view3$(EXT_SUFFIX)
TARGET_GEMMA  := mk_gemma$(EXT_SUFFIX)
TARGET_AE     := mk_ae$(EXT_SUFFIX)

.PHONY: all clean
all: $(TARGET_SIGLIP_VIEW1) $(TARGET_SIGLIP_VIEW2) $(TARGET_SIGLIP_VIEW3) $(TARGET_GEMMA) $(TARGET_AE)

$(TARGET_SIGLIP_VIEW1): mk_siglip.cu $(DEPS) common/globals.cuh common/instruction_list.cuh
	$(NVCC) $(NVCCFLAGS) -DPI0_SIGLIP_VARIANT=1 -DPI0_SIGLIP_PY_MODULE=mk_siglip_view1 mk_siglip.cu -o $@

$(TARGET_SIGLIP_VIEW2): mk_siglip.cu $(DEPS) common/globals.cuh common/instruction_list.cuh
	$(NVCC) $(NVCCFLAGS) -DPI0_SIGLIP_VARIANT=2 -DPI0_SIGLIP_PY_MODULE=mk_siglip_view2 mk_siglip.cu -o $@

$(TARGET_SIGLIP_VIEW3): mk_siglip.cu $(DEPS) common/globals.cuh common/instruction_list.cuh
	$(NVCC) $(NVCCFLAGS) -DPI0_SIGLIP_VARIANT=3 -DPI0_SIGLIP_PY_MODULE=mk_siglip_view3 mk_siglip.cu -o $@

GEMMA_SRCS := backbone/gemma/gemma_rmsn.cu backbone/gemma/gemma_attn.cu \
              backbone/gemma/gemma_qkv_rope.cu backbone/gemma/gemma_ffn_dual.cu \
              backbone/gemma/gemma_resid_gemm.cu backbone/gemma/gemma_down_splitk.cu

$(TARGET_GEMMA): mk_gemma.cu $(DEPS) common/gemma_globals.cuh common/gemma_instruction_list.cuh \
                 $(GEMMA_SRCS)
	$(NVCC) $(NVCCFLAGS) mk_gemma.cu -o $@

AE_SRCS := action_expert/ae_gemm.cu action_expert/ae_rmsn.cu \
           action_expert/ae_qkv_rope.cu action_expert/ae_ffn_dual.cu \
           action_expert/ae_action.cu action_expert/ae_attn.cu

$(TARGET_AE): mk_ae.cu $(DEPS) common/ae_globals.cuh common/ae_instruction_list.cuh \
              $(AE_SRCS)
	$(NVCC) $(NVCCFLAGS) mk_ae.cu -o $@

clean:
	rm -f mk_siglip$(EXT_SUFFIX) $(TARGET_SIGLIP_VIEW1) $(TARGET_SIGLIP_VIEW2) $(TARGET_SIGLIP_VIEW3) $(TARGET_GEMMA) $(TARGET_AE)
