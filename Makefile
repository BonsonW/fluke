CC = gcc
CXX = c++
AR = ar
CPPFLAGS += -I include/
CFLAGS += -g -Wall -O2 -fPIC
CXXFLAGS += -g -O2 -std=c++17 -fPIC
# auto-generate header dependencies so editing a .h (e.g. nn_kernel_cuda.h) rebuilds dependent objects
DEPFLAGS = -MMD -MP
LDFLAGS += $(LIBS) -lm -lpthread
BUILD_DIR = lib

STATICLIB = $(BUILD_DIR)/libfluke.a
SHAREDLIB = $(BUILD_DIR)/libfluke.so

# CPU-portable objects (always built)
OBJ = $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/nn_cpu.o \

GPU_LIB =
GPU_OBJ =
FUSED_OBJ =
AOT_OBJ =
SHARED_LINK = $(CC) -shared

# make asan=1 enables address sanitiser
ifdef asan
	CFLAGS += -fsanitize=address -fno-omit-frame-pointer
	LDFLAGS += -fsanitize=address -fno-omit-frame-pointer
endif

# make cuda=1 builds the CUDA backend (nvcc). Set CUDA_ARCH, e.g.
#   make cuda=1 CUDA_ARCH="-gencode arch=compute_80,code=sm_80"
ifdef cuda
	CUDA_ROOT ?= /usr/local/cuda
    CUDA_LIB ?= $(CUDA_ROOT)/lib64
    CUDA_INC ?= $(CUDA_ROOT)/include
    CUDA_OBJ += $(BUILD_DIR)/nn_cuda.o
    NVCC ?= $(CUDA_ROOT)/bin/nvcc
    CUDA_CFLAGS += -g -O2 -lineinfo $(CUDA_ARCH) -Xcompiler -Wall -Xcompiler -fPIC
    CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart_static -lrt -ldl
    GPU_LIB = $(BUILD_DIR)/cuda.a
    GPU_OBJ = $(CUDA_OBJ)
    SHARED_LINK = $(NVCC) -shared $(CUDA_ARCH)
    CPPFLAGS += -DHAVE_CUDA=1 -I $(CUDA_INC)
    # Fused DSL kernels use the CUDA 12 library API; only bundle the AOT objects on CUDA >= 12.
    # fused_cuda.o is always built (null-backend stub on older CUDA), so downstream still links.
    FUSED_OBJ = $(BUILD_DIR)/fused_cuda.o
    CUDART_VER := $(shell grep -E 'define +CUDART_VERSION' $(CUDA_INC)/cuda_runtime_api.h 2>/dev/null | grep -oE '[0-9]+' | head -1)
    ifeq ($(shell [ "$(CUDART_VER)" -ge 12000 ] 2>/dev/null && echo 1),1)
        AOT_OBJ = $(BUILD_DIR)/gemm_i8_rotary_N1536_K512_H8D64R64S2048.o \
                  $(BUILD_DIR)/gemm_i8_dual_silu_N2048_K512.o
    endif
# make rocm=1 builds the HIP backend (hipcc). Set ROCM_ARCH, e.g.
#   make rocm=1 ROCM_ARCH="--offload-arch=gfx1200"
else ifdef rocm
	ROCM_ROOT ?= /opt/rocm
	ROCM_LIB ?= $(ROCM_ROOT)/lib
	HIPCC ?= $(ROCM_ROOT)/bin/hipcc
	ROCM_CFLAGS += -g -Wall $(ROCM_ARCH)
	ROCM_OBJ += $(BUILD_DIR)/nn_hip.o
	GPU_LIB = $(BUILD_DIR)/hip_code.a
	GPU_OBJ = $(ROCM_OBJ)
	SHARED_LINK = $(HIPCC) -shared $(ROCM_ARCH)
	ROCM_LDFLAGS = -L$(ROCM_LIB) -lamdhip64 -lrt -ldl
	CPPFLAGS += -DHAVE_ROCM=1
	# No fused (fly) AOT kernels for AMD yet; fused_cuda.o compiles to null-backend stubs.
	FUSED_OBJ = $(BUILD_DIR)/fused_cuda.o
else
	GPU_LIB = $(BUILD_DIR)/cpu_decoy.a
endif

ifdef debug
	CPPFLAGS += -DDEBUG=1
	CFLAGS += -fopenmp
endif

.PHONY: all shared clean distclean

all: $(STATICLIB)
shared: $(SHAREDLIB)

# Static lib bundles the CPU/GPU nn kernels + the fused-int8 dispatch + the AOT kernel
# objects (symbol-rewritten below). This is what slorado links (like libopenfish.a).
$(STATICLIB): $(OBJ) $(GPU_LIB) $(FUSED_OBJ) $(AOT_OBJ)
	cp $(GPU_LIB) $@
	$(AR) rcs $@ $(OBJ) $(FUSED_OBJ) $(AOT_OBJ)

# Shared lib (PIC) for ctypes-based Python tests: just the nn kernels (tests don't use the
# fused DSL ABI), so no AOT objects here.
$(SHAREDLIB): $(OBJ) $(GPU_OBJ)
	$(SHARED_LINK) -o $@ $(OBJ) $(GPU_OBJ) $(LDFLAGS)

$(BUILD_DIR)/misc.o: src/misc.c src/misc.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/error.o: src/error.c include/fluke/fluke_error.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/nn_cpu.o: src/nn_cpu.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

# Fused-int8 dispatch (host C++; includes the generated artifacts/<arch>/*.h). -I . lets the
# `#include "artifacts/sm80/..."` resolve. Self-guards on CUDART>=12 (else null-backend stub).
$(BUILD_DIR)/fused_cuda.o: src/fused_cuda.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -I . $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# cpu decoy (no GPU backend selected)
$(BUILD_DIR)/cpu_decoy.a: | $(BUILD_DIR)
	rm -f $@
	$(AR) -r $@

# cuda
$(BUILD_DIR)/cuda.a: $(CUDA_OBJ)
	$(AR) rcs $@ $^

$(BUILD_DIR)/nn_cuda.o: src/nn_cuda.c | $(BUILD_DIR)
	$(NVCC) -x cu $(CUDA_CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

# AOT fused kernels: rewrite the underscore-prefixed CUDA driver symbols in the exported .o
# to their real ELF names so they resolve against cudart_static / libcuda at the final link.
$(BUILD_DIR)/%.o: artifacts/sm80/%.o | $(BUILD_DIR)
	objcopy \
	  --redefine-sym _cudaDeviceGetAttribute=cudaDeviceGetAttribute \
	  --redefine-sym _cudaFuncSetAttribute=cudaFuncSetAttribute \
	  --redefine-sym _cudaGetDevice=cudaGetDevice \
	  --redefine-sym _cudaKernelSetAttributeForDevice=cudaKernelSetAttributeForDevice \
	  --redefine-sym _cudaLaunchKernelEx=cudaLaunchKernelExC \
	  --redefine-sym _cudaLibraryGetKernel=cudaLibraryGetKernel \
	  --redefine-sym _cudaLibraryLoadData=cudaLibraryLoadData \
	  --redefine-sym _cuKernelGetAttribute=cuKernelGetAttribute \
	  $< $@

# hip
$(BUILD_DIR)/hip_code.a: $(ROCM_OBJ)
	$(HIPCC) $(ROCM_CFLAGS) --emit-static-lib -fPIC --hip-link $^ -o $@

$(BUILD_DIR)/nn_hip.o: src/nn_hip.c | $(BUILD_DIR)
	$(HIPCC) -x hip $(ROCM_CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -fPIC -c $< -o $@

# pull in auto-generated header dependencies (.d files emitted by -MMD)
-include $(BUILD_DIR)/*.d

clean:
	rm -rf $(BUILD_DIR)/*

distclean: clean
	git clean -f -X
