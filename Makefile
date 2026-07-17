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

# The fused dispatch + per-arch launch TUs (fused_{cuda,hip}.c, {i8,fp8}_arch.c) are pure C host
# code — no device kernels of their own — so they build with $(CC). The rocm branch adds the HIP
# headers to CPPFLAGS (below) so fused_hip.c / fp8_arch.c find <hip/hip_runtime.h>.

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
    # fused_hip.o compiles to the fp8 null-backend stub here (HAVE_ROCM undefined).
    FUSED_OBJ = $(BUILD_DIR)/fused_cuda.o $(BUILD_DIR)/fused_hip.o
    # Generated INT8 header names (one per fused kernel; they bake the model dims). Passed to
    # i8_arch.c as -D so the single per-arch source (compiled once per sm) resolves them from
    # artifacts/<sm>/. (The factored-LSTM fused path was removed — transformer kernels only.)
    I8_ROTARY_HDR   = gemm_i8_rotary_N1536_K512_H8D64R64S2048.h
    I8_MLP_HDR      = gemm_i8_dual_silu_N2048_K512.h
    CUDART_VER := $(shell grep -E 'define +CUDART_VERSION' $(CUDA_INC)/cuda_runtime_api.h 2>/dev/null | grep -oE '[0-9]+' | head -1)
    ifeq ($(shell [ "$(CUDART_VER)" -ge 12000 ] 2>/dev/null && echo 1),1)
        # Per-arch launch vtable (i8_arch_<sm>.o, one per SM arch) + the CuTe-exported kernel objects
        # (symbol-rewritten below). Adding an SM arch = add its i8_arch_<sm>.o + kernel .o's here,
        # and a g_archs[] row in fused_cuda.c.
        AOT_OBJ = $(BUILD_DIR)/i8_arch_sm80.o \
                  $(BUILD_DIR)/gemm_i8_rotary_N1536_K512_H8D64R64S2048.o \
                  $(BUILD_DIR)/gemm_i8_dual_silu_N2048_K512.o
    endif
# make rocm=1 builds the HIP backend (hipcc). Set ROCM_ARCH, e.g.
#   make rocm=1 ROCM_ARCH="--offload-arch=gfx1200"
else ifdef rocm
	ROCM_ROOT ?= /opt/rocm
	ROCM_LIB ?= $(ROCM_ROOT)/lib
	ROCM_INC ?= $(ROCM_ROOT)/include
	HIPCC ?= $(ROCM_ROOT)/bin/hipcc
	ROCM_CFLAGS += -g -Wall $(ROCM_ARCH)
	ROCM_OBJ += $(BUILD_DIR)/nn_hip.o
	GPU_LIB = $(BUILD_DIR)/hip_code.a
	GPU_OBJ = $(ROCM_OBJ)
	SHARED_LINK = $(HIPCC) -shared $(ROCM_ARCH)
	ROCM_LDFLAGS = -L$(ROCM_LIB) -lamdhip64 -lrt -ldl
	# HIP headers for the pure-C host dispatch (fused_hip.c / fp8_arch.c compiled with $(CC)).
	CPPFLAGS += -DHAVE_ROCM=1 -I $(ROCM_INC) -D__HIP_PLATFORM_AMD__
	# fp8 fused dispatch (fused_hip.o, real backend) + fused_cuda.o (int8 null stub here).
	FUSED_OBJ = $(BUILD_DIR)/fused_cuda.o $(BUILD_DIR)/fused_hip.o
	# AOT fp8 kernels: one HSACO per concrete RDNA4 arch (gfx12-generic won't compile through
	# FlyDSL's MLIR and per-chip objects don't cross-load), embedded into libfluke.a and loaded
	# by gcnArchName in fused_hip.c. Baked shapes match the CUDA int8 kernels.
	ROTARY_HSACO = rdna_fp8_gemm_rotary_N1536_K512_TM64_TN256.hsaco
	MLP_HSACO    = rdna_fp8_dual_gemm_silu_N2048_K512_TM32_TN256.hsaco
	# Per-arch fused-fp8 wrapper objects (fp8_arch_<arch>.o, one launch vtable each) + the embedded
	# HSACO images they load. Adding a chip = add its fp8_arch_<arch>.o + embed_*_<arch>.o here
	# and a g_archs[] row in fused_hip.c (one source, src/fp8_arch.c, compiled once per arch).
	AOT_OBJ = $(BUILD_DIR)/fp8_arch_gfx1200.o    $(BUILD_DIR)/fp8_arch_gfx1201.o \
	          $(BUILD_DIR)/embed_rotary_gfx1200.o $(BUILD_DIR)/embed_rotary_gfx1201.o \
	          $(BUILD_DIR)/embed_mlp_gfx1200.o    $(BUILD_DIR)/embed_mlp_gfx1201.o
else
	GPU_LIB = $(BUILD_DIR)/cpu_decoy.a
endif

# The fused DSL kernels (int8 CUDA / fp8 HIP) are OPT-IN. They need AOT artifacts exported
# per target arch (see README), which aren't portable — so by default we compile
# fused_{cuda,hip}.o to their null-backend stubs (fluke_*_select returns NULL, callers keep
# fp16) and link no artifact objects. This lets `make cuda=1` / `make rocm=1` succeed out of
# the box (and on archs with no fused backend at all, e.g. RDNA3 gfx1100). Pass fused=1 once
# you've exported the artifacts for the target arch to link the real fused kernels.
ifneq ($(fused),1)
	CPPFLAGS += -DFLUKE_NO_FUSED
	AOT_OBJ =
endif

ifdef debug
	CPPFLAGS += -DDEBUG=1
	CFLAGS += -fopenmp
endif

.PHONY: all shared fp8_shared clean distclean

all: $(STATICLIB)
shared: $(SHAREDLIB)

# fp8 fused ABI as a standalone .so for the ctypes-based fly/ tests (rocm=1 only). Bundles the
# fp8 dispatch + the embedded per-arch HSACOs; loaded by fly/test_*.py to exercise the real
# C ABI (fluke_fp8_select / fluke_qkv_rotary_fp8_gpu / fluke_gated_mlp_fp8_gpu) on device ptrs.
FP8_SHAREDLIB = $(BUILD_DIR)/libfluke_fp8.so
fp8_shared: $(FP8_SHAREDLIB)
$(FP8_SHAREDLIB): $(BUILD_DIR)/fused_hip.o $(AOT_OBJ)
	$(HIPCC) -shared -fPIC $^ $(ROCM_LDFLAGS) -o $@

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

# Fused-int8 dispatch (pure C; arch-neutral — no artifact includes). Self-guards on CUDART>=12
# (else null-backend stub). The per-arch launch code lives in i8_arch.c below.
$(BUILD_DIR)/fused_cuda.o: src/fused_cuda.c src/i8_ops.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I . $(DEPFLAGS) -c $< -o $@

# Per-arch fused-int8 launch vtable: ONE source (src/i8_arch.c) compiled once per SM arch, with the
# arch tag + that arch's generated headers passed as -D. Exports fluke_i8_ops_<sm> (bound by compute
# capability in fused_cuda.c). Pure C host code (the device kernels are the linked-in CuTe .o's);
# -I . resolves the artifacts/<sm>/*.h includes. The headers are the real prerequisites (make fails
# clearly if the arch wasn't exported yet).
$(BUILD_DIR)/i8_arch_%.o: src/i8_arch.c src/i8_ops.h \
                          artifacts/%/$(I8_ROTARY_HDR) artifacts/%/$(I8_MLP_HDR) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I . \
	    -DFLUKE_I8_ARCH_NAME=$* \
	    -DFLUKE_I8_ROTARY_HDR='"artifacts/$*/$(I8_ROTARY_HDR)"' \
	    -DFLUKE_I8_MLP_HDR='"artifacts/$*/$(I8_MLP_HDR)"' \
	    $(DEPFLAGS) -c $< -o $@

# Fused-fp8 dispatch (pure C; arch-neutral — no artifact includes). Real backend on rocm (CPPFLAGS
# carries the HIP headers), fp8 null-backend stub elsewhere (HAVE_ROCM undefined).
$(BUILD_DIR)/fused_hip.o: src/fused_hip.c src/fp8_ops.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I . $(DEPFLAGS) -c $< -o $@

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

# Per-arch fused-fp8 wrapper object: ONE source (src/fp8_arch.c) compiled once per arch, with the
# arch tag + that arch's generated headers passed as -D. Exports the fluke_fp8_ops_<arch> vtable
# (bound by gcnArchName in fused_hip.c). Pure C host code — it only drives the HIP module/launch
# API (no __global__ kernels of its own) — so it builds with $(CC) (CPPFLAGS carries the HIP
# headers). Builds without the target GPU present. The generated headers are the real prerequisites
# (make fails clearly if the arch wasn't exported yet).
$(BUILD_DIR)/fp8_arch_%.o: src/fp8_arch.c src/fp8_ops.h \
                           artifacts/%/$(ROTARY_HSACO:.hsaco=.h) artifacts/%/$(MLP_HSACO:.hsaco=.h) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I . \
	    -DFLUKE_FP8_ARCH_NAME=$* \
	    -DFLUKE_FP8_ROTARY_HDR='"artifacts/$*/$(ROTARY_HSACO:.hsaco=.h)"' \
	    -DFLUKE_FP8_MLP_HDR='"artifacts/$*/$(MLP_HSACO:.hsaco=.h)"' \
	    $(DEPFLAGS) -c $< -o $@

# Embed a per-arch HSACO image into an object exposing arch-qualified symbols
# fluke_fp8_<role>_<gfxNNNN>{,_end} (fused_hip.c loads them via hipModuleLoadData). Symbols are
# arch-qualified so gfx1200/gfx1201 images don't collide at link. Self-contained: no runtime file.
# The stem (%) is the gfx arch. Fails with a clear message if the artifact wasn't exported yet.
$(BUILD_DIR)/embed_rotary_%.o: artifacts/%/$(ROTARY_HSACO) | $(BUILD_DIR)
	@test -f $< || { echo "[fluke] missing $< — run fly/rdna4/rotary/export_fp8_gemm_rotary.py first"; exit 1; }
	printf '.section .rodata\n.global fluke_fp8_rotary_%s\nfluke_fp8_rotary_%s:\n.incbin "%s"\n.global fluke_fp8_rotary_%s_end\nfluke_fp8_rotary_%s_end:\n' $* $* '$(abspath $<)' $* $* > $(BUILD_DIR)/embed_rotary_$*.s
	$(CC) -c $(BUILD_DIR)/embed_rotary_$*.s -o $@

$(BUILD_DIR)/embed_mlp_%.o: artifacts/%/$(MLP_HSACO) | $(BUILD_DIR)
	@test -f $< || { echo "[fluke] missing $< — run fly/rdna4/dual_gemm_silu/export_fp8_dual_gemm_silu.py first"; exit 1; }
	printf '.section .rodata\n.global fluke_fp8_mlp_%s\nfluke_fp8_mlp_%s:\n.incbin "%s"\n.global fluke_fp8_mlp_%s_end\nfluke_fp8_mlp_%s_end:\n' $* $* '$(abspath $<)' $* $* > $(BUILD_DIR)/embed_mlp_$*.s
	$(CC) -c $(BUILD_DIR)/embed_mlp_$*.s -o $@

# pull in auto-generated header dependencies (.d files emitted by -MMD)
-include $(BUILD_DIR)/*.d

clean:
	rm -rf $(BUILD_DIR)/*

distclean: clean
	git clean -f -X
