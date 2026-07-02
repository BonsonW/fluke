CC = gcc
AR = ar
CPPFLAGS += -I include/
CFLAGS += -g -Wall -O2
# auto-generate header dependencies so editing a .h (e.g. nn_kernel_cuda.h) rebuilds dependent objects
DEPFLAGS = -MMD -MP
LDFLAGS += $(LIBS) -lm -lpthread
BUILD_DIR = lib

STATICLIB = $(BUILD_DIR)/libfluke.a

# CPU-portable objects (always built)
OBJ = $(BUILD_DIR)/misc.o \
	  $(BUILD_DIR)/error.o \
	  $(BUILD_DIR)/nn_cpu.o \

GPU_LIB =

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
    CUDA_OBJ += $(BUILD_DIR)/nn_cuda.o
    NVCC ?= $(CUDA_ROOT)/bin/nvcc
    CUDA_CFLAGS += -g -O2 -lineinfo $(CUDA_ARCH) -Xcompiler -Wall
    CUDA_LDFLAGS = -L$(CUDA_LIB) -lcudart_static -lrt -ldl
    GPU_LIB = $(BUILD_DIR)/cuda.a
    CPPFLAGS += -DHAVE_CUDA=1
# make rocm=1 builds the HIP backend (hipcc). Set ROCM_ARCH, e.g.
#   make rocm=1 ROCM_ARCH="--offload-arch=gfx1200"
else ifdef rocm
	ROCM_ROOT ?= /opt/rocm
	ROCM_LIB ?= $(ROCM_ROOT)/lib
	HIPCC ?= $(ROCM_ROOT)/bin/hipcc
	ROCM_CFLAGS += -g -Wall $(ROCM_ARCH)
	ROCM_OBJ += $(BUILD_DIR)/nn_hip.o
	GPU_LIB = $(BUILD_DIR)/hip_code.a
	ROCM_LDFLAGS = -L$(ROCM_LIB) -lamdhip64 -lrt -ldl
	CPPFLAGS += -DHAVE_ROCM=1
else
	GPU_LIB = $(BUILD_DIR)/cpu_decoy.a
endif

ifdef debug
	CPPFLAGS += -DDEBUG=1
	CFLAGS += -fopenmp
endif

.PHONY: all clean distclean

all: $(STATICLIB)

$(STATICLIB): $(OBJ) $(GPU_LIB)
	cp $(GPU_LIB) $@
	$(AR) rcs $@ $(OBJ)

$(BUILD_DIR)/misc.o: src/misc.c src/misc.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/error.o: src/error.c include/fluke/fluke_error.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

$(BUILD_DIR)/nn_cpu.o: src/nn_cpu.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(DEPFLAGS) -c $< -o $@

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
