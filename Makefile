# Compilers
CC := gcc
CUDACC := nvcc

# Flags
CFLAGS = -g
OPT := -O3
LIBS := -lm
ARCH := -arch=sm_50

# Targets to build
OBJS = 	STENCIL_c\
				STENCIL_cuda

all: $(OBJS)

STENCIL_c: ./src/cpu/stencil.c
	${CC} ${CFLAGS} ${OPT} $< -o ./out/bin/$@

STENCIL_cuda: ./src/cuda/stencil.cu
	${CUDACC} $(LIBS) $(ARCH) -Wno-deprecated-gpu-targets $< -o ./out/bin/$@

clean:
	rm ./out/bin/STENCIL_*