# Compilers
CC := gcc
CUDACC := nvcc
CSL := cslc

# Flags
CFLAGS = -g
OPT := -O3
LIBS := -lm
ARCH := -arch=sm_50

# Targets to build
OBJS = 	c\
				star2d box2d

all: $(OBJS)

c: src/cpu/stencil.c
	${CC} ${CFLAGS} ${OPT} $< -o ./build/$@


# === WSE Configuration Variables ===
arch            ?= wse3
kernel_dim_x    ?= 2
kernel_dim_y    ?= 2
radius          ?= 1
inp_rows        ?= 8
inp_cols        ?= 8
iterations      ?= 1

# === Derived / Fixed Values ===
channels        := 1
out_dir         := build

fabric_dim_x := $(shell echo $$(( $(kernel_dim_x) + 7 )))
fabric_dim_y := $(shell echo $$(( $(kernel_dim_y) + 2 )))


star2d: src/wse/star2d/layout.csl
	${CSL} --arch=$(arch) $< -o $(out_dir)/$@ \
	--fabric-dims=$(fabric_dim_x),$(fabric_dim_y) \
	--fabric-offsets=4,1 \
	--params=kernel_dim_x:$(kernel_dim_x),kernel_dim_y:$(kernel_dim_y),\
	M:$(inp_rows),N:$(inp_cols),iterations:$(iterations),radius:$(radius) \
	--memcpy --channels $(channels)

box2d: src/wse/box2d/layout.csl
	${CSL} --arch=$(arch) $< -o $(out_dir)/$@ \
	--fabric-dims=$(fabric_dim_x),$(fabric_dim_y) \
	--fabric-offsets=4,1 \
	--params=kernel_dim_x:$(kernel_dim_x),kernel_dim_y:$(kernel_dim_y),\
	M:$(inp_rows),N:$(inp_cols),iterations:$(iterations),radius:$(radius) \
	--memcpy --channels $(channels) 

clean:
	rm ./out/bin/STENCIL_*