#!/usr/bin/env bash

set -e

# required by memcpy infrastructure: (dim_x >= 7 + width) (dim_y >= 2 + height) (x >= 4) (y >= 1)
cslc --arch=wse2 ./layout.csl \
--fabric-dims=15,10 \
--fabric-offsets=4,1 \
--params=kernel_dim_x:4,kernel_dim_y:4,M:32,N:32,iterations:10 \
-o out --memcpy --channels 1

cs_python run.py --name out
