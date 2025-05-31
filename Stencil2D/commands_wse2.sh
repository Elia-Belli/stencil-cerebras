#!/usr/bin/env bash

set -e

# required by memcpy infrastructure: (dim_x >= 7 + width) (dim_y >= 2 + height) (x >= 4) (y >= 1)
cslc --arch=wse2 ./layout.csl \
--fabric-dims=11,6 \
--fabric-offsets=4,1 \
--params=kernel_dim_x:4,kernel_dim_y:4,M:256,N:256,iterations:1 \
-o out --memcpy --channels 1 

cs_python run.py --name out
