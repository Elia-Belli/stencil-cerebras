#!/usr/bin/env bash

set -e

# required by memcpy infrastructure: (dim_x >= 7 + width) (dim_y >= 2 + height) (x >= 4) (y >= 1)
cslc --arch=wse2 ./layout.csl \
--fabric-dims=11,6 \
--fabric-offsets=4,1 \
--params=width:2,height:2,M:8,N:8,iterations:1 \
-o out --memcpy --channels 1

cs_python run.py --name out
