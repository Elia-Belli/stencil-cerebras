#!/usr/bin/env bash

set -eu

arch="wse2"
iterations=1 # simulation iterations

# Strong Scaling Test 
inp_rows=256
inp_cols=256

for i in {4,8,12,16,20,24,28,32}; do
    kernel_dim_x=$i
    kernel_dim_y=$i

    source ./commands.sh
    run_worker
done

# Weak Scaling Test
for i in {4,8,12,16,20,24,28,32}; do
    kernel_dim_x=$i
    kernel_dim_y=$i
    inp_rows=$((i * 64))
    inp_cols=$((i * 64))
    
    source ./commands.sh
    run_worker
done