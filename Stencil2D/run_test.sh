#!/usr/bin/env bash

set -eu

arch="wse3"
iterations=1 # simulation iterations

# Strong Scaling Test 
inp_rows=256
inp_cols=256

# for i in {4,8,12,16,20,24,28,32}; do
#     kernel_dim_x=$i
#     kernel_dim_y=$i

#     source ./commands.sh
#     run_worker
# done

# Weak Scaling Test
# for i in {4,8,12}; do
#     kernel_dim_x=$i
#     kernel_dim_y=$i
#     inp_rows=$((i * 71))
#     inp_cols=$((i * 71))
    
#     source ./commands.sh
#     run_worker
# done

kernel_dim_x=4
kernel_dim_y=4
for((i=20; i<=280; i+=20)); do
    inp_rows=$i
    inp_cols=$i
    
    source ./commands.sh
    run_worker
done

# arch="wse3"
# for i in {4,8,12}; do
#     kernel_dim_x=$i
#     kernel_dim_y=$i
#     inp_rows=$((i * 71))
#     inp_cols=$((i * 71))
    
#     source ./commands.sh
#     run_worker
# done