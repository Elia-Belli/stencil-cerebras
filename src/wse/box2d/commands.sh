#!/bin/bash

: "${kernel_dim_x:=2}"
: "${kernel_dim_y:=2}"
: "${inp_rows:=16}"
: "${inp_cols:=16}"
: "${iterations:=1}"
: "${radius:=3}"
: "${arch:=wse3}"

fabric_dim_x=$((7 + kernel_dim_x))
fabric_dim_y=$((2 + kernel_dim_y))

#stencil=(0 1 0 0 0) 

run_worker() {
    cslc --arch=$arch layout.csl \
    --fabric-dims=$fabric_dim_x,$fabric_dim_y \
    --fabric-offsets=4,1 \
    --params=kernel_dim_x:$kernel_dim_x,kernel_dim_y:$kernel_dim_y,\
radius:$radius,M:$inp_rows,N:$inp_cols,iterations:$iterations \
    -o out --memcpy --channels 1

    echo ""
    echo "Running with kernel: ${kernel_dim_x}x${kernel_dim_y}, input: ${inp_rows}x${inp_cols}, stencil radius: ${radius}, iterations: $iterations"

    cs_python run.py --name out --arch=$arch --stencil "${stencil[@]}" --verify #--verbose #--traces 
}

# If script is sourced, don't auto-run
# If executed directly, run the function
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -eu  # Only apply strict mode when run directly
    run_worker
fi
