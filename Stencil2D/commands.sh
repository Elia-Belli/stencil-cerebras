# Only set variables if not already defined (allows overrides)
: "${kernel_dim_x:=2}"
: "${kernel_dim_y:=2}"
: "${inp_rows:=2}"
: "${inp_cols:=2}"
: "${iterations:=1}"
: "${arch:=wse2}"

fabric_dim_x=$((7 + kernel_dim_x))
fabric_dim_y=$((2 + kernel_dim_y))

run_worker() {
    echo "Running with kernel: ${kernel_dim_x}x${kernel_dim_y}, input: ${inp_rows}x${inp_cols}, iterations: $iterations"

    cslc --arch=$arch ./layout.csl \
    --fabric-dims=$fabric_dim_x,$fabric_dim_y \
    --fabric-offsets=4,1 \
    --params=kernel_dim_x:$kernel_dim_x,kernel_dim_y:$kernel_dim_y,\
M:$inp_rows,N:$inp_cols,iterations:$iterations \
    -o out --memcpy --channels 1

    cs_python run.py --name out  --verify #--verbose --traces
}

# If script is sourced, don't auto-run
# If executed directly, run the function
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    set -eu  # Only apply strict mode when run directly
    run_worker
fi
