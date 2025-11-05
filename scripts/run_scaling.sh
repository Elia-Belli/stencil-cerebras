#!/usr/bin/env bash
# Weak scaling experiment for Cerebras stencil compilation
# Usage: bash weak_scaling.sh

set -euo pipefail

# Parameters
ITERATIONS=1000
KERNELS=(2 4 8 16 32 64 128 256 512 700)
INPUTS=(128 256 512 1024 2048 4096 8192 16384 32768 44800)
CHANNELS=1

LOGFILE="../../../logs/scaling.log"

# -----------------------------------------------------------------------------
# Function: run_weak_scaling
# Runs compile + run for all INPUTS / KERNELS
# -----------------------------------------------------------------------------
run_weak_scaling() {
    echo "Weak scaling results - $(date)" > "$LOGFILE"
    echo "Running weak scaling study..."

    for i in "${!INPUTS[@]}"; do
        local ROWS=${INPUTS[$i]}
        local COLS=${INPUTS[$i]}
        local KERNEL_X=${KERNELS[$i]}
        local KERNEL_Y=${KERNELS[$i]}

        echo "--------------------------------------------------" | tee -a "$LOGFILE"
        echo "Input size: ${ROWS}x${COLS}, Kernel: ${KERNEL_X}x${KERNEL_Y}" | tee -a "$LOGFILE"

        python "appliance_compile.py" \
            --kernel-dim-x "$KERNEL_X" \
            --kernel-dim-y "$KERNEL_Y" \
            --inp-rows "$ROWS" \
            --inp-cols "$COLS" \
            --iterations "$ITERATIONS" \
            --channels "$CHANNELS"

        python "appliance_run.py" | tee -a "$LOGFILE"
    done

    echo "--------------------------------------------------" | tee -a "$LOGFILE"
    echo "Weak scaling experiment completed - $(date)" | tee -a "$LOGFILE"
    echo "Results saved to: $LOGFILE"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

PROGRAMS_PATH="$(pwd)/../src/wse"

cd "$PROGRAMS_PATH/1r-star2d/"
run_weak_scaling

cd "$PROGRAMS_PATH/1r-box2d/"
run_weak_scaling
