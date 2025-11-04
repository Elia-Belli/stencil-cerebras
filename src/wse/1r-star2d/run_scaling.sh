#!/usr/bin/env bash
# Weak scaling experiment for Cerebras stencil compilation
# Usage: bash weak_scaling.sh

set -euo pipefail

# Parameters
ITERATIONS=10000

KERNELS=(2 4 8 16 32 64 128 256 512 700)
INPUTS=(128 256 512 1024 2048 4096 8192 16384 32768 44800)

# Output log
LOGFILE="weak_scaling_results.log"
echo "Weak scaling results - $(date)" > "$LOGFILE"

echo "Running weak scaling study..."
for i in "${!INPUTS[@]}"; do
    # Scale input rows/cols proportionally
    ROWS=${INPUTS[$i]}
    COLS=${INPUTS[$i]}
    KERNEL_X=${KERNELS[$i]}
    KERNEL_Y=${KERNELS[$i]}

    echo "--------------------------------------------------" | tee -a "$LOGFILE"
    echo "Input size: ${ROWS}x${COLS}, Kernel: ${KERNEL_X}x${KERNEL_Y}" | tee -a "$LOGFILE"

    python "appliance_compile.py" \
        --kernel-dim-x "$KERNEL_X" \
        --kernel-dim-y "$KERNEL_Y" \
        --inp-rows "$ROWS" \
        --inp-cols "$COLS" \
        --iterations "$ITERATIONS"

    python "appliance_run.py"

done

echo "--------------------------------------------------" | tee -a "$LOGFILE"
echo "Weak scaling experiment completed. Results in $LOGFILE"
