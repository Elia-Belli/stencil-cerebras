#!/usr/bin/env bash
# Weak scaling experiment for Cerebras stencil compilation
# Usage: bash weak_scaling.sh

set -euo pipefail

# Baseline parameters
BASE_KERNEL_X=2
BASE_KERNEL_Y=2

BASE_ROWS=128
BASE_COLS=128
ITERATIONS=1000

# List of scale factors (e.g., 1x, 2x, 4x problem sizes)
SCALE_FACTORS=(1 2 4 8 16 32 64 128 256)

KERNELS=(2 4 8 16 32 64 128 256 512 700)
INPUTS=(128 256 512 1024 2048 4096 8192 16384 32768 44800)

# Output log
program_path="."
LOGFILE="weak_scaling_results.log"
echo "Weak scaling results - $(date)" > "$LOGFILE"

echo "Running weak scaling study..."
for SCALE in "${SCALE_FACTORS[@]}"; do
    # Scale input rows/cols proportionally
    ROWS=$((BASE_ROWS * SCALE))
    COLS=$((BASE_COLS * SCALE))
    KERNEL_X=$((BASE_KERNEL_X * SCALE))
    KERNEL_Y=$((BASE_KERNEL_Y * SCALE))

    OUT_JSON="artifact_scale${SCALE}.json"

    echo "--------------------------------------------------" | tee -a "$LOGFILE"
    echo "Scale factor: ${SCALE}" | tee -a "$LOGFILE"
    echo "Input size: ${ROWS}x${COLS}" | tee -a "$LOGFILE"

    python "$program_path"/appliance_compile.py \
        --kernel-dim-x "$KERNEL_X" \
        --kernel-dim-y "$KERNEL_Y" \
        --inp-rows "$ROWS" \
        --inp-cols "$COLS" \
        --iterations "$ITERATIONS" \

    python "$program_path"/appliance_run.py

done

echo "--------------------------------------------------" | tee -a "$LOGFILE"
echo "Weak scaling experiment completed. Results in $LOGFILE"
