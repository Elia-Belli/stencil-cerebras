#!/usr/bin/env bash
#
# Simple frontend for running WSE program commands.sh
# Supports switching between simulation and real WSE runs.
#

set -euo pipefail

ROOT_DIR="$(dirname "$(realpath "$0")")/.."
SRC_DIR="$ROOT_DIR/src/wse"

usage() {
    echo "Usage: $0 <program> [sim|real] [options]"
    echo
    echo "Examples:"
    echo "  $0 star2d sim kernel_dim_x=8 kernel_dim_y=8 input_x=64 input_y=64 iterations=10 radius=3"
    echo "  $0 star2d real ip=192.168.0.10:9000 kernel_dim_x=4"
    exit 1
}

# --- Basic argument parsing ---
if [[ $# -lt 2 ]]; then
    usage
fi

PROGRAM="$1"
MODE="$2"
shift 2

PROGRAM_DIR="$SRC_DIR/$PROGRAM"
COMMANDS="$PROGRAM_DIR/commands.sh"

if [[ ! -f "$COMMANDS" ]]; then
    echo "❌ Error: commands.sh not found for '$PROGRAM'"
    exit 1
fi

# --- Set up execution mode ---
case "$MODE" in
    sim)
        export ip=${ip:-0.0.0.0:9000}
        export suppress_simfab_trace=${suppress_simfab_trace:-false}
        export simfab_numthreads=${simfab_numthreads:-5}
        echo "🧪 Running in simulation mode (simfabric)"
        ;;
    real)
        if ! env | grep -q '^ip='; then
            echo "⚠️  Note: You must provide ip=<IP:PORT> for real WSE runs."
        fi
        echo "💎 Running on real WSE hardware"
        ;;
    *)
        echo "❌ Invalid mode '$MODE' — use 'sim' or 'real'"
        usage
        ;;
esac

# --- Apply variable overrides (any VAR=value pairs) ---
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        export "$arg"
    else
        echo "⚠️  Ignoring invalid argument: $arg"
    fi
done

# --- Run ---
echo "▶️  Executing $PROGRAM/commands.sh"
(cd "$PROGRAM_DIR" && ./commands.sh)
