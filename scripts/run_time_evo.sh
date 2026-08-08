#!/bin/bash
set -e

# Jump to the directory where this script is located
cd "$(dirname "$0")"

# Path to the executable
EXEC="../build/bin/time_evo"

# System parameters
SIZE=1000000
L=100000.0
ALPHA=0.27
H0=1.1       # Fixed h0 for the time evolution study
HA=0.0
OMEGA=0.0
DT=0.3926990816987
MAX_POWER=30 # Will compute up to 2^22 steps

# Hardware and output
BACKEND="gpu"
EPS=0.05
SEED=42
PRECISION="double"

# Base output directory
BASE_OUT_DIR="../output/time_evo_${L}"
OUT_DIR="$BASE_OUT_DIR"

# Incremental copy suffix logic
count=1
while [ -d "$OUT_DIR" ]; do
    OUT_DIR="${BASE_OUT_DIR}(${count})"
    ((count++))
done

# Create the unique output directory
mkdir -p "$OUT_DIR"
PREFIX="${OUT_DIR}/time_data"

echo "Starting time evolution in double precision..."
echo "Saving outputs to: $OUT_DIR"

$EXEC $SIZE $L $ALPHA $H0 $HA $OMEGA $DT $MAX_POWER $BACKEND $PREFIX $EPS $SEED $PRECISION

echo "Time evolution complete."