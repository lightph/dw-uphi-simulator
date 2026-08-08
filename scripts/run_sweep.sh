#!/bin/bash
set -e

# Jump to the directory where this script is located
cd "$(dirname "$0")"

# Path to the executable
EXEC="../build/bin/sweep"

# System parameters
SIZE=10000
L=1000.0
ALPHA=0.27
HA=0.0
OMEGA=1.0
DT=0.1

# Hardware and output
BACKEND="gpu" 
EPS=0.05
SEED=42
PRECISION="double" 

# New randomize flag (1 for true, 0 for false)
RANDOMIZE=1

# Base output directory
BASE_OUT_DIR="../output/sweep_${L}_${H0}_${HA}_${OMEGA}"
OUT_DIR="$BASE_OUT_DIR"

# Incremental copy suffix logic
count=1
while [ -d "$OUT_DIR" ]; do
    OUT_DIR="${BASE_OUT_DIR}(${count})"
    ((count++))
done

# Create the unique output directory
mkdir -p "$OUT_DIR"
PREFIX="${OUT_DIR}/sweep_data"

echo "Starting sequential sweep in double precision..."
echo "Saving outputs to: $OUT_DIR"

# Single batch for h0 from 0.0 to 10.0 (101 steps)
$EXEC $SIZE $L $ALPHA $HA $OMEGA $DT 0.0 10.0 101 $BACKEND "$PREFIX" $EPS $SEED $PRECISION $RANDOMIZE

echo "Sweep complete."