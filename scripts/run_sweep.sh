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
OMEGA=1.0
DT=0.1
TRANSIENT=1000.0
EVAL=1000.0

# Sweep parameters
H0_VAL=4.0
HA_MIN=0.0
HA_MAX=10.0
HA_STEPS=101

# Hardware and output
BACKEND="gpu" 
EPS=0.05
SEED=42

# Base output directory
BASE_OUT_DIR="../output/sweep_${L}_${H0_VAL}_${HA_MIN}_${HA_MAX}_${OMEGA}"
OUT_DIR="$BASE_OUT_DIR"

# Incremental copy suffix logic
count=1
while [ -d "$OUT_DIR" ]; do
    OUT_DIR="${BASE_OUT_DIR}(${count})"
    ((count++))
done

# Create the unique output directory
mkdir -p "$OUT_DIR"
PREFIX="${OUT_DIR}/sweep_data.txt"

echo "Starting sequential sweep..."
echo "Saving outputs to: $OUT_DIR"

# Execute the C++ code
$EXEC $SIZE $L $ALPHA $OMEGA $DT $TRANSIENT $EVAL $H0_VAL $HA_MIN $HA_MAX $HA_STEPS "$PREFIX" $BACKEND

echo "Sweep complete."
