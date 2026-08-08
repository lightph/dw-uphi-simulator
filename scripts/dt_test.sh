#!/bin/bash
set -e

# Jump to the directory where this script is located
cd "$(dirname "$0")"

# Path to the executable
EXEC="../build/bin/time_evo"

# System parameters
SIZE=8192
L=8192.0
ALPHA=0.27
H0=1.1
HA=0.0
OMEGA=0.0

# Target maximum time: T = 2^MAX_TIME_POWER
# (e.g., 2^15 = 32768 total physical time)
MAX_TIME_POWER=15 

# Hardware and output
BACKEND="gpu"
EPS=0.05
SEED=42
PRECISION="double"

# Base output directory for the time step test
BASE_OUT_DIR="../output/dt_dependence_${L}_${H0}"
OUT_DIR="$BASE_OUT_DIR"

# Incremental copy suffix logic to prevent overwrites
count=1
while [ -d "$OUT_DIR" ]; do
    OUT_DIR="${BASE_OUT_DIR}(${count})"
    ((count++))
done

# Create the unique output directory
mkdir -p "$OUT_DIR"

echo "Starting time step dependence test..."
echo "Target total time = 2^$MAX_TIME_POWER"
echo "Saving outputs to: $OUT_DIR"

# Loop k from 3 (2^3 = 8) down to -8 (2^-8 = 0.00390625)
for k in {3..-8}; do
    # Calculate dt = 2^k
    DT=$(awk "BEGIN { printf \"%.10f\", 2^$k }")
    
    # Calculate max_power for steps = MAX_TIME_POWER - k
    # This ensures total time (dt * 2^MAX_POWER) is always 2^MAX_TIME_POWER
    MAX_POWER=$((MAX_TIME_POWER - k))
    
    echo "=================================================="
    echo "Running with dt = $DT (2^$k)"
    echo "Max steps power = $MAX_POWER (2^$MAX_POWER steps)"
    
    # Create a unique output directory for this specific dt inside the safely suffixed OUT_DIR
    DT_DIR="${OUT_DIR}/dt_${DT}"
    mkdir -p "$DT_DIR"
    PREFIX="${DT_DIR}/time_data"
    
    # Execute the C++ code
    $EXEC $SIZE $L $ALPHA $H0 $HA $OMEGA $DT $MAX_POWER $BACKEND $PREFIX $EPS $SEED $PRECISION
    
done

echo "=================================================="
echo "All time step runs completed successfully."
echo "Results are stored in: $OUT_DIR"