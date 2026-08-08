#!/bin/bash
set -e

EXEC="./sweep"

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

echo "Starting sequential sweep in double precision..."

# Single batch for h0 from 0.0 to 10.0 (101 steps)
$EXEC $SIZE $L $ALPHA $HA $OMEGA $DT 0.0 10.0 101 $BACKEND "sweep_data" $EPS $SEED $PRECISION $RANDOMIZE

echo "Sweep complete."
