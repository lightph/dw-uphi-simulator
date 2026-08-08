#!/bin/bash

# System Parameters
SIZE=32768
L=10000.0 # Domain length
ALPHA=0.27
HA=0
OMEGA=1.0
DT=0.628318530718

# Simulation Parameters
H0=4
W=10000 # Steps per window for computing the rolling mean of spatial variance
N=1000   # Total number of windows to simulate and log

# Execution Parameters
BACKEND="gpu"
OUTPUT_FILE="windows_output.txt"
EPS=0.05 # Initial noise amplitude
SEED=42  # RNG seed
PRECISION="double"

./test_windows \
  $SIZE $L $ALPHA $HA $OMEGA $DT $H0 $W $N \
  $BACKEND $OUTPUT_FILE $EPS $SEED $PRECISION
