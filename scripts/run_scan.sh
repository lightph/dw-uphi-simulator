#!/bin/bash

# System Parameters
SIZE=2048
L=1000.0  # 2*pi
ALPHA=0.27
HA=3
OMEGA=1.0
DT=0.1

# Parameter Scan Range
H0_MIN=0.
H0_MAX=5
N_H0=51          # Number of steps between min and max

# Convergence Parameters
W=100000           # Steps per window for averaging
N=10             # Size of variance queue
TOLERANCE=1e-6   # Std deviation threshold for settling

# Execution Parameters
BACKEND="gpu"
OUTPUT_FILE="scan_output.txt"
EPS=0.05         # Initial noise amplitude
SEED=42          # RNG seed
PRECISION="double"

./scan \
    $SIZE $L $ALPHA $HA $OMEGA $DT $H0_MIN $H0_MAX $N_H0 \
    $W $N $TOLERANCE $BACKEND $OUTPUT_FILE $EPS $SEED $PRECISION
