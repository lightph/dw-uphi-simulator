#!/bin/bash
set -e

# Assuming you compile the new cpp file to an executable named 'time_evolution'
EXEC="./time_evo"

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
PREFIX="time_data"
EPS=0.05
SEED=42
PRECISION="double"

echo "Starting time evolution in double precision..."

$EXEC $SIZE $L $ALPHA $H0 $HA $OMEGA $DT $MAX_POWER $BACKEND $PREFIX $EPS $SEED $PRECISION

echo "Time evolution complete."
