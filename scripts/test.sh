#!/bin/bash

set -e

# Fixed Mathematical / Stepping Parameters
ALPHA=0.27
H0=1.1
HA=3
OMEGA=1.0
DT=0.1

SIZES=(10 100 1000 10000)
BACKENDS=("cpu" "gpu")
PRECISIONS=("single" "double")

RESULTS_FILE="benchmark_results.txt"
echo "Domain Wall Model Benchmark Results" > $RESULTS_FILE
echo "===================================" >> $RESULTS_FILE

for size in "${SIZES[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        for precision in "${PRECISIONS[@]}"; do
            
            output_file="output_${size}_${backend}_${precision}.dat"
            
            echo "Running -> Size: $size | Backend: $backend | Precision: $precision"
            
            echo "-----------------------------------" >> $RESULTS_FILE
            echo "Config: Size=$size, Backend=$backend, Precision=$precision" >> $RESULTS_FILE
            
            # Record wall clock time and peak memory using /usr/bin/time
            /usr/bin/time -v ./dummy $size $ALPHA $H0 $HA $OMEGA $DT $backend $precision $output_file 2>> $RESULTS_FILE
            
        done
    done
done

echo "Benchmarks complete. Outputs written to .dat files and timings stored in $RESULTS_FILE."
