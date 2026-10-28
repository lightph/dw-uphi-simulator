#!/usr/bin/env bash
set -eo pipefail

# Jump to the project root directory
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)
EXEC="${PROJECT_DIR}/build_local/bin/sweep"

# Ensure the executable exists
if [ ! -f "$EXEC" ]; then
  echo "Error: Executable '$EXEC' not found. Please run ./compile_local.sh first."
  exit 1
fi

# --- System & Physics Parameters ---
SIZE=512
L=256
ALPHA=0.27
OMEGA=1.0

# 1/10th of the cluster DT
DT=0.0196349540849362

# Simulation Durations
TRANSIENT=131072
# Define EVAL as an exact multiple of 4*pi/omega (twice the driving period)
EVAL=$(awk -v omega=$OMEGA -v cycles=1000 'BEGIN { printf "%.10f", cycles * 4.0 * 3.14159265358979323846 / omega }')

# Define h0 grid parameters (0 to 10, dh0=0.1 -> 100 intervals = 101 points)
H0_MIN=0.0
H0_MAX=10.0
H0_STEPS=1000

# Define ha grid parameters (Locked to 3.0)
HA_MIN=3.0
HA_MAX=3.0
HA_STEPS=1

# --- Tidy Output & Log Directory Management ---
# Uses a timestamp to keep outputs unique since there is no SLURM Job ID
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MASTER_DIR="${PROJECT_DIR}/output/sweep_local_${TIMESTAMP}"
mkdir -p "$MASTER_DIR"

# --- Save Run Parameters ---
PARAMS_FILE="${MASTER_DIR}/run_parameters.txt"
cat <<EOF >"$PARAMS_FILE"
==========================================================
Run Parameters for Local Sweep
==========================================================
Timestamp:      $(date)
Executable:     $EXEC
Output Dir:     $MASTER_DIR
----------------------------------------------------------
System Size:    $SIZE
Length (L):     $L
Alpha:          $ALPHA
Omega:          $OMEGA
dt:             $DT
Transient Time: $TRANSIENT
Eval Time:      $EVAL
h0 Range:       [$H0_MIN, $H0_MAX] with $H0_STEPS intervals
ha Range:       [$HA_MIN, $HA_MAX] with $HA_STEPS intervals (Locked to $HA_MIN)
Backend:        CPU
==========================================================
EOF

echo "Starting local sweep. Outputs will be saved to: $MASTER_DIR"
echo "----------------------------------------------------------"

# --- Loop over H0 ---
START_TIME=$(date +%s)
TOTAL_TASKS=$((H0_STEPS + 1))

for TASK_ID in $(seq 0 $H0_STEPS); do
  # Calculate h0 for this specific loop iteration
  H0_VAL=$(awk -v min=$H0_MIN -v max=$H0_MAX -v steps=$H0_STEPS -v task=$TASK_ID \
    'BEGIN { if (steps==0) print min; else printf "%.6f", min + task * (max - min) / steps }')

  OUTPUT_FILE="${MASTER_DIR}/results_h0_${H0_VAL}.txt"

  echo "=========================================================="
  echo "Sweep Task $((TASK_ID + 1)) / $TOTAL_TASKS"
  echo "H0 = $H0_VAL | HA = $HA_MIN"
  echo "=========================================================="

  # Run the executable (forcing CPU backend)
  "$EXEC" $SIZE $L $ALPHA $OMEGA $DT $TRANSIENT $EVAL $H0_VAL $HA_MIN $HA_MAX $HA_STEPS "$OUTPUT_FILE" cpu

  # Calculate Global ETA
  CURRENT_TIME=$(date +%s)
  ELAPSED=$((CURRENT_TIME - START_TIME))
  TASKS_DONE=$((TASK_ID + 1))

  AVG_TIME_PER_TASK=$((ELAPSED / TASKS_DONE))
  REMAINING_TASKS=$((TOTAL_TASKS - TASKS_DONE))
  ETA_SECONDS=$((AVG_TIME_PER_TASK * REMAINING_TASKS))

  # Format seconds into HH:MM:SS
  ETA_FORMATTED=$(date -u -d @"$ETA_SECONDS" +'%Hh %Mm %Ss')

  echo "----------------------------------------------------------"
  echo "Avg time per task: ${AVG_TIME_PER_TASK}s"
  echo "Estimated global time remaining: $ETA_FORMATTED"
  echo "----------------------------------------------------------"
done

echo "=========================================================="
echo "All sweep tasks completed successfully!"
echo "Outputs stored in: $MASTER_DIR"
echo "=========================================================="
