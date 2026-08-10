#!/bin/bash
#$ -cwd
#$ -N time_evo_array
#$ -j y
#$ -S /bin/bash
#$ -q gpu
#$ -l mem=2G
#$ -V
#$ -l gpu=1
#$ -t 1-3

# Jump to the directory where the script was submitted from
cd $SGE_O_WORKDIR

echo "Starting Job: $JOB_NAME (ID: $JOB_ID, Task: $SGE_TASK_ID)"
echo "Running on Node: $HOSTNAME"
echo "--- GPU Hardware Info ---"
nvidia-smi -L
echo "-------------------------"

# Path to the executable (now relative to your original submission directory)
EXEC="../build/bin/time_evo"

# --- Shared System Parameters ---
SIZE=131072         # 2^17
L=131072.0
ALPHA=0.27
OMEGA=1.0
DT=0.25
MAX_POWER=30        

# --- Shared Hardware and Output Parameters ---
BACKEND="gpu"
EPS=0.05
SEED=42
PRECISION="double"

# --- Varying Parameters based on Array Task ID ---
case $SGE_TASK_ID in
    1)
        H0=1.1
        HA=0.0
        LABEL="DC"
        ;;
    2)
        H0=4.0
        HA=3.0
        LABEL="SC"
        ;;
    3)
        H0=5.0
        HA=4.0
        LABEL="NC"
        ;;
    *)
        echo "Error: Invalid task ID"
        exit 1
        ;;
esac

# Base output directory including the specific label (DC, SC, or NC)
BASE_OUT_DIR="../output/time_evo_${LABEL}_${L}_${H0}_${HA}_${OMEGA}"
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

echo "=========================================================="
echo "Running configuration: $LABEL (Array Task $SGE_TASK_ID)"
echo "Physics: H0=$H0 | HA=$HA | OMEGA=$OMEGA | L=$L | ALPHA=$ALPHA"
echo "Num    : dt=$DT | max_power=$MAX_POWER | size=$SIZE"
echo "Saving outputs to: $OUT_DIR"
echo "=========================================================="

# Execute
time $EXEC $SIZE $L $ALPHA $H0 $HA $OMEGA $DT $MAX_POWER $BACKEND $PREFIX $EPS $SEED $PRECISION

echo "Time evolution complete for $LABEL."