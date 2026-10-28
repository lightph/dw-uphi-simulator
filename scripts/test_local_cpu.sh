#!/usr/bin/env bash
set -eo pipefail

# Jump to the project root directory
cd "$(dirname "$0")/.."

echo "========================================"
echo "Testing All CPU Backends Locally (Double Precision)"
echo "========================================"
BUILD_DIR="build_local"

# 1. Ensure all executables exist
EXECUTABLES=("main" "moments" "snapshots" "movies" "sweep" "sat_check")
for EXE in "${EXECUTABLES[@]}"; do
  if [ ! -f "${BUILD_DIR}/bin/${EXE}" ]; then
    echo "Error: Executable '${EXE}' not found. Please run ./compile_local.sh first."
    exit 1
  fi
done

# 2. Shared Parameters
SIZE=128
L=64
ALPHA=0.27
OMEGA=1.0
DT=0.0196349540849362
BATCH=2
H0=4.0
HA=3.0
EPS=0.05
SEED=42
BACKEND="cpu"

echo "Using Parameters: Size=$SIZE, L=$L, alpha=$ALPHA, Omega=$OMEGA, dt=$DT"
echo "----------------------------------------"

# 3. Test 'main' (Standard Time Evolution)
echo "[1/6] Testing 'main'..."
# Args: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <averaging_time> <window_time> <subsample_rate> <output_prefix> [eps] [seed] [backend]
./${BUILD_DIR}/bin/main $SIZE $BATCH $L $ALPHA $H0 $HA $OMEGA $DT 10.0 10.0 2.0 5 "test_main" $EPS $SEED $BACKEND

# 4. Test 'moments'
echo -e "\n[2/6] Testing 'moments'..."
# Args: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <total_time> <start_time> <num_checkpoints> <num_sq_checkpoints> <output_prefix> [eps] [seed] [backend]
./${BUILD_DIR}/bin/moments $SIZE $BATCH $L $ALPHA $H0 $HA $OMEGA $DT 15.0 5.0 5 5 "test_moments" $EPS $SEED $BACKEND

# 5. Test 'snapshots'
echo -e "\n[3/6] Testing 'snapshots'..."
# Args: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <transient_time> <snapshot_time> <window_time> <snapshot_rate> <output_prefix> [eps] [seed] [backend]
./${BUILD_DIR}/bin/snapshots $SIZE $BATCH $L $ALPHA $H0 $HA $OMEGA $DT 5.0 5.0 2.0 5 "test_snapshots" $EPS $SEED $BACKEND

# 6. Test 'movies'
echo -e "\n[4/6] Testing 'movies'..."
# Args: <size> <batch> <L> <alpha> <h0> <ha> <omega> <dt> <t_max> <window_time> <movie_start> <movie_end> <movie_rate> <output_prefix> [eps] [seed] [backend]
./${BUILD_DIR}/bin/movies $SIZE $BATCH $L $ALPHA $H0 $HA $OMEGA $DT 10.0 2.0 5.0 10.0 5 "test_movies" $EPS $SEED $BACKEND

# 7. Test 'sweep' (Tests a range of ha values)
echo -e "\n[5/6] Testing 'sweep'..."
# Args: <size> <L> <alpha> <omega> <dt> <transient_time> <eval_time> <h0_val> <ha_min> <ha_max> <ha_steps> <output_file> [backend]
./${BUILD_DIR}/bin/sweep $SIZE $L $ALPHA $OMEGA $DT 5.0 5.0 $H0 2.0 4.0 3 "test_sweep.txt" $BACKEND

# 8. Test 'sat_check'
echo -e "\n[6/6] Testing 'sat_check'..."
# Args: <size> <L> <alpha> <dt> <backend>
# Note: sat_check runs to t=2^20. We use 'timeout' to run it for 3 seconds just to verify execution, accepting exit code 124 (timeout).
set +e # Disable strict exit for the timeout
timeout 3s ./${BUILD_DIR}/bin/sat_check $SIZE $L $ALPHA $DT $BACKEND
TIMEOUT_STATUS=$?
set -e
if [ $TIMEOUT_STATUS -eq 124 ] || [ $TIMEOUT_STATUS -eq 0 ]; then
  echo "sat_check executed successfully (terminated by timeout as intended)."
else
  echo "sat_check failed with exit code $TIMEOUT_STATUS"
  exit 1
fi

echo ""
echo "========================================"
echo "Success! All CPU backend programs ran as intended."
echo "========================================"
