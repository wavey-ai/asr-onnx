#!/usr/bin/env bash
set -euo pipefail

GPUS=(0 1 2 3) # GPU IDs to configure
GI_PROFILE=19  # 1 g.5 GB on A100-40 GB
TARGET=7       # target instances per GPU
SLEEP_SECS=2   # wait between re-checks

enable_mig() {
  local g=$1
  # loop until MIG shows Enabled or we time out (±20 s)
  for _ in {1..10}; do
    if nvidia-smi -i "$g" --query-gpu=mig.mode.current --format=csv,noheader |
      grep -q Enabled; then
      return 0
    fi
    # try to flip MIG on; ignore harmless errors
    sudo nvidia-smi -i "$g" -mig 1 2>/dev/null || true
    sleep "$SLEEP_SECS"
  done
  return 1 # still not enabled
}

create_instances() {
  local g=$1 created=0
  # wipe old GI/CI
  sudo nvidia-smi mig -i "$g" -dci 2>/dev/null || true
  sudo nvidia-smi mig -i "$g" -dgi 2>/dev/null || true

  for ((i = 1; i <= TARGET; i++)); do
    # capture stderr because nvidia-smi exits 0 even on resource errors
    if ! err=$(sudo nvidia-smi mig -i "$g" --create-gpu-instance="$GI_PROFILE" -C 2>&1); then
      printf "    unexpected failure creating GI %d on GPU %d: %s\\n" "$i" "$g" "$err"
      break
    fi
    if echo "$err" | grep -q "Insufficient Resources"; then
      break
    fi
    created=$i
  done
  echo "$created"
}

echo "=== MIG partition: ${TARGET}×profile ${GI_PROFILE} per GPU ==="
for gpu in "${GPUS[@]}"; do
  echo "GPU $gpu:"
  if enable_mig "$gpu"; then
    made=$(create_instances "$gpu")
    echo "  → ${made}/${TARGET} GI(s) present."
  else
    echo "  → MIG mode still pending (GPU busy). Skipped."
  fi
done
echo "============================================================="
