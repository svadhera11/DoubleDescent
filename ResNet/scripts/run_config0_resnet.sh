# #!/usr/bin/env bash
# set -euo pipefail

# # Which python env to use (optional)
# PYTHON=python   # or e.g. /mnt/nas/mayukh/.conda/envs/llmnar/bin/python

DATA_ROOT="./data"
NOISE=0.15
EPOCHS=4000
BATCH_SIZE=128
SEED=0
TOLERANCE=10    # 0 or None-like to disable early stopping
LOG_DIR="logs/config0"
OPTIMIZER="adam"
LOSS="ce"

# List of k values to sweep
K_LIST=(1 2 4 6 8 10 12 16 20 24 30 32 40 44 50 52 54 60 64)

mkdir -p "${LOG_DIR}"

echo "Launching runs for k in: ${K_LIST[*]}"

for K in "${K_LIST[@]}"; do
  LOG_FILE="${LOG_DIR}/k_${K}.log"

  echo "  -> starting k=${K}, log=${LOG_FILE}"

  # If you want to pin to a specific GPU, add CUDA_VISIBLE_DEVICES here.
  CUDA_VISIBLE_DEVICES=4 \
  python resnet18.py \
    --data-root "${DATA_ROOT}" \
    --k "${K}" \
    --noise "${NOISE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --tolerance "${TOLERANCE}" \
    --loss "${LOSS}" \
    --num_workers 0 \
    --optimizer "${OPTIMIZER}" &
done

echo "All runs launched, waiting..."
wait
echo "All runs finished."
