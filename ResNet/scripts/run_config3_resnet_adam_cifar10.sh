# #!/usr/bin/env bash
# set -euo pipefail

# # Which python env to use (optional)
# PYTHON=python   # or e.g. /mnt/nas/mayukh/.conda/envs/llmnar/bin/python

DATA_ROOT="./data"
NOISE=0.0
EPOCHS=4000
BATCH_SIZE=256
SEED=0
TOLERANCE=100    # 0 or None-like to disable early stopping
LOG_DIR="logs/config3"
OPTIMIZER="adam"
LOSS="ce"
DATASET="cifar10"
LR=0.001

# List of k values to sweep
K_LIST=(2 4 8 16 32 64)

mkdir -p "${LOG_DIR}"

echo "Launching runs for k in: ${K_LIST[*]}"

for K in "${K_LIST[@]}"; do
  LOG_FILE="${LOG_DIR}/k_${K}.log"

  echo "  -> starting k=${K}, log=${LOG_FILE}"

  # If you want to pin to a specific GPU, add CUDA_VISIBLE_DEVICES here.
  python run_resnet18.py \
    --data-root "${DATA_ROOT}" \
    --log-dir "${LOG_DIR}" \
    --k "${K}" \
    --noise "${NOISE}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --tolerance "${TOLERANCE}" \
    --loss "${LOSS}" \
    --num-workers 0 \
    --lr "${LR}" \
    --dataset "${DATASET}" \
    --optimizer "${OPTIMIZER}" &
done

echo "All runs launched, waiting..."
wait
echo "All runs finished."
