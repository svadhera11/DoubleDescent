python run_dd.py \
  --mode model-wise \
  --dataset cifar10 \
  --train-sizes 12500 \
  --widths 8 16 32 64 128 \
  --noise-probs 0.2 \
  --seeds 0 1 2 \
  --num-steps 200000 \
  --eval-every 10000 \
  --out-root results
