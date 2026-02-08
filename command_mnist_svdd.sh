#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=./data
NUM_WORKERS=0
AE_BATCH=200
SVDD_BATCH=200

AE_EPOCHS=250
SVDD_EPOCHS=150

NU=0.1

SEEDS=(42 73 17)

echo "=============================="
echo " DeepSVDD MNIST EXPERIMENTS"
echo "=============================="

for digit in 0 1 2 3 4 5 6 7 8 9; do
  for seed in "${SEEDS[@]}"; do

    echo ""
    echo "-------------------------------------------"
    echo "[ONE-CLASS] digit=${digit} seed=${seed}"
    echo "-------------------------------------------"

    python svdd_main.py \
      --dataset mnist \
      --normal-digit ${digit} \
      --pollution-rate 0.0 \
      --data-dir ${DATA_DIR} \
      --num-workers ${NUM_WORKERS} \
      \
      --ae-arch autoencoder2 \
      --rep-dim 32 \
      --ae-epochs ${AE_EPOCHS} \
      --ae-lr 1e-3 \
      --ae-batch-size ${AE_BATCH} \
      \
      --svdd-objective one-class \
      --svdd-nu ${NU} \
      --svdd-epochs ${SVDD_EPOCHS} \
      --svdd-batch-size ${SVDD_BATCH} \
      --svdd-lr-encoder 1e-4 \
      --svdd-weight-decay 1e-6 \
      --svdd-warmup-epochs 10 \
      \
      --seed ${seed}


    echo ""
    echo "-------------------------------------------"
    echo "[SOFT-BOUNDARY] digit=${digit} seed=${seed}"
    echo "-------------------------------------------"

    python svdd_main.py \
      --dataset mnist \
      --normal-digit ${digit} \
      --pollution-rate 0.0 \
      --data-dir ${DATA_DIR} \
      --num-workers ${NUM_WORKERS} \
      \
      --ae-arch autoencoder2 \
      --rep-dim 32 \
      --ae-epochs ${AE_EPOCHS} \
      --ae-lr 1e-3 \
      --ae-batch-size ${AE_BATCH} \
      \
      --svdd-objective soft-boundary \
      --svdd-nu ${NU} \
      --svdd-epochs ${SVDD_EPOCHS} \
      --svdd-batch-size ${SVDD_BATCH} \
      --svdd-lr-encoder 1e-4 \
      --svdd-weight-decay 1e-6 \
      --svdd-warmup-epochs 10 \
      \
      --seed ${seed}

  done
done

echo ""
echo "ALL DEEPSVDD MNIST RUNS FINISHED"
