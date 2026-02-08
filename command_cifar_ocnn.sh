set -euo pipefail

# CIFAR-10 only — sequential execution (one run at a time)

DATA_DIR=./data
NUM_WORKERS=0
AE_BATCH=200
OCNN_BATCH=200
NU=0.1
EPOCHS_AE=250
EPOCHS_OCNN=250

SEEDS=(42 73 17 21 9)

echo "===== CIFAR-10 (sequential) ====="

# classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
for cls in 7 8 9; do
  for seed in "${SEEDS[@]}"; do
    echo "[CIFAR10] class=${cls} seed=${seed}"
    python main_ocnn.py \
      --dataset cifar10 \
      --normal-class ${cls} \
      --pollution-rate 0.01 \
      --data-dir ${DATA_DIR} \
      --num-workers ${NUM_WORKERS} \
      --ae-arch autoencoder_cifar \
      --rep-dim 32 \
      --ae-epochs ${EPOCHS_AE} \
      --ae-lr 1e-3 \
      --ae-batch-size ${AE_BATCH} \
      --nu ${NU} \
      --ocnn-epochs ${EPOCHS_OCNN} \
      --ocnn-hidden-dim 32 \
      --ocnn-batch-size ${OCNN_BATCH} \
      --activation relu \
      --seed ${seed}
  done
done

echo "ALL CIFAR-10 RUNS FINISHED"
