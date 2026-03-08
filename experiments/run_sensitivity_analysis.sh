#!/usr/bin/env bash
set -euo pipefail

# 参数敏感性实验（默认基于 MoE 解码器）
# 用法：bash experiments/run_sensitivity_analysis.sh

DATASETS=(texas cornell wisconsin chameleon squirrel cora citeseer pubmed)
SEEDS=(0 1 2)
LOG_ROOT="${1:-logs/sensitivity}"

mkdir -p "${LOG_ROOT}"

# 1) dropout
for DROPOUT in 0.2 0.5 0.8; do
  TAG="dropout_${DROPOUT//./p}"
  echo "[Sensitivity] ${TAG}"
  python train_dual_channel_lp.py \
    --decoder moe \
    --dropout "${DROPOUT}" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --log-dir "${LOG_ROOT}/${TAG}"
done

# 2) learning rate
for LR in 0.005 0.01 0.02; do
  TAG="lr_${LR//./p}"
  echo "[Sensitivity] ${TAG}"
  python train_dual_channel_lp.py \
    --decoder moe \
    --lr "${LR}" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --log-dir "${LOG_ROOT}/${TAG}"
done

# 3) output embedding dimension
for OUT_DIM in 32 64 128; do
  TAG="outdim_${OUT_DIM}"
  echo "[Sensitivity] ${TAG}"
  python train_dual_channel_lp.py \
    --decoder moe \
    --out-dim "${OUT_DIM}" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --log-dir "${LOG_ROOT}/${TAG}"
done

echo "Sensitivity experiments complete. Logs are saved under: ${LOG_ROOT}"
