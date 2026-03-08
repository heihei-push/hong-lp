#!/usr/bin/env bash
set -euo pipefail

# 对比三种解码器基线：dot / mlp / moe
# 用法：bash experiments/run_baseline_compare.sh

DATASETS=(texas cornell wisconsin chameleon squirrel cora citeseer pubmed)
SEEDS=(0 1 2)
LOG_ROOT="${1:-logs/baseline}"

mkdir -p "${LOG_ROOT}"

for DECODER in dot mlp moe; do
  echo "[Baseline] decoder=${DECODER}"
  python train_dual_channel_lp.py \
    --decoder "${DECODER}" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --log-dir "${LOG_ROOT}/${DECODER}"
done

echo "Baseline experiment complete. Logs are saved under: ${LOG_ROOT}"
