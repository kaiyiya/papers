#!/bin/bash
# 消融实验一键训练脚本
# 用法: bash ablation/run_all_ablations.sh
# 或单独运行某个变体: bash ablation/run_all_ablations.sh full

cd "$(dirname "$0")/.."

VARIANTS=("full" "no_coord_att" "no_hierarchical" "no_coverage_loss" "lstm_baseline")

if [ $# -eq 1 ]; then
    VARIANTS=("$1")
fi

for variant in "${VARIANTS[@]}"; do
    echo "========================================"
    echo "Training variant: $variant"
    echo "========================================"
    conda run -n eye_scan_pytorch26 python ablation/train_ablation.py --variant "$variant"
    echo "Done: $variant"
    echo ""
done

echo "========================================"
echo "All training done. Running evaluation..."
echo "========================================"
conda run -n eye_scan_pytorch26 python ablation/eval_ablation.py
