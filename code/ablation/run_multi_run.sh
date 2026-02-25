#!/bin/bash
# 多次运行消融实验脚本 (3 runs per variant)
# 用法: bash ablation/run_multi_run.sh
# 单变体: bash ablation/run_multi_run.sh full

cd "$(dirname "$0")/.."

VARIANTS=("full" "no_coord_att" "no_hierarchical" "no_coverage_loss" "lstm_baseline")
N_RUNS=3

if [ $# -eq 1 ]; then
    VARIANTS=("$1")
fi

for variant in "${VARIANTS[@]}"; do
    for run in $(seq 0 $((N_RUNS-1))); do
        echo "========================================"
        echo "Training variant=$variant  run=$run"
        echo "========================================"
        conda run -n eye_scan_pytorch26 python ablation/train_ablation.py \
            --variant "$variant" --run "$run"
        echo "Done: $variant run$run"
        echo ""
    done
done

echo "========================================"
echo "All training done. Running evaluation..."
echo "========================================"
conda run -n eye_scan_pytorch26 python ablation/eval_ablation.py --runs 3

echo "========================================"
echo "Measuring efficiency..."
echo "========================================"
conda run -n eye_scan_pytorch26 python ablation/measure_efficiency.py
