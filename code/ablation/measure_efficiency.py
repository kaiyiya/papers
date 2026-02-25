"""
效率测量脚本 - 参数量 + 推理时间
用法: python ablation/measure_efficiency.py
"""
import os
import sys
import time
import torch
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(project_root)

from ablation_config import AblationConfig
from ablation_models import create_model

VARIANTS = ['full', 'no_coord_att', 'no_hierarchical', 'no_coverage_loss', 'lstm_baseline']

WARMUP = 10
REPEAT = 100


def measure(variant, config, device):
    model = create_model(config, variant=variant).to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 构造虚拟输入
    image  = torch.randn(1, 3, 128, 256, device=device)
    salmap = torch.randn(1, 1, 128, 256, device=device).abs()
    salmap = salmap / salmap.sum()

    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(image, salmap, None, 0.0)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPEAT):
            _ = model(image, salmap, None, 0.0)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / REPEAT * 1000  # ms

    return n_params, elapsed


def main():
    config = AblationConfig()
    device = config.device
    print(f"Device: {device}\n")
    print(f"{'Variant':<22} {'Params':>12} {'Inference(ms)':>15}")
    print("-" * 52)

    results = {}
    for variant in VARIANTS:
        n_params, ms = measure(variant, config, device)
        results[variant] = {'params': n_params, 'inference_ms': round(ms, 3)}
        print(f"{variant:<22} {n_params:>12,} {ms:>14.2f}ms")

    import json
    out = os.path.join('ablation', 'efficiency_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
