"""
V7.2评估脚本 - X-Y平衡优化模型
"""
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from datetime import datetime
import argparse

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 添加V7.2目录到路径
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, v72_dir)

from config_v7_2 import Config
from model_v7_2 import create_model
from simple_dataset import create_dataloaders

# 导入评估指标
try:
    import editdistance
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("⚠️ 缺少评估依赖，请安装: pip install editdistance fastdtw scipy")


def compute_lev_distance(pred, gt):
    """计算Levenshtein距离"""
    pred_discrete = (pred * 10).astype(int)
    gt_discrete = (gt * 10).astype(int)
    pred_str = [f"{x},{y}" for x, y in pred_discrete]
    gt_str = [f"{x},{y}" for x, y in gt_discrete]
    return editdistance.eval(pred_str, gt_str)


def compute_dtw_distance(pred, gt):
    """计算DTW距离"""
    distance, _ = fastdtw(pred, gt, dist=euclidean)
    return distance


def compute_recurrence_rate(pred, threshold=50.0):
    """计算回访率"""
    n = len(pred)
    if n < 2:
        return 0.0
    recurrence_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(pred[i] - pred[j])
            if dist < threshold:
                recurrence_count += 1
    total_pairs = n * (n - 1) / 2
    return (recurrence_count / total_pairs) * 100 if total_pairs > 0 else 0.0


def compute_coverage(scanpaths):
    """计算覆盖率"""
    x_coords = scanpaths[:, 0]
    y_coords = scanpaths[:, 1]
    x_range = x_coords.max() - x_coords.min()
    y_range = y_coords.max() - y_coords.min()
    x_std = x_coords.std()
    y_std = y_coords.std()
    return {
        'x_coverage': x_range * 100,
        'y_coverage': y_range * 100,
        'x_std': x_std,
        'y_std': y_std,
        'xy_ratio': (x_range / y_range) if y_range > 0 else 0
    }


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_lev = []
    all_dtw = []
    all_rec = []
    all_coverage = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            gt_scanpaths = batch['scanpath'].to(device)
            saliency_maps = batch.get('saliency_map')
            if saliency_maps is None:
                continue
            saliency_maps = saliency_maps.to(device)
            pred_scanpaths = model(images, saliency_maps, None, 0.0)
            pred_np = pred_scanpaths.cpu().numpy()
            gt_np = gt_scanpaths.cpu().numpy()
            B, _, H, W = images.shape
            pred_pixel = pred_np.copy()
            gt_pixel = gt_np.copy()
            pred_pixel[:, :, 0] *= W
            pred_pixel[:, :, 1] *= H
            gt_pixel[:, :, 0] *= W
            gt_pixel[:, :, 1] *= H
            for i in range(len(pred_np)):
                if HAS_METRICS:
                    lev = compute_lev_distance(pred_pixel[i], gt_pixel[i])
                    dtw = compute_dtw_distance(pred_pixel[i], gt_pixel[i])
                    rec = compute_recurrence_rate(pred_pixel[i])
                    all_lev.append(lev)
                    all_dtw.append(dtw)
                    all_rec.append(rec)
                coverage = compute_coverage(pred_np[i])
                all_coverage.append(coverage)

    results = {
        'num_samples': len(all_lev),
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    if HAS_METRICS:
        results.update({
            'LEV': float(np.mean(all_lev)),
            'LEV_std': float(np.std(all_lev)),
            'DTW': float(np.mean(all_dtw)),
            'DTW_std': float(np.std(all_dtw)),
            'REC': float(np.mean(all_rec)),
            'REC_std': float(np.std(all_rec))
        })
    coverage_stats = {
        'x_coverage_mean': float(np.mean([c['x_coverage'] for c in all_coverage])),
        'x_coverage_std': float(np.std([c['x_coverage'] for c in all_coverage])),
        'y_coverage_mean': float(np.mean([c['y_coverage'] for c in all_coverage])),
        'y_coverage_std': float(np.std([c['y_coverage'] for c in all_coverage])),
        'xy_ratio_mean': float(np.mean([c['xy_ratio'] for c in all_coverage])),
        'xy_ratio_std': float(np.std([c['xy_ratio'] for c in all_coverage])),
        'x_std_mean': float(np.mean([c['x_std'] for c in all_coverage])),
        'y_std_mean': float(np.mean([c['y_std'] for c in all_coverage]))
    }
    results.update(coverage_stats)
    return results


def main():
    parser = argparse.ArgumentParser(description='V7.2模型评估')
    parser.add_argument('--checkpoint', type=str, default='best_model_v7_2.pth')
    parser.add_argument('--epoch', type=int, default=None)
    args = parser.parse_args()

    config = Config()
    v72_dir = os.path.join('versions', 'v7.2_xy_balanced')
    if args.epoch is not None:
        checkpoint_name = f'checkpoint_epoch_{args.epoch}.pth'
    else:
        checkpoint_name = args.checkpoint
    checkpoint_path = os.path.join(v72_dir, 'checkpoints', checkpoint_name)
    log_dir = os.path.join(v72_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return

    device = config.device
    print(f"使用设备: {device}")
    print("\n加载数据...")
    _, test_loader = create_dataloaders(config)
    print(f"测试集: {len(test_loader.dataset)} 样本")
    print("\n创建V7.2模型...")
    model = create_model(config).to(device)
    print(f"\n加载检查点: {checkpoint_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_num = checkpoint.get('epoch', 'unknown')
    print(f"  Epoch: {epoch_num}")
    if 'val_loss' in checkpoint:
        print(f"  Val Loss: {checkpoint['val_loss']:.4f}")
    if 'coverage_stats' in checkpoint:
        stats = checkpoint['coverage_stats']
        print(f"  X覆盖: {stats['x_coverage']:.1f}%")
        print(f"  Y覆盖: {stats['y_coverage']:.1f}%")
        print(f"  X/Y比例: {stats['xy_ratio']:.2f}")

    print("\n开始评估...")
    results = evaluate_model(model, test_loader, device)
    print("\n" + "=" * 70)
    print(f"V7.2评估结果 - {checkpoint_name}")
    print("=" * 70)
    if HAS_METRICS:
        print(f"\n核心指标:")
        print(f"  LEV: {results['LEV']:.2f} ± {results['LEV_std']:.2f}")
        print(f"  DTW: {results['DTW']:.2f} ± {results['DTW_std']:.2f}")
        print(f"  REC: {results['REC']:.2f}% ± {results['REC_std']:.2f}%")
    print(f"\n覆盖率:")
    print(f"  X覆盖: {results['x_coverage_mean']:.1f}% ± {results['x_coverage_std']:.1f}%")
    print(f"  Y覆盖: {results['y_coverage_mean']:.1f}% ± {results['y_coverage_std']:.1f}%")
    print(f"  X/Y比例: {results['xy_ratio_mean']:.2f} ± {results['xy_ratio_std']:.2f}")
    print(f"  X标准差: {results['x_std_mean']:.3f}")
    print(f"  Y标准差: {results['y_std_mean']:.3f}")
    print(f"\n样本数: {results['num_samples']}")
    print("=" * 70)

    timestamp = results['timestamp']
    epoch_suffix = f"_epoch_{args.epoch}" if args.epoch is not None else ""
    output_file = os.path.join(log_dir, f'evaluation_results_v7_2{epoch_suffix}_{timestamp}.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ 结果已保存: {output_file}")


if __name__ == '__main__':
    main()
