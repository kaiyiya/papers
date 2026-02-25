"""
消融实验评估脚本 - 在所有4个数据集上评估所有变体
用法:
  python ablation/eval_ablation.py                    # 评估所有变体
  python ablation/eval_ablation.py --variant full     # 评估单个变体
"""
import os
import sys
import json
import argparse
import torch
import numpy as np
import pickle
from tqdm import tqdm

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(project_root)

from ablation_config import AblationConfig
from ablation_models import create_model

import editdistance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

VARIANTS = ['full', 'no_coord_att', 'no_hierarchical', 'no_coverage_loss', 'lstm_baseline']

DATASETS = {
    'Salient360': ('datasets/Salient360.pkl', 'test', 25),
    'AOI':        ('datasets/AOI.pkl',        'test',  9),
    'JUFE':       ('datasets/JUFE.pkl',       None,   15),
    'Sitzmann':   ('datasets/Sitzmann.pkl',   'test', 25),
}


# ─── 指标计算 ─────────────────────────────────────────────────────────────────

def scanpath_to_string(sp, height_width, Xbins=12, Ybins=8):
    H, W = height_width
    hs, ws = H // Ybins, W // Xbins
    s = ''
    for pt in sp.astype(np.int32):
        xb = min(Xbins-1, max(0, pt[0] // ws))
        yb = min(Ybins-1, max(0, (H - pt[1]) // hs))
        s += chr(97 + yb) + chr(65 + xb)
    return s


def compute_metrics(pred_px, gt_px, hw):
    lev = editdistance.eval(
        scanpath_to_string(pred_px, hw),
        scanpath_to_string(gt_px,  hw)
    )
    dtw, _ = fastdtw(pred_px, gt_px, dist=euclidean)

    min_len = min(len(pred_px), len(gt_px))
    p, g = pred_px[:min_len], gt_px[:min_len]
    rec_upper = sum(
        1 for i in range(min_len) for j in range(i+1, min_len)
        if np.linalg.norm(p[i] - g[j]) < 12
    )
    total = min_len * (min_len - 1)
    rec = min(100.0, 100 * 2 * rec_upper / total) if total > 0 else 0.0

    return float(lev), float(dtw), float(rec)


# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_image_groups(dataset_path, split, gt_seq_len, model_seq_len=25):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    data_dict = data[split] if split else data

    groups = {}
    for img_key, img_data in data_dict.items():
        image = img_data.get('image')
        if image is None:
            continue
        if not isinstance(image, torch.Tensor):
            image = torch.FloatTensor(image)

        salmap = img_data.get('saliency_map') or img_data.get('salmap')
        if salmap is None:
            salmap = torch.ones(1, image.shape[-2], image.shape[-1])
        else:
            if not isinstance(salmap, torch.Tensor):
                salmap = torch.FloatTensor(salmap)
            if salmap.dim() == 2:
                salmap = salmap.unsqueeze(0)

        scanpaths_raw = img_data.get('scanpaths_2d') or img_data.get('scanpaths')
        if scanpaths_raw is None:
            continue

        gt_list = []
        for i in range(len(scanpaths_raw)):
            sp = np.array(scanpaths_raw[i], dtype=np.float32)
            if sp.ndim == 1:
                sp = sp.reshape(-1, 2)
            if sp.shape[-1] > 2:
                x, y, z = sp[:, 0], sp[:, 1], sp[:, 2]
                lon = np.arctan2(y, x)
                lat = np.arcsin(np.clip(z, -1, 1))
                u = (lon + np.pi) / (2 * np.pi)
                v = (lat + np.pi / 2) / np.pi
                sp = np.stack([u, v], axis=-1)
            sp = np.clip(sp, 0, 1)
            gt_list.append(sp[:gt_seq_len])  # 截断到GT原始长度

        # pad image scanpath tensor to model_seq_len for model input
        gt_tensors = []
        for sp in gt_list:
            if len(sp) >= model_seq_len:
                sp_pad = sp[:model_seq_len]
            else:
                sp_pad = np.vstack([sp, np.tile(sp[-1:], (model_seq_len - len(sp), 1))])
            gt_tensors.append(torch.FloatTensor(sp_pad))

        groups[img_key] = {
            'image': image,
            'salmap': salmap,
            'gt_list': gt_list,        # 原始长度，用于指标计算
            'gt_tensors': gt_tensors,  # pad到model_seq_len，用于teacher forcing
        }
    return groups


# ─── 评估单个变体 ─────────────────────────────────────────────────────────────

def evaluate_variant(variant, config, device, run=None):
    if run is not None:
        checkpoint_path = os.path.join('ablation', variant, f'run{run}', 'checkpoints', 'best_model.pth')
    else:
        checkpoint_path = os.path.join('ablation', variant, 'checkpoints', 'best_model.pth')
    if not os.path.exists(checkpoint_path):
        print(f"  [{variant}] No checkpoint found at {checkpoint_path}, skipping.")
        return None

    model = create_model(config, variant=variant).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  [{variant}] Loaded epoch {ckpt.get('epoch', '?')}, val_loss={ckpt.get('val_loss', '?'):.4f}")

    results = {}
    H, W = 128, 256

    for ds_name, (ds_path, split, gt_seq_len) in DATASETS.items():
        if not os.path.exists(ds_path):
            continue

        groups = load_image_groups(ds_path, split, gt_seq_len, config.seq_len)
        all_lev, all_dtw, all_rec = [], [], []

        with torch.no_grad():
            for img_key, gdata in tqdm(groups.items(), desc=f"  {ds_name}", leave=False):
                image  = gdata['image'].unsqueeze(0).to(device)
                salmap = gdata['salmap'].unsqueeze(0).to(device)

                for gt_sp in gdata['gt_list']:
                    pred = model(image, salmap, None, 0.0)  # (1, seq_len, 2)
                    pred_np = pred[0].cpu().numpy()[:len(gt_sp)]

                    pred_px = pred_np * np.array([W, H])
                    gt_px   = gt_sp  * np.array([W, H])

                    lev, dtw, rec = compute_metrics(pred_px, gt_px, (H, W))
                    all_lev.append(lev)
                    all_dtw.append(dtw)
                    all_rec.append(rec)

        results[ds_name] = {
            'LEV':  {'mean': float(np.mean(all_lev)), 'std': float(np.std(all_lev))},
            'DTW':  {'mean': float(np.mean(all_dtw)), 'std': float(np.std(all_dtw))},
            'REC':  {'mean': float(np.mean(all_rec)), 'std': float(np.std(all_rec))},
            'n':    len(all_lev),
        }
        print(f"    {ds_name}: LEV={results[ds_name]['LEV']['mean']:.2f}  "
              f"DTW={results[ds_name]['DTW']['mean']:.2f}  "
              f"REC={results[ds_name]['REC']['mean']:.2f}%  (n={len(all_lev)})")

    return results


# ─── 汇总表格 ─────────────────────────────────────────────────────────────────

def print_summary_table(all_results):
    print("\n" + "="*90)
    print("ABLATION STUDY RESULTS")
    print("="*90)
    for ds in ['Salient360', 'AOI', 'JUFE', 'Sitzmann']:
        print(f"\n── {ds} ──")
        print(f"{'Variant':<22} {'LEV↓':>10} {'DTW↓':>12} {'REC↑':>10}")
        print("-" * 58)
        for variant in VARIANTS:
            if variant not in all_results or ds not in all_results[variant]:
                print(f"{variant:<22} {'N/A':>10} {'N/A':>12} {'N/A':>10}")
                continue
            r = all_results[variant][ds]
            print(f"{variant:<22} "
                  f"{r['LEV']['mean']:>7.2f}±{r['LEV']['std']:>4.2f}  "
                  f"{r['DTW']['mean']:>8.1f}±{r['DTW']['std']:>6.1f}  "
                  f"{r['REC']['mean']:>6.2f}%±{r['REC']['std']:>4.2f}")
    print("="*90)


def aggregate_runs(variant, config, device, n_runs=3):
    """评估多次run，返回跨run的mean±std"""
    run_results = []
    for run in range(n_runs):
        res = evaluate_variant(variant, config, device, run=run)
        if res:
            run_results.append(res)
    if not run_results:
        return None

    # 聚合：对每个dataset每个指标取跨run的mean/std
    aggregated = {}
    for ds in run_results[0]:
        aggregated[ds] = {}
        for metric in ['LEV', 'DTW', 'REC']:
            means = [r[ds][metric]['mean'] for r in run_results]
            aggregated[ds][metric] = {
                'mean': float(np.mean(means)),
                'std':  float(np.std(means)),
                'run_means': means,
            }
        aggregated[ds]['n'] = run_results[0][ds]['n']
    return aggregated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default=None, choices=VARIANTS)
    parser.add_argument('--runs', type=int, default=3, help='number of runs to aggregate')
    parser.add_argument('--single', action='store_true', help='evaluate single checkpoint (no run dirs)')
    args = parser.parse_args()

    config = AblationConfig()
    device = config.device
    print(f"Device: {device}")

    variants_to_eval = [args.variant] if args.variant else VARIANTS
    all_results = {}

    for variant in variants_to_eval:
        print(f"\nEvaluating: {variant}")
        if args.single:
            res = evaluate_variant(variant, config, device, run=None)
        else:
            res = aggregate_runs(variant, config, device, n_runs=args.runs)
        if res:
            all_results[variant] = res

    # 保存结果
    out_path = os.path.join('ablation', 'ablation_results_multi_run.json') if not args.single else os.path.join('ablation', 'ablation_results.json')
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(all_results)
        all_results = existing
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    print_summary_table(all_results)


if __name__ == '__main__':
    main()
