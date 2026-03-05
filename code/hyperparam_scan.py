"""
超参数敏感性扫描脚本
扫描 alpha (phase transition ratio) 和 phase-wise exploration rates
在 Salient360 测试集上仅做推理，不重训练

运行方式（在项目根目录）:
    python hyperparam_scan.py

结果保存到: hyperparam_scan_results.json
            hyperparam_scan_table.txt   (可直接复制进论文)
"""

import os, sys, json, itertools
import torch
import numpy as np
from tqdm import tqdm

# ── 路径设置 ──────────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.dirname(__file__))
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, project_root)
sys.path.insert(0, v72_dir)
os.chdir(project_root)

from config_v7_2 import Config
from model_v7_2 import create_model

try:
    import editdistance
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean
    HAS_METRICS = True
except ImportError:
    sys.exit("请先安装依赖: pip install editdistance fastdtw scipy")

# ── 超参数搜索空间 ────────────────────────────────────────────────────────────
ALPHA_VALUES   = [0.3, 0.4, 0.5]          # phase transition ratio
EPS1_VALUES    = [0.50, 0.60, 0.70]       # Phase-1 exploration rate
EPS2_VALUES    = [0.20, 0.30, 0.40]       # Phase-2 exploration rate

# 只搜索 eps1 > eps2 的合法组合（Phase-1 探索率应高于 Phase-2）
CONFIGS = [
    (alpha, eps1, eps2)
    for alpha, eps1, eps2 in itertools.product(ALPHA_VALUES, EPS1_VALUES, EPS2_VALUES)
    if eps1 > eps2
]

CKPT = os.path.join(v72_dir, 'checkpoints', 'best_model_v7_2.pth')
N_REPEAT = 3   # 每个配置重复推理次数（取均值，减少随机性影响）

# ── 评估指标 ──────────────────────────────────────────────────────────────────
def lev(pred_px, gt_px):
    p = [f"{int(x)},{int(y)}" for x, y in (pred_px * 10).astype(int)]
    g = [f"{int(x)},{int(y)}" for x, y in (gt_px   * 10).astype(int)]
    return editdistance.eval(p, g)

def dtw(pred_px, gt_px):
    d, _ = fastdtw(pred_px, gt_px, dist=euclidean)
    return d

def rec(pred_px, thr=50.0):
    n = len(pred_px)
    if n < 2:
        return 0.0
    cnt = sum(
        np.linalg.norm(pred_px[i] - pred_px[j]) < thr
        for i in range(n) for j in range(i+1, n)
    )
    return cnt / (n*(n-1)/2) * 100

# ── 数据加载 ──────────────────────────────────────────────────────────────────
def load_test_data(cfg):
    """直接从 pkl 加载测试集，避免 DataLoader 的 worker 复杂性"""
    import pickle
    with open(cfg.processed_data_path, 'rb') as f:
        data = pickle.load(f)
    return data['test']   # dict: image_name -> {image, scanpaths, salmap, ...}

# ── 单次评估 ──────────────────────────────────────────────────────────────────
def run_eval(model, test_data, device, alpha, eps1, eps2):
    """
    对测试集每张图推理一条 scanpath，计算 LEV / DTW / REC 均值。
    通过 monkey-patch 覆盖采样器中的超参数，不修改权重。
    """
    # 动态替换采样器超参
    sampler = model.sampler
    sampler._scan_alpha = alpha
    sampler._scan_eps1  = eps1
    sampler._scan_eps2  = eps2

    # 为了让 monkey-patch 生效，需要替换 get_adaptive_exploration_rate
    def patched_rate(step, seq_len, visited_positions):
        if step < seq_len * sampler._scan_alpha:
            scheduled = sampler._scan_eps1
        else:
            scheduled = sampler._scan_eps2
        # 保留原有的 coverage boost 逻辑
        if len(visited_positions) > 3:
            positions = torch.stack(visited_positions[-5:], dim=0)
            y_std = positions[:, :, 1].std(dim=0).mean().item()
            if y_std < 0.1:
                boost = 0.15
            elif y_std < 0.15:
                boost = 0.10
            else:
                boost = 0.0
            return min(0.80, scheduled + boost)
        return scheduled

    import types
    sampler.get_adaptive_exploration_rate = types.MethodType(patched_rate, sampler)

    model.eval()
    all_lev, all_dtw, all_rec = [], [], []

    with torch.no_grad():
        for name, entry in test_data.items():
            img = torch.tensor(entry['image'],
                               dtype=torch.float32).unsqueeze(0).to(device)
            sal = torch.tensor(entry['salmap'],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            pred = model(img, sal, teacher_forcing_ratio=0.0)   # (1, T, 2)
            pred_np = pred[0].cpu().numpy()                      # (T, 2)

            H, W = img.shape[2], img.shape[3]
            pred_px = pred_np * np.array([W, H])

            # 与所有 GT scanpaths 比较，取均值
            gts = entry['scanpaths']       # list of (T,2) arrays 或 list of lists
            for gt in gts:
                gt_arr = np.array(gt)
                gt_px  = gt_arr * np.array([W, H])
                all_lev.append(lev(pred_px, gt_px))
                all_dtw.append(dtw(pred_px, gt_px))
                all_rec.append(rec(pred_px))

    return {
        'LEV': float(np.mean(all_lev)),
        'DTW': float(np.mean(all_dtw)),
        'REC': float(np.mean(all_rec)),
    }

# ── 主程序 ────────────────────────────────────────────────────────────────────
def main():
    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Checkpoint: {CKPT}")
    print(f"Total configs to scan: {len(CONFIGS)}  ×  {N_REPEAT} repeats\n")

    # 加载模型（只加载一次）
    model = create_model(cfg).to(device)
    ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Model loaded (epoch {ckpt.get('epoch','?')})\n")

    # 加载测试数据
    print("Loading test data...")
    test_data = load_test_data(cfg)
    print(f"Test images: {len(test_data)}\n")

    results = []
    for idx, (alpha, eps1, eps2) in enumerate(CONFIGS):
        label = f"α={alpha} ε₁={eps1} ε₂={eps2}"
        levs, dtws, recs = [], [], []

        for r in range(N_REPEAT):
            metrics = run_eval(model, test_data, device, alpha, eps1, eps2)
            levs.append(metrics['LEV'])
            dtws.append(metrics['DTW'])
            recs.append(metrics['REC'])

        entry = {
            'alpha': alpha, 'eps1': eps1, 'eps2': eps2,
            'LEV':  round(float(np.mean(levs)), 2),
            'DTW':  round(float(np.mean(dtws)), 2),
            'REC':  round(float(np.mean(recs)), 2),
            'DTW_std': round(float(np.std(dtws)),  2),
        }
        results.append(entry)

        star = ' ◀ default' if (alpha==0.4 and eps1==0.60 and eps2==0.30) else ''
        print(f"[{idx+1:2d}/{len(CONFIGS)}] {label:28s}  "
              f"LEV={entry['LEV']:6.2f}  DTW={entry['DTW']:8.2f}  "
              f"REC={entry['REC']:5.2f}{star}")

    # ── 保存 JSON ─────────────────────────────────────────────────────────────
    out_json = os.path.join(project_root, 'hyperparam_scan_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_json}")

    # ── 生成 LaTeX 小表 ───────────────────────────────────────────────────────
    out_tex = os.path.join(project_root, 'hyperparam_scan_table.txt')
    best_dtw = min(r['DTW'] for r in results)

    lines = []
    lines.append(r"% ── Hyperparameter sensitivity table (paste into main.tex) ──")
    lines.append(r"\begin{table}[t]")
    lines.append(r"  \centering")
    lines.append(r"  \caption{Sensitivity of HSM to the phase transition ratio $\alpha$ and")
    lines.append(r"    phase-wise exploration rates $\varepsilon_1$ / $\varepsilon_2$ on Salient360.")
    lines.append(r"    Results are averaged over 3 runs. \textbf{Bold}: selected configuration.}")
    lines.append(r"  \label{tab:hyperparam}")
    lines.append(r"  \begin{tabular}{ccc|ccc}")
    lines.append(r"    \toprule")
    lines.append(r"    $\alpha$ & $\varepsilon_1$ & $\varepsilon_2$ & LEV $\downarrow$ & DTW $\downarrow$ & REC $\uparrow$ \\")
    lines.append(r"    \midrule")

    for r in sorted(results, key=lambda x: (x['alpha'], x['eps1'], x['eps2'])):
        is_default = (r['alpha']==0.4 and r['eps1']==0.60 and r['eps2']==0.30)
        lev_s = f"{r['LEV']:.2f}"
        dtw_s = f"{r['DTW']:.2f}"
        rec_s = f"{r['REC']:.2f}"
        if is_default:
            row = (f"    \\textbf{{{r['alpha']}}} & \\textbf{{{r['eps1']}}} & "
                   f"\\textbf{{{r['eps2']}}} & "
                   f"\\textbf{{{lev_s}}} & \\textbf{{{dtw_s}}} & \\textbf{{{rec_s}}} \\\\")
        else:
            row = f"    {r['alpha']} & {r['eps1']} & {r['eps2']} & {lev_s} & {dtw_s} & {rec_s} \\\\"
        lines.append(row)

    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"Saved: {out_tex}")
    print("\nDone. Copy the table from hyperparam_scan_table.txt into main.tex.")

if __name__ == '__main__':
    main()
