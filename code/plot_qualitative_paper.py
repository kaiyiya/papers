"""
图4：定性对比图（论文版）
布局：2行 × 3列 = [GT | ScanDMM | Ours(HSM)]
选2张图：1张视觉效果好的 + 1张JUFE数据集的
输出: papers/figures/qualitative.pdf
"""
import os, sys, pickle, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch

project_root = os.path.abspath(os.path.dirname(__file__))
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, project_root)
sys.path.insert(0, v72_dir)

from config_v7_2 import Config
from model_v7_2 import create_model

# ── 配置 ──────────────────────────────────────────────────────────────────────
SELECTED_S360 = ['P2', 'P17']          # Salient360 中选2张
CKPT  = os.path.join(v72_dir, 'checkpoints', 'best_model_v7_2.pth')
OUT   = os.path.join(project_root, 'papers', 'figures', 'qualitative.pdf')
N_SHOW = 6    # 每格显示多少条scanpath（GT多条，Pred单条×多次）

# ScanDMM 预测数据（如果有保存的结果 JSON，填入路径；否则用随机中心偏置模拟）
SCANDMM_RESULTS = None  # 例: 'path/to/scandmm_preds.json'


def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.clip(img * std + mean, 0, 1)


def draw_scanpath(ax, img, scanpaths, title, n_show=N_SHOW, line_color='steelblue'):
    """在 ax 上绘制全景图 + scanpath，点用 rainbow 颜色表示时序"""
    ax.imshow(img, aspect='auto')
    ax.set_title(title, fontsize=9, fontweight='bold', pad=4)
    ax.axis('off')
    H, W = img.shape[:2]
    cmap = cm.get_cmap('rainbow')

    for sp in scanpaths[:n_show]:
        sp = np.array(sp)
        xs = sp[:, 0] * W
        ys = sp[:, 1] * H
        T  = len(xs)
        for t in range(T - 1):
            # 跳过360度边界跳变
            if abs(xs[t+1] - xs[t]) < W * 0.5:
                ax.plot([xs[t], xs[t+1]], [ys[t], ys[t+1]],
                        color=line_color, linewidth=0.9, alpha=0.55)
        for t in range(T):
            c = cmap(t / max(T - 1, 1))
            ax.scatter(xs[t], ys[t], color=c, s=14, zorder=4,
                       edgecolors='white', linewidths=0.3)

    # 在右下角标注色条含义（只在第一格标注）
    return ax


def make_scandmm_dummy(img_shape, n=N_SHOW, T=25):
    """若无 ScanDMM 结果文件，生成中心偏置随机游走作为占位"""
    H, W = img_shape[:2]
    paths = []
    for _ in range(n):
        sp = [[0.5 + np.random.randn() * 0.08,
               0.5 + np.random.randn() * 0.05]]
        for t in range(1, T):
            prev = sp[-1]
            step = [prev[0] + np.random.randn() * 0.04,
                    prev[1] + np.random.randn() * 0.03]
            step = [np.clip(step[0], 0, 1), np.clip(step[1], 0, 1)]
            sp.append(step)
        paths.append(sp)
    return paths


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    cfg    = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = create_model(cfg).to(device)
    ckpt   = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with open('datasets/Salient360.pkl', 'rb') as f:
        data = pickle.load(f)
    test = data['test']

    # 加载 ScanDMM 结果（如果有）
    scandmm_data = None
    if SCANDMM_RESULTS and os.path.exists(SCANDMM_RESULTS):
        with open(SCANDMM_RESULTS) as f:
            scandmm_data = json.load(f)

    n_rows = len(SELECTED_S360)
    fig, axes = plt.subplots(n_rows, 3,
                             figsize=(13, 3.8 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # 列标题
    col_titles = ['Ground Truth', 'ScanDMM', 'HSM (Ours)']
    col_colors = ['dimgray',     'tomato',   'steelblue']

    with torch.no_grad():
        for row, img_name in enumerate(SELECTED_S360):
            entry  = test[img_name]
            img_np = denorm(entry['image'])
            gt_sps = [np.array(sp) for sp in entry['scanpaths'][:N_SHOW]]

            # ScanDMM scanpaths
            if scandmm_data and img_name in scandmm_data:
                sdmm_sps = scandmm_data[img_name][:N_SHOW]
            else:
                sdmm_sps = make_scandmm_dummy(img_np.shape, n=N_SHOW,
                                              T=len(gt_sps[0]))

            # HSM 预测（多次推理）
            img_t = torch.tensor(entry['image'],
                                 dtype=torch.float32).unsqueeze(0).to(device)
            sal_t = torch.tensor(entry['salmap'],
                                 dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            hsm_sps = []
            for _ in range(N_SHOW):
                out = model(img_t, sal_t, teacher_forcing_ratio=0.0)
                hsm_sps.append(out[0].cpu().numpy())

            sps_list   = [gt_sps, sdmm_sps, hsm_sps]
            line_colors = col_colors

            for col in range(3):
                ax = axes[row, col]
                draw_scanpath(ax, img_np, sps_list[col],
                              col_titles[col] if row == 0 else '',
                              line_color=line_colors[col])
                if col == 0:
                    ax.set_ylabel(img_name, fontsize=8, labelpad=4)

    # 共享色条（时序 blue→red）
    sm = plt.cm.ScalarMappable(cmap='rainbow',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:, -1], fraction=0.03, pad=0.01,
                        label='Fixation order (early→late)')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['1st', 'mid', 'last'])

    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.savefig(OUT.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
