"""
Teaser Figure（论文第一页大图）
布局：1行 × 3列 = [全景图原图 | Ground Truth路径 | HSM预测路径]
选1张视觉效果最好的 Salient360 图
输出: papers/figures/teaser.pdf
"""
import os, sys, pickle, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

project_root = os.path.abspath(os.path.dirname(__file__))
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, project_root)
sys.path.insert(0, v72_dir)

from config_v7_2 import Config
from model_v7_2 import create_model

# ── 配置 ──────────────────────────────────────────────────────────────────────
IMG_NAME = 'P2'    # 选视觉效果最好的图，可改成 P17/P82 等
CKPT = os.path.join(v72_dir, 'checkpoints', 'best_model_v7_2.pth')
OUT  = os.path.join(project_root, 'papers', 'figures', 'teaser.pdf')
N_GT   = 5   # 展示几条 GT scanpath
N_PRED = 1   # Teaser 展示1条代表性预测路径即可（更清晰）


def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.clip(img * std + mean, 0, 1)


def draw_path(ax, img, scanpaths, title, subtitle,
              line_color, n_show, marker_scale=1.0):
    ax.imshow(img, aspect='auto')
    ax.axis('off')
    H, W = img.shape[:2]
    cmap = cm.get_cmap('plasma')

    for idx, sp in enumerate(scanpaths[:n_show]):
        sp = np.array(sp)
        xs = sp[:, 0] * W
        ys = sp[:, 1] * H
        T  = len(xs)
        alpha = 0.9 if n_show == 1 else (0.8 - idx * 0.1)

        for t in range(T - 1):
            if abs(xs[t+1] - xs[t]) < W * 0.5:
                ax.plot([xs[t], xs[t+1]], [ys[t], ys[t+1]],
                        color=line_color, linewidth=1.2 * marker_scale,
                        alpha=alpha, solid_capstyle='round')

        for t in range(T):
            c = cmap(t / max(T - 1, 1))
            s = (22 if t == 0 or t == T-1 else 14) * marker_scale
            ec = 'white'
            ax.scatter(xs[t], ys[t], color=c, s=s, zorder=5,
                       edgecolors=ec, linewidths=0.4)
            # 标注起点终点
            if t == 0 and n_show == 1:
                ax.text(xs[t]+4, ys[t]-6, 'start', fontsize=6.5,
                        color='white',
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                     foreground='black')])
            if t == T-1 and n_show == 1:
                ax.text(xs[t]+4, ys[t]-6, 'end', fontsize=6.5,
                        color='white',
                        path_effects=[pe.withStroke(linewidth=1.5,
                                                     foreground='black')])

    ax.set_title(title, fontsize=11, fontweight='bold', pad=5,
                 color='white',
                 bbox=dict(boxstyle='round,pad=0.3', fc='#222222', alpha=0.75))
    ax.text(0.5, -0.03, subtitle, transform=ax.transAxes,
            ha='center', va='top', fontsize=8.5, color='#444444',
            style='italic')


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

    entry  = test[IMG_NAME]
    img_np = denorm(entry['image'])
    gt_sps = [np.array(sp) for sp in entry['scanpaths'][:N_GT]]

    # HSM 预测
    img_t = torch.tensor(entry['image'],
                         dtype=torch.float32).unsqueeze(0).to(device)
    sal_t = torch.tensor(entry['salmap'],
                         dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_t, sal_t, teacher_forcing_ratio=0.0)
    hsm_sp = [out[0].cpu().numpy()]

    # ── 绘图 ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2),
                             constrained_layout=True)
    fig.patch.set_facecolor('#f8f8f8')

    # 左：原图（无路径）
    axes[0].imshow(img_np, aspect='auto')
    axes[0].axis('off')
    axes[0].set_title('360° Panoramic Image', fontsize=11,
                      fontweight='bold', pad=5,
                      color='white',
                      bbox=dict(boxstyle='round,pad=0.3',
                                fc='#222222', alpha=0.75))
    axes[0].text(0.5, -0.03, 'Input equirectangular image',
                 transform=axes[0].transAxes,
                 ha='center', va='top', fontsize=8.5,
                 color='#444444', style='italic')

    # 中：Ground Truth（多条观察者路径）
    draw_path(axes[1], img_np, gt_sps,
              'Ground Truth Scanpaths',
              f'{N_GT} human observers — rainbow color: early→late fixation',
              line_color='#4a9eda', n_show=N_GT, marker_scale=0.9)

    # 右：HSM 预测（单条，清晰展示层次结构）
    draw_path(axes[2], img_np, hsm_sp,
              'HSM Predicted Scanpath',
              'Phase 1: broad exploration (blue) → Phase 2: saliency refinement (red)',
              line_color='#e05c5c', n_show=N_PRED, marker_scale=1.1)

    # 共享色条
    sm = plt.cm.ScalarMappable(cmap='plasma',
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        fraction=0.015, pad=0.01,
                        label='Fixation order  (early → late)')
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['1st', 'mid', 'last'])
    cbar.ax.yaxis.label.set_size(9)

    plt.savefig(OUT, dpi=220, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.savefig(OUT.replace('.pdf', '.png'), dpi=220,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
