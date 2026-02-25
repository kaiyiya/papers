"""
定性对比图：GT scanpaths vs Ours (ScanDMM风格)
选4张图（2好2差），每张2列：GT | Ours
输出: versions/v7.2_xy_balanced/qualitative_comparison.pdf
"""
import os, sys, pickle, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

project_root = os.path.abspath(os.path.dirname(__file__))
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, project_root)
sys.path.insert(0, v72_dir)

from config_v7_2 import Config
from model_v7_2 import create_model

SELECTED = ['P2', 'P82', 'P17', 'P41']   # 2 best + 2 worst DTW
CKPT = os.path.join(v72_dir, 'checkpoints', 'best_model_v7_2.pth')
OUT  = os.path.join(v72_dir, 'qualitative_comparison.pdf')
N_SHOW = 8   # 每格显示几条scanpath


def denorm(img):
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.clip(img * std + mean, 0, 1)


def draw_scanpath(ax, img, scanpaths, title, n_show=N_SHOW):
    ax.imshow(img)
    ax.set_title(title, fontsize=10, pad=3)
    ax.axis('off')
    H, W = img.shape[:2]
    cmap = cm.get_cmap('rainbow')
    for sp in scanpaths[:n_show]:
        sp = np.array(sp)
        xs = sp[:, 0] * W
        ys = sp[:, 1] * H
        T = len(xs)
        colors = [cmap(t / max(T-1, 1)) for t in range(T)]
        for t in range(T - 1):
            # 处理360度边界跳变
            if abs(xs[t+1] - xs[t]) < W * 0.5:
                ax.plot([xs[t], xs[t+1]], [ys[t], ys[t+1]],
                        color='blue', linewidth=0.8, alpha=0.6)
        for t in range(T):
            ax.scatter(xs[t], ys[t], color=colors[t], s=12, zorder=3)


def main():
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(cfg).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with open('datasets/Salient360.pkl', 'rb') as f:
        data = pickle.load(f)
    test = data['test']

    fig, axes = plt.subplots(len(SELECTED), 2,
                             figsize=(14, 4.5 * len(SELECTED)),
                             constrained_layout=True)

    with torch.no_grad():
        for row, img_name in enumerate(SELECTED):
            entry = test[img_name]
            img_np = denorm(entry['image'])
            gt_sps = [np.array(sp) for sp in entry['scanpaths'][:N_SHOW]]

            # 推理
            img_t = torch.tensor(entry['image'], dtype=torch.float32).unsqueeze(0).to(device)
            sal_t = torch.tensor(entry['salmap'], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred_sps = []
            for _ in range(N_SHOW):
                out = model(img_t, sal_t, teacher_forcing_ratio=0.0)
                sp = out[0].cpu().numpy()  # (25, 2)
                pred_sps.append(sp)

            draw_scanpath(axes[row, 0], img_np, gt_sps,
                          f'{img_name} — Ground Truth')
            draw_scanpath(axes[row, 1], img_np, pred_sps,
                          f'{img_name} — Ours')

    plt.savefig(OUT, dpi=200, bbox_inches='tight')
    plt.savefig(OUT.replace('.pdf', '.png'), dpi=200, bbox_inches='tight')
    print(f'Saved: {OUT}')


if __name__ == '__main__':
    main()
