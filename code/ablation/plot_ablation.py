"""
Ablation study visualization for HSM paper.
Reads ablation_results_multi_run.json and produces a publication-quality figure.
Output: ablation_figure.pdf  (also saves .png for preview)
"""

import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── font / style ──────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'axes.linewidth': 0.8,
    'axes.unicode_minus': False,
    'pdf.fonttype': 42,   # embeds fonts for camera-ready PDF
    'ps.fonttype': 42,
})

# ── load data ─────────────────────────────────────────────────────────────────
with open('ablation_results_multi_run.json', 'r') as f:
    raw = json.load(f)

# ── variant metadata ──────────────────────────────────────────────────────────
VARIANTS = [
    ('full',            'HSM (Full)',          '#2563EB'),   # blue
    ('no_coord_att',    'w/o CoordAtt',        '#DC2626'),   # red
    ('no_hierarchical', 'w/o Hier. Sampling',  '#D97706'),   # amber
    ('no_coverage_loss','w/o Coverage Loss',   '#7C3AED'),   # purple
    ('lstm_baseline',   'LSTM Baseline',       '#059669'),   # green
]

DATASETS  = ['Salient360', 'AOI', 'JUFE', 'Sitzmann']
METRICS   = [
    ('DTW',  'DTW $\\downarrow$',  True),   # (key, label, lower_is_better)
    ('LEV',  'LEV $\\downarrow$',  True),
    ('REC',  'REC $\\uparrow$',    False),
]

# ── extract means and stds ────────────────────────────────────────────────────
def get(variant, dataset, metric):
    m = raw[variant][dataset][metric]
    return m['mean'], m['std']

# ── figure layout ─────────────────────────────────────────────────────────────
# 3 rows (one per metric) × 4 cols (one per dataset)
fig = plt.figure(figsize=(7.16, 5.5))   # 7.16 in = full ACM double-column width
gs  = GridSpec(3, 4, figure=fig,
               hspace=0.52, wspace=0.35,
               left=0.07, right=0.99, top=0.91, bottom=0.13)

n_variants = len(VARIANTS)
x          = np.arange(n_variants)
bar_w      = 0.62
tick_labels = [v[1] for v in VARIANTS]

# abbreviate x-tick labels for space
SHORT = {
    'HSM (Full)':         'Full',
    'w/o CoordAtt':       'w/o\nCoordAtt',
    'w/o Hier. Sampling': 'w/o\nHier.',
    'w/o Coverage Loss':  'w/o\nCov.',
    'LSTM Baseline':      'LSTM',
}
short_labels = [SHORT[v[1]] for v in VARIANTS]

for row, (metric_key, metric_label, lower_better) in enumerate(METRICS):
    for col, dataset in enumerate(DATASETS):

        ax = fig.add_subplot(gs[row, col])

        means = np.array([get(v[0], dataset, metric_key)[0] for v in VARIANTS])
        stds  = np.array([get(v[0], dataset, metric_key)[1] for v in VARIANTS])
        colors = [v[2] for v in VARIANTS]

        # highlight best bar
        best_idx = int(np.argmin(means) if lower_better else np.argmax(means))

        bars = ax.bar(x, means, bar_w,
                      color=colors, alpha=0.82,
                      error_kw=dict(elinewidth=0.8, capsize=2.5, ecolor='#333'))
        ax.errorbar(x, means, yerr=stds,
                    fmt='none', elinewidth=0.8, capsize=2.5, ecolor='#222', zorder=5)

        # gold star on best bar
        ax.text(x[best_idx], means[best_idx] + stds[best_idx],
                '★', ha='center', va='bottom', fontsize=7, color='#B45309')

        # thin grid
        ax.yaxis.grid(True, linewidth=0.4, linestyle='--', color='#ccc', zorder=0)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)

        # titles and labels
        if row == 0:
            ax.set_title(dataset, fontsize=8.5, fontweight='bold', pad=4)
        if col == 0:
            ax.set_ylabel(metric_label, fontsize=8, labelpad=3)

        ax.set_xticks(x)
        if row == 2:
            ax.set_xticklabels(short_labels, fontsize=7)
        else:
            ax.set_xticklabels([])

        # y-axis: tight range so bars look distinct
        margin = (means.max() - means.min()) * 0.25 + stds.max()
        lo = max(0, means.min() - margin * 0.6)
        hi = means.max() + margin * 1.2
        ax.set_ylim(lo, hi)

        # numeric labels on bars (compact)
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(i, lo + (hi - lo) * 0.01, f'{m:.1f}',
                    ha='center', va='bottom', fontsize=5.8,
                    color='white', fontweight='bold')

# ── legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color=v[2], alpha=0.82, label=v[1])
    for v in VARIANTS
]
star_handle = plt.Line2D([0], [0], marker='*', color='w',
                          markerfacecolor='#B45309', markersize=8, label='Best')
handles.append(star_handle)

fig.legend(handles=handles,
           loc='lower center',
           ncol=len(handles),
           fontsize=7,
           frameon=True,
           framealpha=0.9,
           edgecolor='#ccc',
           bbox_to_anchor=(0.5, 0.001))

# ── save ──────────────────────────────────────────────────────────────────────
fig.savefig('ablation_figure.pdf', dpi=300, bbox_inches='tight')
fig.savefig('ablation_figure.png', dpi=200, bbox_inches='tight')
print("Saved: ablation_figure.pdf / ablation_figure.png")
plt.show()
