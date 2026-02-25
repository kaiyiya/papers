"""
创建所有数据集的对比图表
"""
import matplotlib.pyplot as plt
import numpy as np
import json

# 加载数据
with open('versions/v7.2_xy_balanced/test_results_all_datasets/summary_all_datasets.json', 'r') as f:
    data = json.load(f)

# 提取数据
datasets = ['Salient360', 'AOI', 'JUFE', 'Sitzmann']
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

dtw = [data[d]['metrics']['DTW_mean'] for d in datasets]
rec = [data[d]['metrics']['REC_mean'] for d in datasets]
x_cov = [data[d]['metrics']['x_coverage_mean'] for d in datasets]
y_cov = [data[d]['metrics']['y_coverage_mean'] for d in datasets]
xy_ratio = [data[d]['metrics']['xy_ratio_mean'] for d in datasets]
num_samples = [data[d]['num_samples'] for d in datasets]

# 创建图表
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. DTW对比
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(datasets, dtw, color=colors)
ax1.set_ylabel('DTW Distance', fontsize=12, fontweight='bold')
ax1.set_title('DTW Comparison (Lower is Better)', fontsize=13, fontweight='bold')
ax1.axhline(y=np.mean(dtw), color='gray', linestyle='--', alpha=0.5, label='Mean')
for i, v in enumerate(dtw):
    ax1.text(i, v + 30, f'{v:.0f}', ha='center', va='bottom', fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# 2. REC对比
ax2 = fig.add_subplot(gs[0, 1])
bars = ax2.bar(datasets, rec, color=colors)
ax2.set_ylabel('REC (%)', fontsize=12, fontweight='bold')
ax2.set_title('Recurrence Rate (Higher is Better)', fontsize=13, fontweight='bold')
ax2.axhline(y=np.mean(rec), color='gray', linestyle='--', alpha=0.5, label='Mean')
for i, v in enumerate(rec):
    ax2.text(i, v + 0.2, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. X覆盖率对比
ax3 = fig.add_subplot(gs[0, 2])
bars = ax3.bar(datasets, x_cov, color=colors)
ax3.set_ylabel('X Coverage (%)', fontsize=12, fontweight='bold')
ax3.set_title('X Coverage (Higher is Better)', fontsize=13, fontweight='bold')
ax3.axhline(y=70, color='green', linestyle='--', alpha=0.3, label='Target: 70%')
for i, v in enumerate(x_cov):
    ax3.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# 4. Y覆盖率对比
ax4 = fig.add_subplot(gs[1, 0])
bars = ax4.bar(datasets, y_cov, color=colors)
ax4.set_ylabel('Y Coverage (%)', fontsize=12, fontweight='bold')
ax4.set_title('Y Coverage (Moderate is Best)', fontsize=13, fontweight='bold')
ax4.axhline(y=27, color='green', linestyle='--', alpha=0.3, label='Target: 27%')
ax4.axhline(y=30, color='orange', linestyle='--', alpha=0.3, label='Upper Limit: 30%')
for i, v in enumerate(y_cov):
    ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. X/Y比例对比
ax5 = fig.add_subplot(gs[1, 1])
bars = ax5.bar(datasets, xy_ratio, color=colors)
ax5.set_ylabel('X/Y Ratio', fontsize=12, fontweight='bold')
ax5.set_title('X/Y Balance (2.5-3.0 is Best)', fontsize=13, fontweight='bold')
ax5.axhline(y=2.5, color='green', linestyle='--', alpha=0.3, label='Lower Bound')
ax5.axhline(y=3.0, color='green', linestyle='--', alpha=0.3, label='Upper Bound')
for i, v in enumerate(xy_ratio):
    ax5.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. 样本数对比
ax6 = fig.add_subplot(gs[1, 2])
bars = ax6.bar(datasets, num_samples, color=colors)
ax6.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
ax6.set_title('Dataset Size', fontsize=13, fontweight='bold')
for i, v in enumerate(num_samples):
    ax6.text(i, v + 30, f'{v}', ha='center', va='bottom', fontweight='bold')
ax6.grid(axis='y', alpha=0.3)

# 7. 雷达图 - 综合性能
ax7 = fig.add_subplot(gs[2, :], projection='polar')

# 归一化指标 (0-1)
metrics_names = ['DTW\n(Lower)', 'REC\n(Higher)', 'X Coverage\n(Higher)', 
                 'Y Coverage\n(Moderate)', 'X/Y Ratio\n(2.5-3.0)']
angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
angles += angles[:1]

for i, dataset in enumerate(datasets):
    # 归一化各指标
    dtw_norm = 1 - (dtw[i] - min(dtw)) / (max(dtw) - min(dtw)) if max(dtw) != min(dtw) else 0.5
    rec_norm = (rec[i] - min(rec)) / (max(rec) - min(rec)) if max(rec) != min(rec) else 0.5
    x_norm = (x_cov[i] - min(x_cov)) / (max(x_cov) - min(x_cov)) if max(x_cov) != min(x_cov) else 0.5
    y_norm = 1 - abs(y_cov[i] - 27) / 10  # 27%为最佳
    xy_norm = 1 - abs(xy_ratio[i] - 2.75) / 1.0  # 2.75为最佳
    
    values = [dtw_norm, rec_norm, x_norm, y_norm, xy_norm]
    values += values[:1]
    
    ax7.plot(angles, values, 'o-', linewidth=2, label=dataset, color=colors[i])
    ax7.fill(angles, values, alpha=0.15, color=colors[i])

ax7.set_xticks(angles[:-1])
ax7.set_xticklabels(metrics_names, fontsize=11)
ax7.set_ylim(0, 1)
ax7.set_title('Overall Performance Comparison (Normalized)', 
              fontsize=14, fontweight='bold', pad=20)
ax7.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
ax7.grid(True)

plt.suptitle('V7.2 Model - All Datasets Comparison', 
             fontsize=16, fontweight='bold', y=0.98)

plt.savefig('versions/v7.2_xy_balanced/test_results_all_datasets/all_datasets_comparison.png', 
            dpi=150, bbox_inches='tight')
print("✓ Chart saved: versions/v7.2_xy_balanced/test_results_all_datasets/all_datasets_comparison.png")
