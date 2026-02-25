"""
创建V7.2评估对比图表
"""
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据
epochs = ['Epoch 5', 'Epoch 10', 'Epoch 50', 'Best (E50)']
dtw = [1419.26, 1394.95, 1322.11, 1329.87]
rec = [49.20, 49.64, 48.00, 48.42]
x_coverage = [68.5, 67.1, 72.6, 72.4]
y_coverage = [27.0, 36.8, 28.0, 26.4]
xy_ratio = [2.87, 1.94, 2.88, 3.08]

# 创建图表
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('V7.2 Model Evaluation Comparison', fontsize=16, fontweight='bold')

# 1. DTW
ax = axes[0, 0]
bars = ax.bar(epochs, dtw, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('DTW Distance', fontsize=12)
ax.set_title('DTW (Lower is Better)', fontsize=12, fontweight='bold')
ax.axhline(y=1419.26, color='gray', linestyle='--', alpha=0.5, label='Epoch 5')
for i, v in enumerate(dtw):
    ax.text(i, v + 20, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 2. REC
ax = axes[0, 1]
bars = ax.bar(epochs, rec, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('REC (%)', fontsize=12)
ax.set_title('Recurrence Rate (Higher is Better)', fontsize=12, fontweight='bold')
ax.axhline(y=49.20, color='gray', linestyle='--', alpha=0.5, label='Epoch 5')
for i, v in enumerate(rec):
    ax.text(i, v + 0.3, f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 3. X Coverage
ax = axes[0, 2]
bars = ax.bar(epochs, x_coverage, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('X Coverage (%)', fontsize=12)
ax.set_title('X Coverage (Higher is Better)', fontsize=12, fontweight='bold')
ax.axhline(y=68.5, color='gray', linestyle='--', alpha=0.5, label='Epoch 5')
for i, v in enumerate(x_coverage):
    ax.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 4. Y Coverage
ax = axes[1, 0]
bars = ax.bar(epochs, y_coverage, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('Y Coverage (%)', fontsize=12)
ax.set_title('Y Coverage (Moderate is Best)', fontsize=12, fontweight='bold')
ax.axhline(y=27.0, color='gray', linestyle='--', alpha=0.5, label='Epoch 5 (Best)')
ax.axhline(y=30.0, color='red', linestyle='--', alpha=0.3, label='Upper Limit')
for i, v in enumerate(y_coverage):
    ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 5. X/Y Ratio
ax = axes[1, 1]
bars = ax.bar(epochs, xy_ratio, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('X/Y Ratio', fontsize=12)
ax.set_title('X/Y Balance (2.5-3.0 is Best)', fontsize=12, fontweight='bold')
ax.axhline(y=2.5, color='green', linestyle='--', alpha=0.3, label='Lower Bound')
ax.axhline(y=3.0, color='green', linestyle='--', alpha=0.3, label='Upper Bound')
for i, v in enumerate(xy_ratio):
    ax.text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 6. Overall Score (综合评分)
# 计算综合评分: DTW越低越好，REC越高越好，X覆盖越高越好，Y覆盖适中最好，X/Y比例2.5-3.0最好
dtw_score = [(1500 - d) / 200 for d in dtw]  # 归一化
rec_score = [r / 50 for r in rec]  # 归一化
x_score = [x / 75 for x in x_coverage]  # 归一化
y_score = [1 - abs(y - 27) / 27 for y in y_coverage]  # 27%为最佳
xy_score = [1 - abs(r - 2.75) / 2.75 for r in xy_ratio]  # 2.75为最佳

overall = [(d + r + x + y + xy) / 5 for d, r, x, y, xy in zip(dtw_score, rec_score, x_score, y_score, xy_score)]

ax = axes[1, 2]
bars = ax.bar(epochs, overall, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
ax.set_ylabel('Overall Score', fontsize=12)
ax.set_title('Overall Performance Score', fontsize=12, fontweight='bold')
for i, v in enumerate(overall):
    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('versions/v7.2_xy_balanced/v7_2_evaluation_comparison.png', dpi=150, bbox_inches='tight')
print("Chart saved: versions/v7.2_xy_balanced/v7_2_evaluation_comparison.png")
