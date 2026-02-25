"""
创建修正前后的对比图表
"""
import matplotlib.pyplot as plt
import numpy as np

# 修正前后的数据对比
datasets = ['Salient360', 'AOI', 'JUFE', 'Sitzmann']

# 修正前的数据（错误的）
before_lev = [36.45, 36.45, 35.92, 36.54]
before_dtw = [1336.26, 1232.88, 1636.25, 1414.52]
before_rec = [5.65, 6.34, 4.88, 4.59]

# 修正后的数据（正确的）
after_lev = [36.07, 13.06, 21.61, 38.12]
after_dtw = [1343.78, 468.16, 1059.64, 1524.94]
after_rec = [5.58, 6.52, 5.21, 4.19]

# SOTA最佳数据
sota_lev = [37.27, 12.13, 23.09, 44.97]
sota_dtw = [1528.59, 537.50, 1086.01, 1965.43]
sota_rec = [3.58, 4.02, 4.33, 3.48]

# 人类数据
human_lev = [35.08, 9.24, 18.31, 41.19]
human_dtw = [1382.59, 389.48, 1038.88, 1836.99]
human_rec = [5.20, 6.23, 7.75, 6.35]

# 序列长度
seq_lengths = [25, 9, 15, 30]

# 创建图表
fig = plt.figure(figsize=(20, 12))

# 1. LEV对比
ax1 = plt.subplot(2, 3, 1)
x = np.arange(len(datasets))
width = 0.2

bars1 = ax1.bar(x - 1.5*width, before_lev, width, label='修正前', color='#ff6b6b', alpha=0.8)
bars2 = ax1.bar(x - 0.5*width, after_lev, width, label='修正后', color='#4ecdc4', alpha=0.8)
bars3 = ax1.bar(x + 0.5*width, sota_lev, width, label='SOTA', color='#95e1d3', alpha=0.8)
bars4 = ax1.bar(x + 1.5*width, human_lev, width, label='Human', color='#f38181', alpha=0.8)

ax1.set_ylabel('LEV (Lower is Better)', fontsize=12, fontweight='bold')
ax1.set_title('LEV Comparison (Before vs After Correction)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, rotation=45, ha='right')
ax1.legend(fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 标注改进
for i, (b, a) in enumerate(zip(before_lev, after_lev)):
    if abs(b - a) > 5:
        improvement = ((b - a) / b) * 100
        ax1.annotate(f'{improvement:.0f}%↓',
                    xy=(i, max(b, a) + 2),
                    ha='center', fontsize=9, color='green', fontweight='bold')

# 2. DTW对比
ax2 = plt.subplot(2, 3, 2)
bars1 = ax2.bar(x - 1.5*width, before_dtw, width, label='修正前', color='#ff6b6b', alpha=0.8)
bars2 = ax2.bar(x - 0.5*width, after_dtw, width, label='修正后', color='#4ecdc4', alpha=0.8)
bars3 = ax2.bar(x + 0.5*width, sota_dtw, width, label='SOTA', color='#95e1d3', alpha=0.8)
bars4 = ax2.bar(x + 1.5*width, human_dtw, width, label='Human', color='#f38181', alpha=0.8)

ax2.set_ylabel('DTW (Lower is Better)', fontsize=12, fontweight='bold')
ax2.set_title('DTW Comparison (Before vs After Correction)', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, rotation=45, ha='right')
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# 标注改进
for i, (b, a) in enumerate(zip(before_dtw, after_dtw)):
    if abs(b - a) > 200:
        improvement = ((b - a) / b) * 100
        ax2.annotate(f'{improvement:.0f}%↓',
                    xy=(i, max(b, a) + 100),
                    ha='center', fontsize=9, color='green', fontweight='bold')

# 3. REC对比
ax3 = plt.subplot(2, 3, 3)
bars1 = ax3.bar(x - 1.5*width, before_rec, width, label='修正前', color='#ff6b6b', alpha=0.8)
bars2 = ax3.bar(x - 0.5*width, after_rec, width, label='修正后', color='#4ecdc4', alpha=0.8)
bars3 = ax3.bar(x + 0.5*width, sota_rec, width, label='SOTA', color='#95e1d3', alpha=0.8)
bars4 = ax3.bar(x + 1.5*width, human_rec, width, label='Human', color='#f38181', alpha=0.8)

ax3.set_ylabel('REC % (Higher is Better)', fontsize=12, fontweight='bold')
ax3.set_title('REC Comparison (Before vs After Correction)', fontsize=14, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(datasets, rotation=45, ha='right')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# 4. 序列长度影响
ax4 = plt.subplot(2, 3, 4)
colors_seq = ['#4ecdc4', '#ff6b6b', '#95e1d3', '#f38181']
bars = ax4.bar(datasets, seq_lengths, color=colors_seq, alpha=0.8)
ax4.set_ylabel('Sequence Length (points)', fontsize=12, fontweight='bold')
ax4.set_title('GT Sequence Length by Dataset', fontsize=14, fontweight='bold')
ax4.set_xticklabels(datasets, rotation=45, ha='right')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# 标注长度
for bar, length in zip(bars, seq_lengths):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{length}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# 5. 改进百分比
ax5 = plt.subplot(2, 3, 5)
lev_improvement = [((b - a) / b) * 100 if b > a else 0 for b, a in zip(before_lev, after_lev)]
dtw_improvement = [((b - a) / b) * 100 if b > a else 0 for b, a in zip(before_dtw, after_dtw)]

x_imp = np.arange(len(datasets))
width_imp = 0.35

bars1 = ax5.bar(x_imp - width_imp/2, lev_improvement, width_imp, label='LEV Improvement', color='#4ecdc4', alpha=0.8)
bars2 = ax5.bar(x_imp + width_imp/2, dtw_improvement, width_imp, label='DTW Improvement', color='#95e1d3', alpha=0.8)

ax5.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
ax5.set_title('Improvement After Sequence Length Correction', fontsize=14, fontweight='bold')
ax5.set_xticks(x_imp)
ax5.set_xticklabels(datasets, rotation=45, ha='right')
ax5.legend(fontsize=10)
ax5.grid(axis='y', alpha=0.3, linestyle='--')
ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# 标注改进值
for i, (lev_imp, dtw_imp) in enumerate(zip(lev_improvement, dtw_improvement)):
    if lev_imp > 5:
        ax5.text(i - width_imp/2, lev_imp + 2, f'{lev_imp:.0f}%',
                ha='center', fontsize=9, fontweight='bold')
    if dtw_imp > 5:
        ax5.text(i + width_imp/2, dtw_imp + 2, f'{dtw_imp:.0f}%',
                ha='center', fontsize=9, fontweight='bold')

# 6. 与SOTA对比（修正后）
ax6 = plt.subplot(2, 3, 6)
categories = ['LEV', 'DTW', 'REC']

# 归一化到0-100（越接近100越好）
def normalize_metrics(v7_vals, sota_vals, human_vals, metric_type):
    """归一化指标，使得100表示最好"""
    all_vals = v7_vals + sota_vals + human_vals
    if metric_type == 'lower':  # LEV, DTW - 越低越好
        max_val = max(all_vals)
        return [(1 - v/max_val) * 100 for v in v7_vals]
    else:  # REC - 越高越好
        max_val = max(all_vals)
        return [(v/max_val) * 100 for v in v7_vals]

# 计算每个数据集的平均表现
v7_scores = []
sota_scores = []
human_scores = []

for i in range(len(datasets)):
    # V7.2
    lev_norm = (1 - after_lev[i] / max(after_lev[i], sota_lev[i], human_lev[i])) * 100
    dtw_norm = (1 - after_dtw[i] / max(after_dtw[i], sota_dtw[i], human_dtw[i])) * 100
    rec_norm = (after_rec[i] / max(after_rec[i], sota_rec[i], human_rec[i])) * 100
    v7_scores.append(np.mean([lev_norm, dtw_norm, rec_norm]))

    # SOTA
    lev_norm = (1 - sota_lev[i] / max(after_lev[i], sota_lev[i], human_lev[i])) * 100
    dtw_norm = (1 - sota_dtw[i] / max(after_dtw[i], sota_dtw[i], human_dtw[i])) * 100
    rec_norm = (sota_rec[i] / max(after_rec[i], sota_rec[i], human_rec[i])) * 100
    sota_scores.append(np.mean([lev_norm, dtw_norm, rec_norm]))

    # Human
    lev_norm = (1 - human_lev[i] / max(after_lev[i], sota_lev[i], human_lev[i])) * 100
    dtw_norm = (1 - human_dtw[i] / max(after_dtw[i], sota_dtw[i], human_dtw[i])) * 100
    rec_norm = (human_rec[i] / max(after_rec[i], sota_rec[i], human_rec[i])) * 100
    human_scores.append(np.mean([lev_norm, dtw_norm, rec_norm]))

x_comp = np.arange(len(datasets))
width_comp = 0.25

bars1 = ax6.bar(x_comp - width_comp, v7_scores, width_comp, label='V7.2 (Ours)', color='#4ecdc4', alpha=0.8)
bars2 = ax6.bar(x_comp, sota_scores, width_comp, label='SOTA', color='#95e1d3', alpha=0.8)
bars3 = ax6.bar(x_comp + width_comp, human_scores, width_comp, label='Human', color='#f38181', alpha=0.8)

ax6.set_ylabel('Normalized Score (Higher is Better)', fontsize=12, fontweight='bold')
ax6.set_title('Overall Performance Comparison (After Correction)', fontsize=14, fontweight='bold')
ax6.set_xticks(x_comp)
ax6.set_xticklabels(datasets, rotation=45, ha='right')
ax6.legend(fontsize=10)
ax6.grid(axis='y', alpha=0.3, linestyle='--')
ax6.set_ylim(0, 110)

# 标注优势
for i, (v7, sota, human) in enumerate(zip(v7_scores, sota_scores, human_scores)):
    if v7 > max(sota, human):
        ax6.text(i - width_comp, v7 + 2, '🏆', ha='center', fontsize=12)

plt.suptitle('V7.2 Model Evaluation: Before vs After Sequence Length Correction',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout()
plt.savefig('versions/v7.2_xy_balanced/correction_comparison.png', dpi=150, bbox_inches='tight')
print("对比图已保存: versions/v7.2_xy_balanced/correction_comparison.png")

# 创建简化的总结图
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('V7.2 Final Results vs SOTA', fontsize=16, fontweight='bold')

# LEV
ax = axes[0]
x = np.arange(len(datasets))
width = 0.35
bars1 = ax.bar(x - width/2, after_lev, width, label='V7.2', color='#4ecdc4', alpha=0.8)
bars2 = ax.bar(x + width/2, sota_lev, width, label='SOTA', color='#95e1d3', alpha=0.8)
ax.axhline(y=np.mean(human_lev), color='#f38181', linestyle='--', linewidth=2, label='Human Avg')
ax.set_ylabel('LEV (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('LEV Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 标注胜负
for i, (v7, sota) in enumerate(zip(after_lev, sota_lev)):
    if v7 < sota:
        ax.text(i, max(v7, sota) + 2, '✓', ha='center', fontsize=14, color='green', fontweight='bold')

# DTW
ax = axes[1]
bars1 = ax.bar(x - width/2, after_dtw, width, label='V7.2', color='#4ecdc4', alpha=0.8)
bars2 = ax.bar(x + width/2, sota_dtw, width, label='SOTA', color='#95e1d3', alpha=0.8)
ax.axhline(y=np.mean(human_dtw), color='#f38181', linestyle='--', linewidth=2, label='Human Avg')
ax.set_ylabel('DTW (Lower is Better)', fontsize=12, fontweight='bold')
ax.set_title('DTW Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 标注胜负
for i, (v7, sota) in enumerate(zip(after_dtw, sota_dtw)):
    if v7 < sota:
        ax.text(i, max(v7, sota) + 100, '✓', ha='center', fontsize=14, color='green', fontweight='bold')

# REC
ax = axes[2]
bars1 = ax.bar(x - width/2, after_rec, width, label='V7.2', color='#4ecdc4', alpha=0.8)
bars2 = ax.bar(x + width/2, sota_rec, width, label='SOTA', color='#95e1d3', alpha=0.8)
ax.axhline(y=np.mean(human_rec), color='#f38181', linestyle='--', linewidth=2, label='Human Avg')
ax.set_ylabel('REC % (Higher is Better)', fontsize=12, fontweight='bold')
ax.set_title('REC Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(datasets, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# 标注胜负
for i, (v7, sota) in enumerate(zip(after_rec, sota_rec)):
    if v7 > sota:
        ax.text(i, max(v7, sota) + 0.3, '✓', ha='center', fontsize=14, color='green', fontweight='bold')

plt.tight_layout()
plt.savefig('versions/v7.2_xy_balanced/final_vs_sota.png', dpi=150, bbox_inches='tight')
print("SOTA对比图已保存: versions/v7.2_xy_balanced/final_vs_sota.png")

print("\n所有对比图表已生成完成！")
