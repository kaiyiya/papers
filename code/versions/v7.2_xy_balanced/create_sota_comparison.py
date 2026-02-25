"""
创建V7.2与SOTA模型的对比图表
"""
import matplotlib.pyplot as plt
import numpy as np
import json

# SOTA数据
sota_data = {
    'Salient360': {
        'models': ['Random\nwalk', 'CLE', 'DeepGaze\nIII', 'SaltiNet', 'ScanGAN', 'ScanDMM', 'Human', 'V7.2\n(Ours)'],
        'LEV': [40.802, 39.774, 40.006, 40.848, 38.932, 37.272, 35.084, 36.45],
        'DTW': [2231.681, 1714.409, 1742.351, 1855.477, 1721.711, 1528.592, 1382.590, 1336.26],
        'REC': [2.744, 3.323, 2.588, 2.305, 3.099, 3.576, 5.202, 5.65]
    },
    'AOI': {
        'models': ['Random\nwalk', 'CLE', 'DeepGaze\nIII', 'SaltiNet', 'ScanGAN', 'ScanDMM', 'Human', 'V7.2\n(Ours)'],
        'LEV': [13.696, 12.865, 13.155, 14.695, 12.889, 12.127, 9.243, 36.45],
        'DTW': [711.516, 547.892, 558.445, 596.544, 552.446, 537.504, 389.477, 1232.88],
        'REC': [2.993, 3.617, 2.892, 2.244, 3.750, 4.024, 6.228, 6.34]
    },
    'JUFE': {
        'models': ['Random\nwalk', 'CLE', 'DeepGaze\nIII', 'SaltiNet', 'ScanGAN', 'ScanDMM', 'Human', 'V7.2\n(Ours)'],
        'LEV': [24.039, 24.844, 24.129, 26.074, 24.209, 23.091, 18.306, 35.92],
        'DTW': [1193.725, 1172.150, 1104.848, 1287.144, 1094.978, 1086.014, 1038.880, 1636.25],
        'REC': [3.109, 3.013, 2.774, 1.540, 3.075, 4.329, 7.745, 4.88]
    },
    'Sitzmann': {
        'models': ['Random\nwalk', 'CLE', 'DeepGaze\nIII', 'SaltiNet', 'ScanGAN', 'ScanDMM', 'Human', 'V7.2\n(Ours)'],
        'LEV': [48.942, 45.176, 46.424, 51.370, 45.270, 44.966, 41.188, 36.54],
        'DTW': [2232.987, 1967.286, 1992.859, 2305.099, 1951.848, 1965.427, 1836.986, 1414.52],
        'REC': [2.669, 3.130, 3.082, 1.564, 3.241, 3.475, 6.345, 4.59]
    }
}

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('V7.2 Model vs SOTA Models Comparison', fontsize=20, fontweight='bold', y=0.995)

datasets = ['Salient360', 'AOI', 'JUFE', 'Sitzmann']
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for dataset, (row, col) in zip(datasets, positions):
    ax = axes[row, col]
    data = sota_data[dataset]

    # 创建三个子图（LEV, DTW, REC）
    x = np.arange(len(data['models']))
    width = 0.25

    # 颜色设置
    colors = ['#1f77b4'] * 6 + ['#ff7f0e', '#2ca02c']  # 蓝色(SOTA), 橙色(Human), 绿色(Ours)

    # 绘制LEV
    ax_lev = ax
    bars_lev = ax_lev.bar(x - width, data['LEV'], width, label='LEV↓', color=colors, alpha=0.8)
    ax_lev.set_ylabel('LEV (Lower is Better)', fontsize=12, fontweight='bold')
    ax_lev.set_title(f'{dataset} Dataset', fontsize=14, fontweight='bold', pad=10)
    ax_lev.set_xticks(x)
    ax_lev.set_xticklabels(data['models'], rotation=45, ha='right', fontsize=9)
    ax_lev.grid(axis='y', alpha=0.3, linestyle='--')

    # 标注V7.2的值
    for i, (bar, val) in enumerate(zip(bars_lev, data['LEV'])):
        if i == len(data['LEV']) - 1:  # V7.2
            height = bar.get_height()
            ax_lev.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}',
                       ha='center', va='bottom', fontweight='bold', fontsize=10)

    # 创建第二个y轴用于DTW
    ax_dtw = ax_lev.twinx()
    bars_dtw = ax_dtw.bar(x, data['DTW'], width, label='DTW↓', color=colors, alpha=0.6)
    ax_dtw.set_ylabel('DTW (Lower is Better)', fontsize=12, fontweight='bold')

    # 创建第三个y轴用于REC
    ax_rec = ax_lev.twinx()
    ax_rec.spines['right'].set_position(('outward', 60))
    bars_rec = ax_rec.bar(x + width, data['REC'], width, label='REC↑', color=colors, alpha=0.4)
    ax_rec.set_ylabel('REC % (Higher is Better)', fontsize=12, fontweight='bold')

    # 添加图例
    lines1, labels1 = ax_lev.get_legend_handles_labels()
    lines2, labels2 = ax_dtw.get_legend_handles_labels()
    lines3, labels3 = ax_rec.get_legend_handles_labels()
    ax_lev.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3,
                  loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig('versions/v7.2_xy_balanced/v7_2_vs_sota_comparison.png', dpi=150, bbox_inches='tight')
print("对比图已保存: versions/v7.2_xy_balanced/v7_2_vs_sota_comparison.png")

# 创建简化版对比表
fig2, axes2 = plt.subplots(2, 2, figsize=(20, 12))
fig2.suptitle('V7.2 vs SOTA: Simplified Comparison', fontsize=18, fontweight='bold')

for dataset, (row, col) in zip(datasets, positions):
    ax = axes2[row, col]
    data = sota_data[dataset]

    # 只显示最佳SOTA、Human和V7.2
    models_simple = ['Best SOTA', 'Human', 'V7.2 (Ours)']

    # 找到每个指标的最佳SOTA值
    best_lev_idx = np.argmin(data['LEV'][:6])
    best_dtw_idx = np.argmin(data['DTW'][:6])
    best_rec_idx = np.argmax(data['REC'][:6])

    lev_simple = [data['LEV'][best_lev_idx], data['LEV'][6], data['LEV'][7]]
    dtw_simple = [data['DTW'][best_dtw_idx], data['DTW'][6], data['DTW'][7]]
    rec_simple = [data['REC'][best_rec_idx], data['REC'][6], data['REC'][7]]

    x = np.arange(len(models_simple))
    width = 0.25

    # 归一化显示
    lev_norm = np.array(lev_simple) / max(lev_simple) * 100
    dtw_norm = np.array(dtw_simple) / max(dtw_simple) * 100
    rec_norm = np.array(rec_simple) / max(rec_simple) * 100

    bars1 = ax.bar(x - width, lev_norm, width, label='LEV↓ (normalized)', color='#1f77b4', alpha=0.8)
    bars2 = ax.bar(x, dtw_norm, width, label='DTW↓ (normalized)', color='#ff7f0e', alpha=0.8)
    bars3 = ax.bar(x + width, rec_norm, width, label='REC↑ (normalized)', color='#2ca02c', alpha=0.8)

    ax.set_ylabel('Normalized Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{dataset} Dataset', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models_simple, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 120)

    # 添加实际值标注
    for bars, values in [(bars1, lev_simple), (bars2, dtw_simple), (bars3, rec_simple)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{val:.1f}',
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('versions/v7.2_xy_balanced/v7_2_vs_sota_simplified.png', dpi=150, bbox_inches='tight')
print("简化对比图已保存: versions/v7.2_xy_balanced/v7_2_vs_sota_simplified.png")

# 创建性能雷达图
fig3, axes3 = plt.subplots(2, 2, figsize=(16, 16), subplot_kw=dict(projection='polar'))
fig3.suptitle('V7.2 vs Human Performance (Radar Chart)', fontsize=18, fontweight='bold')

for dataset, (row, col) in zip(datasets, positions):
    ax = axes3[row, col]
    data = sota_data[dataset]

    # 三个指标
    categories = ['LEV\n(inverted)', 'DTW\n(inverted)', 'REC']
    N = len(categories)

    # 归一化到0-100，LEV和DTW需要反转（越小越好）
    human_lev = (1 - data['LEV'][6] / max(data['LEV'])) * 100
    human_dtw = (1 - data['DTW'][6] / max(data['DTW'])) * 100
    human_rec = data['REC'][6] / max(data['REC']) * 100

    v72_lev = (1 - data['LEV'][7] / max(data['LEV'])) * 100
    v72_dtw = (1 - data['DTW'][7] / max(data['DTW'])) * 100
    v72_rec = data['REC'][7] / max(data['REC']) * 100

    human_values = [human_lev, human_dtw, human_rec]
    v72_values = [v72_lev, v72_dtw, v72_rec]

    # 闭合雷达图
    human_values += human_values[:1]
    v72_values += v72_values[:1]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # 绘制
    ax.plot(angles, human_values, 'o-', linewidth=2, label='Human', color='#ff7f0e')
    ax.fill(angles, human_values, alpha=0.25, color='#ff7f0e')
    ax.plot(angles, v72_values, 'o-', linewidth=2, label='V7.2 (Ours)', color='#2ca02c')
    ax.fill(angles, v72_values, alpha=0.25, color='#2ca02c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title(f'{dataset}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)

plt.tight_layout()
plt.savefig('versions/v7.2_xy_balanced/v7_2_vs_human_radar.png', dpi=150, bbox_inches='tight')
print("雷达图已保存: versions/v7.2_xy_balanced/v7_2_vs_human_radar.png")

print("\n所有对比图表已生成完成！")
