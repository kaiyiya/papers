"""
V7.2模型完整测试 - 支持所有数据集
生成16条眼动序列可视化和评估指标
"""
import os
import sys
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import pickle

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 添加V7.2目录到路径
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, v72_dir)

from config_v7_2 import Config
from model_v7_2 import create_model
from torch.utils.data import Dataset, DataLoader

# 导入评估指标
import editdistance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def sphere_to_equirect(xyz):
    """将3D球面坐标转换为2D等距柱状投影坐标"""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z, -1, 1))
    u = (lon + np.pi) / (2 * np.pi)
    v = (lat + np.pi/2) / np.pi
    return np.stack([u, v], axis=-1)


def denormalize_image(image_array):
    """反归一化ImageNet图像"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if image_array.ndim == 3 and image_array.shape[0] == 3:
        image_array = np.transpose(image_array, (1, 2, 0))
    image_array = image_array * std + mean
    image_array = np.clip(image_array, 0, 1)
    image_array = (image_array * 255).astype(np.uint8)
    return image_array


class UnifiedScanpathDataset(Dataset):
    """统一的扫描路径数据集"""
    def __init__(self, data_dict, dataset_name, seq_len=25):
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.samples = []
        self.image_groups = {}  # 按图片分组

        if dataset_name == 'AOI':
            self._load_aoi(data_dict)
        elif dataset_name == 'JUFE':
            self._load_jufe(data_dict)
        elif dataset_name == 'Sitzmann':
            self._load_sitzmann(data_dict)
        elif dataset_name == 'Salient360':
            self._load_salient360(data_dict)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def _load_aoi(self, data_dict):
        """加载AOI数据集"""
        print(f"Loading AOI dataset...")
        for img_key, img_data in data_dict.items():
            image = img_data['image']
            salmap = img_data['salmap']
            scanpaths_3d = img_data['scanpaths']

            if img_key not in self.image_groups:
                self.image_groups[img_key] = []

            for sp_3d in scanpaths_3d:
                sp_2d = sphere_to_equirect(sp_3d)
                original_len = len(sp_2d)  # 保存原始长度

                if len(sp_2d) >= self.seq_len:
                    sp_2d = sp_2d[:self.seq_len]
                else:
                    pad_len = self.seq_len - len(sp_2d)
                    sp_2d = np.vstack([sp_2d, np.tile(sp_2d[-1:], (pad_len, 1))])

                sample = {
                    'image': image,
                    'scanpath': torch.FloatTensor(sp_2d),
                    'saliency_map': torch.FloatTensor(salmap).unsqueeze(0) / 255.0,
                    'key': img_key,
                    'original_len': original_len  # 添加原始长度
                }
                self.samples.append(sample)
                self.image_groups[img_key].append(len(self.samples) - 1)

        print(f"Loaded {len(self.samples)} samples from {len(self.image_groups)} images")

    def _load_jufe(self, data_dict):
        """加载JUFE数据集"""
        print(f"Loading JUFE dataset...")
        for sample_key, sample_data in data_dict.items():
            image = sample_data['image']
            scanpaths_3d = sample_data['scanpaths']
            salmap = torch.ones(1, 128, 256)

            if sample_key not in self.image_groups:
                self.image_groups[sample_key] = []

            for sp_3d in scanpaths_3d:
                sp_2d = sphere_to_equirect(sp_3d)
                original_len = len(sp_2d)  # 保存原始长度

                if len(sp_2d) >= self.seq_len:
                    sp_2d = sp_2d[:self.seq_len]
                else:
                    pad_len = self.seq_len - len(sp_2d)
                    sp_2d = np.vstack([sp_2d, np.tile(sp_2d[-1:], (pad_len, 1))])

                sample = {
                    'image': image,
                    'scanpath': torch.FloatTensor(sp_2d),
                    'saliency_map': salmap,
                    'key': sample_key,
                    'original_len': original_len  # 添加原始长度
                }
                self.samples.append(sample)
                self.image_groups[sample_key].append(len(self.samples) - 1)

        print(f"Loaded {len(self.samples)} samples from {len(self.image_groups)} images")

    def _load_sitzmann(self, data_dict):
        """加载Sitzmann数据集"""
        print(f"Loading Sitzmann dataset...")
        for img_key, img_data in data_dict.items():
            image = img_data['image']
            salmap = img_data['salmap']
            scanpaths = img_data['scanpaths']

            if img_key not in self.image_groups:
                self.image_groups[img_key] = []

            for sp in scanpaths:
                sp_array = np.array(sp)
                if sp_array.shape[1] == 3:
                    sp_array = sphere_to_equirect(sp_array)

                original_len = len(sp_array)  # 保存原始长度

                if len(sp_array) >= self.seq_len:
                    sp_array = sp_array[:self.seq_len]
                else:
                    pad_len = self.seq_len - len(sp_array)
                    sp_array = np.vstack([sp_array, np.tile(sp_array[-1:], (pad_len, 1))])

                sample = {
                    'image': image,
                    'scanpath': torch.FloatTensor(sp_array),
                    'saliency_map': torch.FloatTensor(salmap).unsqueeze(0),
                    'key': img_key,
                    'original_len': original_len  # 添加原始长度
                }
                self.samples.append(sample)
                self.image_groups[img_key].append(len(self.samples) - 1)

        print(f"Loaded {len(self.samples)} samples from {len(self.image_groups)} images")

    def _load_salient360(self, data_dict):
        """加载Salient360数据集"""
        print(f"Loading Salient360 dataset...")
        for img_key, img_data in data_dict.items():
            image = img_data['image']
            salmap = img_data['salmap']
            scanpaths = img_data['scanpaths']

            if img_key not in self.image_groups:
                self.image_groups[img_key] = []

            for sp in scanpaths:
                sp_array = np.array(sp)

                # 如果是3D坐标，转换为2D
                if sp_array.shape[1] == 3:
                    sp_array = sphere_to_equirect(sp_array)

                original_len = len(sp_array)  # 保存原始长度

                if len(sp_array) >= self.seq_len:
                    sp_array = sp_array[:self.seq_len]
                else:
                    pad_len = self.seq_len - len(sp_array)
                    sp_array = np.vstack([sp_array, np.tile(sp_array[-1:], (pad_len, 1))])

                sample = {
                    'image': image,
                    'scanpath': torch.FloatTensor(sp_array),
                    'saliency_map': torch.FloatTensor(salmap).unsqueeze(0),
                    'key': img_key,
                    'original_len': original_len  # 添加原始长度
                }
                self.samples.append(sample)
                self.image_groups[img_key].append(len(self.samples) - 1)

        print(f"Loaded {len(self.samples)} samples from {len(self.image_groups)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def scanpath_to_string(scanpath, height_width, Xbins, Ybins):
    """将扫描路径转换为字符串（用于LEV计算）"""
    height, width = height_width
    height_step, width_step = height // Ybins, width // Xbins
    string = ''

    for i in range(scanpath.shape[0]):
        fixation = scanpath[i].astype(np.int32)
        xbin = min(Xbins-1, max(0, fixation[0] // width_step))
        ybin = min(Ybins-1, max(0, (height - fixation[1]) // height_step))
        corrs_x = chr(65 + xbin)
        corrs_y = chr(97 + ybin)
        string += (corrs_y + corrs_x)

    return string


def compute_metrics(pred_pixels, gt_pixels, height_width):
    """计算评估指标"""
    height, width = height_width

    # LEV - 使用12x8网格离散化
    pred_str = scanpath_to_string(pred_pixels, height_width, Xbins=12, Ybins=8)
    gt_str = scanpath_to_string(gt_pixels, height_width, Xbins=12, Ybins=8)
    lev = editdistance.eval(pred_str, gt_str)

    # DTW
    dtw_dist, _ = fastdtw(pred_pixels, gt_pixels, dist=euclidean)

    # REC - 交叉递归率（预测和GT之间）
    threshold = 2 * 6
    min_len = min(len(pred_pixels), len(gt_pixels))
    pred_trimmed = pred_pixels[:min_len]
    gt_trimmed = gt_pixels[:min_len]

    rec_count = 0
    for i in range(min_len):
        for j in range(min_len):
            dist = np.linalg.norm(pred_trimmed[i] - gt_trimmed[j])
            if dist < threshold:
                rec_count += 1

    # 只计算上三角部分（不包括对角线）
    rec_upper = 0
    for i in range(min_len):
        for j in range(i + 1, min_len):
            dist = np.linalg.norm(pred_trimmed[i] - gt_trimmed[j])
            if dist < threshold:
                rec_upper += 1

    total_pairs = min_len * (min_len - 1)
    rec = min(100.0, 100 * (2 * rec_upper) / total_pairs) if total_pairs > 0 else 0.0

    return lev, dtw_dist, rec


def visualize_16_scanpaths(image, pred_scanpaths, gt_scanpaths, output_path, img_key):
    """生成16条眼动序列的可视化（4x4网格）"""
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle(f'{img_key} - 16 Scanpath Predictions', fontsize=16, fontweight='bold')

    img = denormalize_image(image)

    for idx in range(16):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        # 使用'nearest'插值避免模糊，关闭抗锯齿
        ax.imshow(img, interpolation='nearest', aspect='auto')

        if idx < len(pred_scanpaths):
            pred = pred_scanpaths[idx]
            gt = gt_scanpaths[idx]

            # 绘制GT（绿色）
            ax.plot(gt[:, 0] * img.shape[1], gt[:, 1] * img.shape[0],
                   'go-', linewidth=1.5, markersize=4, alpha=0.6, label='GT')

            # 绘制预测（红色）
            ax.plot(pred[:, 0] * img.shape[1], pred[:, 1] * img.shape[0],
                   'ro-', linewidth=1.5, markersize=4, alpha=0.8, label='Pred')

            ax.set_title(f'Scanpath {idx+1}', fontsize=10)
        else:
            ax.set_title(f'Scanpath {idx+1} (N/A)', fontsize=10)

        ax.axis('off')
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    # 提高DPI到300，设置高质量PNG保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none',
                pil_kwargs={'compress_level': 1})
    plt.close()


def test_on_dataset(model, dataset_name, dataset_path, device, config, output_dir):
    """在指定数据集上测试模型"""
    print(f"\n{'='*70}")
    print(f"测试数据集: {dataset_name}")
    print(f"{'='*70}")
    
    # 创建输出目录
    dataset_output_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 加载数据
    try:
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        if dataset_name in ['AOI', 'Sitzmann', 'Salient360']:
            test_data = data['test']
        elif dataset_name == 'JUFE':
            test_data = data
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset = UnifiedScanpathDataset(test_data, dataset_name, config.seq_len)
        
        print(f"加载成功: {len(dataset)} 个样本, {len(dataset.image_groups)} 张图片")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 评估
    model.eval()
    all_metrics = []
    
    height, width = 128, 256
    height_width = (height, width)
    
    print("\n开始生成预测和可视化...")
    
    # 按图片分组处理
    with torch.no_grad():
        for img_idx, (img_key, sample_indices) in enumerate(tqdm(dataset.image_groups.items(), desc="处理图片")):
            # 取前16条scanpath（如果有的话）
            selected_indices = sample_indices[:16]
            
            # 批量预测
            batch_samples = [dataset[idx] for idx in selected_indices]
            images = torch.stack([s['image'] for s in batch_samples]).to(device)
            gt_scanpaths = torch.stack([s['scanpath'] for s in batch_samples]).to(device)
            saliency_maps = torch.stack([s['saliency_map'] for s in batch_samples]).to(device)
            
            # 预测
            pred_scanpaths = model(images, saliency_maps, None, 0.0)
            
            # 转换为numpy
            pred_np = pred_scanpaths.cpu().numpy()
            gt_np = gt_scanpaths.cpu().numpy()
            image_np = images[0].cpu().numpy()
            
            # 计算指标
            for i in range(len(pred_np)):
                # 获取原始长度
                original_len = batch_samples[i]['original_len']

                # 只使用原始长度的部分进行评估
                pred_pixels = pred_np[i, :original_len] * np.array([width, height])
                gt_pixels = gt_np[i, :original_len] * np.array([width, height])

                lev, dtw_dist, rec = compute_metrics(pred_pixels, gt_pixels, height_width)

                # 计算覆盖率（使用原始长度）
                x_range = (pred_np[i, :original_len, 0].max() - pred_np[i, :original_len, 0].min()) * 100
                y_range = (pred_np[i, :original_len, 1].max() - pred_np[i, :original_len, 1].min()) * 100
                xy_ratio = x_range / y_range if y_range > 0 else 0

                all_metrics.append({
                    'image': img_key,
                    'scanpath_idx': i,
                    'LEV': float(lev),
                    'DTW': float(dtw_dist),
                    'REC': float(rec),
                    'x_coverage': float(x_range),
                    'y_coverage': float(y_range),
                    'xy_ratio': float(xy_ratio),
                    'original_len': original_len  # 记录原始长度
                })
            
            # 生成可视化
            vis_path = os.path.join(dataset_output_dir, f'{img_key}_16scanpaths.png')
            visualize_16_scanpaths(image_np, pred_np, gt_np, vis_path, img_key)
    
    # 汇总统计
    if all_metrics:
        results = {
            'dataset': dataset_name,
            'num_samples': len(all_metrics),
            'num_images': len(dataset.image_groups),
            'metrics': {
                'LEV_mean': float(np.mean([m['LEV'] for m in all_metrics])),
                'LEV_std': float(np.std([m['LEV'] for m in all_metrics])),
                'DTW_mean': float(np.mean([m['DTW'] for m in all_metrics])),
                'DTW_std': float(np.std([m['DTW'] for m in all_metrics])),
                'REC_mean': float(np.mean([m['REC'] for m in all_metrics])),
                'REC_std': float(np.std([m['REC'] for m in all_metrics])),
                'x_coverage_mean': float(np.mean([m['x_coverage'] for m in all_metrics])),
                'x_coverage_std': float(np.std([m['x_coverage'] for m in all_metrics])),
                'y_coverage_mean': float(np.mean([m['y_coverage'] for m in all_metrics])),
                'y_coverage_std': float(np.std([m['y_coverage'] for m in all_metrics])),
                'xy_ratio_mean': float(np.mean([m['xy_ratio'] for m in all_metrics])),
                'xy_ratio_std': float(np.std([m['xy_ratio'] for m in all_metrics]))
            },
            'per_sample_metrics': all_metrics
        }
        
        # 保存结果
        results_path = os.path.join(dataset_output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*70}")
        print(f"{dataset_name} 评估结果:")
        print(f"{'='*70}")
        print(f"样本数: {results['num_samples']}")
        print(f"图片数: {results['num_images']}")
        print(f"\n核心指标:")
        print(f"  LEV: {results['metrics']['LEV_mean']:.2f} ± {results['metrics']['LEV_std']:.2f}")
        print(f"  DTW: {results['metrics']['DTW_mean']:.2f} ± {results['metrics']['DTW_std']:.2f}")
        print(f"  REC: {results['metrics']['REC_mean']:.2f}% ± {results['metrics']['REC_std']:.2f}%")
        print(f"\n覆盖率:")
        print(f"  X覆盖: {results['metrics']['x_coverage_mean']:.1f}% ± {results['metrics']['x_coverage_std']:.1f}%")
        print(f"  Y覆盖: {results['metrics']['y_coverage_mean']:.1f}% ± {results['metrics']['y_coverage_std']:.1f}%")
        print(f"  X/Y比例: {results['metrics']['xy_ratio_mean']:.2f} ± {results['metrics']['xy_ratio_std']:.2f}")
        print(f"\n结果已保存: {results_path}")
        print(f"可视化已保存: {dataset_output_dir}/")
        
        return results
    
    return None


def main():
    config = Config()
    
    # 输出目录
    output_dir = os.path.join('versions', 'v7.2_xy_balanced', 'test_results_all_datasets')
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置设备
    device = config.device
    print(f"使用设备: {device}")
    
    # 加载最佳模型
    checkpoint_path = os.path.join('versions', 'v7.2_xy_balanced', 'checkpoints', 'best_model_v7_2.pth')
    print(f"\n加载模型: {checkpoint_path}")
    
    model = create_model(config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功 (Epoch {checkpoint.get('epoch', 'unknown')})")
    
    # 数据集配置
    datasets = {
        'Salient360': 'datasets/Salient360.pkl',
        'AOI': 'datasets/AOI.pkl',
        'JUFE': 'datasets/JUFE.pkl',
        'Sitzmann': 'datasets/Sitzmann.pkl'
    }
    
    # 测试所有数据集
    all_results = {}
    for dataset_name, dataset_path in datasets.items():
        if os.path.exists(dataset_path):
            results = test_on_dataset(model, dataset_name, dataset_path, device, config, output_dir)
            if results:
                all_results[dataset_name] = results
        else:
            print(f"\n⚠️ 数据集不存在: {dataset_path}")
    
    # 保存汇总结果
    summary_path = os.path.join(output_dir, 'summary_all_datasets.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"所有数据集测试完成!")
    print(f"汇总结果已保存: {summary_path}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
