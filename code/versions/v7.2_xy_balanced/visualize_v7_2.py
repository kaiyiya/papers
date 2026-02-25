"""
V7.2可视化脚本 - 生成预测可视化图像
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)
os.chdir(project_root)

# 添加V7.2目录到路径
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, v72_dir)

from config_v7_2 import Config
from model_v7_2 import create_model
from simple_dataset import create_dataloaders


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


def visualize_predictions(model, test_loader, device, output_dir, num_samples=5):
    """生成预测可视化"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if sample_count >= num_samples:
                break
                
            images = batch['image'].to(device)
            gt_scanpaths = batch['scanpath'].to(device)
            saliency_maps = batch.get('saliency_map')
            
            if saliency_maps is None:
                continue
                
            saliency_maps = saliency_maps.to(device)
            pred_scanpaths = model(images, saliency_maps, None, 0.0)
            
            # 转换为numpy
            images_np = images.cpu().numpy()
            pred_np = pred_scanpaths.cpu().numpy()
            gt_np = gt_scanpaths.cpu().numpy()
            
            # 处理batch中的每个样本
            for i in range(min(images.shape[0], num_samples - sample_count)):
                fig, axes = plt.subplots(1, 2, figsize=(16, 6))

                # 反归一化图像
                img = denormalize_image(images_np[i])

                # 左图：Ground Truth - 使用'nearest'插值避免模糊
                axes[0].imshow(img, interpolation='nearest', aspect='auto')
                axes[0].plot(gt_np[i, :, 0] * img.shape[1],
                            gt_np[i, :, 1] * img.shape[0],
                            'go-', linewidth=2, markersize=8, label='GT')
                axes[0].set_title('Ground Truth Scanpath', fontsize=14)
                axes[0].axis('off')
                axes[0].legend()

                # 右图：Prediction - 使用'nearest'插值避免模糊
                axes[1].imshow(img, interpolation='nearest', aspect='auto')
                axes[1].plot(pred_np[i, :, 0] * img.shape[1],
                            pred_np[i, :, 1] * img.shape[0],
                            'ro-', linewidth=2, markersize=8, label='Pred')
                axes[1].set_title('V7.2 Predicted Scanpath', fontsize=14)
                axes[1].axis('off')
                axes[1].legend()

                # 计算覆盖率
                x_range = (pred_np[i, :, 0].max() - pred_np[i, :, 0].min()) * 100
                y_range = (pred_np[i, :, 1].max() - pred_np[i, :, 1].min()) * 100
                xy_ratio = x_range / y_range if y_range > 0 else 0

                fig.suptitle(f'Sample {sample_count + 1} | X覆盖: {x_range:.1f}% | Y覆盖: {y_range:.1f}% | X/Y: {xy_ratio:.2f}',
                            fontsize=16, fontweight='bold')

                plt.tight_layout()
                output_path = os.path.join(output_dir, f'sample_{sample_count + 1:02d}.png')
                # 提高DPI到300，设置高质量PNG保存
                plt.savefig(output_path, dpi=300, bbox_inches='tight',
                            facecolor='white', edgecolor='none',
                            pil_kwargs={'compress_level': 1})
                plt.close()
                
                print(f"✓ 保存: {output_path}")
                sample_count += 1
                
                if sample_count >= num_samples:
                    break


def main():
    parser = argparse.ArgumentParser(description='V7.2可视化')
    parser.add_argument('--checkpoint', type=str, default='best_model_v7_2.pth')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--num_samples', type=int, default=10)
    args = parser.parse_args()

    config = Config()
    v72_dir = os.path.join('versions', 'v7.2_xy_balanced')
    
    if args.epoch is not None:
        checkpoint_name = f'checkpoint_epoch_{args.epoch}.pth'
        output_subdir = f'visualizations_epoch_{args.epoch}'
    else:
        checkpoint_name = args.checkpoint
        output_subdir = 'visualizations_best'
    
    checkpoint_path = os.path.join(v72_dir, 'checkpoints', checkpoint_name)
    output_dir = os.path.join(v72_dir, output_subdir)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return

    device = config.device
    print(f"使用设备: {device}")
    print(f"\n加载数据...")
    _, test_loader = create_dataloaders(config)
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    print(f"\n创建V7.2模型...")
    model = create_model(config).to(device)
    
    print(f"\n加载检查点: {checkpoint_name}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch_num = checkpoint.get('epoch', 'unknown')
    print(f"  Epoch: {epoch_num}")
    
    print(f"\n生成可视化 ({args.num_samples}个样本)...")
    visualize_predictions(model, test_loader, device, output_dir, args.num_samples)
    print(f"\n✓ 可视化完成! 保存在: {output_dir}")


if __name__ == '__main__':
    main()
