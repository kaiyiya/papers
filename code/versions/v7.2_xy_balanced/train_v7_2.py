"""
V7.2训练脚本 - X-Y平衡优化模型
改进：同时优化X和Y方向，X方向权重更高
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

# 切换到项目根目录
os.chdir(project_root)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️ TensorBoard not available")

# 导入V7.2模块
v72_dir = os.path.join(project_root, 'versions', 'v7.2_xy_balanced')
sys.path.insert(0, v72_dir)

from config_v7_2 import Config
from model_v7_2 import create_model
from improved_coverage_loss import ImprovedCoverageLoss
from simple_dataset import create_dataloaders
from losses.soft_dtw_loss import SoftDTWLoss

try:
    from metrics.metrics import compute_metrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("⚠️ Metrics module not available")


def train_epoch(model, train_loader, optimizer, path_criterion, coverage_criterion, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_path_loss = 0
    total_coverage_loss = 0
    num_batches = 0

    # Teacher forcing ratio (指数衰减)
    tf_ratio = max(0.0, 0.8 * (0.95 ** epoch))

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    for batch in pbar:
        images = batch['image'].to(device)
        gt_scanpaths = batch['scanpath'].to(device)
        saliency_maps = batch.get('saliency_map')

        if saliency_maps is None:
            continue

        saliency_maps = saliency_maps.to(device)

        # 前向传播
        pred_scanpaths = model(images, saliency_maps, gt_scanpaths, tf_ratio)

        # 计算路径损失
        path_loss = path_criterion(pred_scanpaths, gt_scanpaths)

        # 计算覆盖率损失
        coverage_loss = coverage_criterion(pred_scanpaths)

        # 总损失
        loss = path_loss + config.coverage_loss_weight * coverage_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_path_loss += path_loss.item()
        total_coverage_loss += coverage_loss.item()
        num_batches += 1

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'path': f'{path_loss.item():.4f}',
            'cov': f'{coverage_loss.item():.4f}',
            'tf': f'{tf_ratio:.3f}'
        })

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_path_loss = total_path_loss / num_batches if num_batches > 0 else 0
    avg_coverage_loss = total_coverage_loss / num_batches if num_batches > 0 else 0

    return avg_loss, avg_path_loss, avg_coverage_loss, tf_ratio


def validate(model, test_loader, path_criterion, coverage_criterion, device, config):
    """验证模型"""
    model.eval()
    total_loss = 0
    total_path_loss = 0
    total_coverage_loss = 0
    num_batches = 0

    all_stats = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Validating"):
            images = batch['image'].to(device)
            gt_scanpaths = batch['scanpath'].to(device)
            saliency_maps = batch.get('saliency_map')

            if saliency_maps is None:
                continue

            saliency_maps = saliency_maps.to(device)

            # 前向传播（无teacher forcing）
            pred_scanpaths = model(images, saliency_maps, None, 0.0)

            # 计算损失
            path_loss = path_criterion(pred_scanpaths, gt_scanpaths)
            coverage_loss = coverage_criterion(pred_scanpaths)
            loss = path_loss + config.coverage_loss_weight * coverage_loss

            total_loss += loss.item()
            total_path_loss += path_loss.item()
            total_coverage_loss += coverage_loss.item()
            num_batches += 1

            # 获取覆盖率统计（包含X和Y）
            stats = coverage_criterion.get_stats(pred_scanpaths)
            all_stats.append(stats)

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    avg_path_loss = total_path_loss / num_batches if num_batches > 0 else 0
    avg_coverage_loss = total_coverage_loss / num_batches if num_batches > 0 else 0

    # 汇总覆盖率统计
    coverage_stats = {
        'x_std': np.mean([s['x_std'] for s in all_stats]),
        'x_range': np.mean([s['x_range'] for s in all_stats]),
        'x_coverage': np.mean([s['x_coverage'] for s in all_stats]),
        'y_std': np.mean([s['y_std'] for s in all_stats]),
        'y_range': np.mean([s['y_range'] for s in all_stats]),
        'y_coverage': np.mean([s['y_coverage'] for s in all_stats]),
        'xy_ratio': np.mean([s['xy_ratio'] for s in all_stats])
    }

    return avg_loss, avg_path_loss, avg_coverage_loss, coverage_stats


def main():
    config = Config()

    # V7.2专用目录
    v72_dir = os.path.join('versions', 'v7.2_xy_balanced')
    log_dir = os.path.join(v72_dir, 'logs')
    checkpoint_dir = os.path.join(v72_dir, 'checkpoints')

    # 创建目录
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 设置设备
    device = config.device
    print(f"使用设备: {device}")

    # 创建数据加载器
    print("\n加载数据...")
    train_loader, test_loader = create_dataloaders(config)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # 创建模型
    print("\n创建V7.2模型...")
    model = create_model(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失函数
    path_criterion = SoftDTWLoss(gamma=0.1)
    coverage_criterion = ImprovedCoverageLoss(
        x_weight=config.x_coverage_weight,
        y_weight=config.y_coverage_weight
    )

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.num_epochs,
        eta_min=1e-6
    )

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir) if HAS_TENSORBOARD else None

    # 训练历史
    history = {
        'train_loss': [],
        'train_path_loss': [],
        'train_coverage_loss': [],
        'val_loss': [],
        'val_path_loss': [],
        'val_coverage_loss': [],
        'val_x_coverage': [],
        'val_y_coverage': [],
        'val_xy_ratio': [],
        'tf_ratio': []
    }

    best_val_loss = float('inf')

    print("\n开始训练V7.2...")
    print(f"分层采样: 前40%步数60%探索 -> 后60%步数30%探索")
    print(f"覆盖率损失权重: {config.coverage_loss_weight}")
    print(f"X方向权重: {config.x_coverage_weight}, Y方向权重: {config.y_coverage_weight}")
    print()

    for epoch in range(config.num_epochs):
        # 训练
        train_loss, train_path_loss, train_coverage_loss, tf_ratio = train_epoch(
            model, train_loader, optimizer, path_criterion, coverage_criterion,
            device, epoch, config
        )

        history['train_loss'].append(train_loss)
        history['train_path_loss'].append(train_path_loss)
        history['train_coverage_loss'].append(train_coverage_loss)
        history['tf_ratio'].append(tf_ratio)

        # 验证
        if (epoch + 1) % config.val_interval == 0:
            val_loss, val_path_loss, val_coverage_loss, coverage_stats = validate(
                model, test_loader, path_criterion, coverage_criterion, device, config
            )

            history['val_loss'].append(val_loss)
            history['val_path_loss'].append(val_path_loss)
            history['val_coverage_loss'].append(val_coverage_loss)
            history['val_x_coverage'].append(coverage_stats['x_coverage'])
            history['val_y_coverage'].append(coverage_stats['y_coverage'])
            history['val_xy_ratio'].append(coverage_stats['xy_ratio'])

            # TensorBoard记录
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Loss/train_path', train_path_loss, epoch)
                writer.add_scalar('Loss/val_path', val_path_loss, epoch)
                writer.add_scalar('Loss/train_coverage', train_coverage_loss, epoch)
                writer.add_scalar('Loss/val_coverage', val_coverage_loss, epoch)
                writer.add_scalar('Coverage/x_coverage', coverage_stats['x_coverage'], epoch)
                writer.add_scalar('Coverage/y_coverage', coverage_stats['y_coverage'], epoch)
                writer.add_scalar('Coverage/xy_ratio', coverage_stats['xy_ratio'], epoch)
                writer.add_scalar('TF_ratio', tf_ratio, epoch)

            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f} (path: {train_path_loss:.4f}, cov: {train_coverage_loss:.4f})")
            print(f"  Val Loss: {val_loss:.4f} (path: {val_path_loss:.4f}, cov: {val_coverage_loss:.4f})")
            print(f"  Coverage: X={coverage_stats['x_coverage']:.1f}%, Y={coverage_stats['y_coverage']:.1f}%, X/Y={coverage_stats['xy_ratio']:.2f}")

            # 早停警告
            if coverage_stats['x_coverage'] < config.early_stop_x_coverage * 100:
                print(f"  ⚠️ X覆盖率过低: {coverage_stats['x_coverage']:.1f}%")
            if coverage_stats['y_coverage'] > config.early_stop_y_coverage * 100:
                print(f"  ⚠️ Y覆盖率过高: {coverage_stats['y_coverage']:.1f}%")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'coverage_stats': coverage_stats
                }, os.path.join(checkpoint_dir, 'best_model_v7_2.pth'))
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")

        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # 更新学习率
        scheduler.step()

    # 保存训练历史
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(log_dir, f'training_history_v7_2_{timestamp}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()
    print("\nV7.2训练完成!")


if __name__ == '__main__':
    main()
