"""
训练脚本 - 显著性引导的Mamba模型
"""
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("⚠️ TensorBoard not available, skipping tensorboard logging")

from config import Config
from model import create_model
from simple_dataset import create_dataloaders
from losses.soft_dtw_loss import SoftDTWLoss

try:
    from metrics.metrics import compute_metrics
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("⚠️ Metrics module not available, skipping metric computation")


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, config):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    # Teacher forcing ratio (指数衰减)
    tf_ratio = max(0.0, 0.8 * (0.95 ** epoch))

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    for batch in pbar:
        images = batch['image'].to(device)
        gt_scanpaths = batch['scanpath'].to(device)
        saliency_maps = batch.get('saliency_map')

        if saliency_maps is None:
            print("⚠️ 警告：批次中缺少显著性图，跳过")
            continue

        saliency_maps = saliency_maps.to(device)

        # 前向传播
        pred_scanpaths = model(images, saliency_maps, gt_scanpaths, tf_ratio)

        # 计算损失
        loss = criterion(pred_scanpaths, gt_scanpaths)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'tf_ratio': f'{tf_ratio:.3f}'})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, tf_ratio


def validate(model, test_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    num_batches = 0

    all_metrics = {
        'dtw': [],
        'frechet': [],
        'hausdorff': [],
        'eyenalysis': []
    }

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
            loss = criterion(pred_scanpaths, gt_scanpaths)
            total_loss += loss.item()
            num_batches += 1

            # 计算指标（如果可用）
            if HAS_METRICS:
                pred_np = pred_scanpaths.cpu().numpy()
                gt_np = gt_scanpaths.cpu().numpy()

                for i in range(len(pred_np)):
                    metrics = compute_metrics(pred_np[i], gt_np[i])
                    for key in all_metrics:
                        if key in metrics:
                            all_metrics[key].append(metrics[key])

    avg_loss = total_loss / num_batches if num_batches > 0 else 0

    # 计算平均指标
    avg_metrics = {}
    if HAS_METRICS:
        for key in all_metrics:
            if len(all_metrics[key]) > 0:
                avg_metrics[key] = np.mean(all_metrics[key])

    return avg_loss, avg_metrics


def main():
    config = Config()

    # 创建目录
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    # 设置设备
    device = config.device
    print(f"使用设备: {device}")

    # 创建数据加载器
    print("\n加载数据...")
    train_loader, test_loader = create_dataloaders(config)
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")

    # 创建模型
    print("\n创建模型...")
    model = create_model(config).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = SoftDTWLoss(gamma=0.1)
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
    writer = SummaryWriter(log_dir=config.log_dir) if HAS_TENSORBOARD else None

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': [],
        'tf_ratio': []
    }

    best_val_loss = float('inf')

    print("\n开始训练...")
    for epoch in range(config.num_epochs):
        # 训练
        train_loss, tf_ratio = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, config
        )

        history['train_loss'].append(train_loss)
        history['tf_ratio'].append(tf_ratio)

        # 验证
        if (epoch + 1) % config.val_interval == 0:
            val_loss, val_metrics = validate(model, test_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)

            # TensorBoard记录
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('TF_ratio', tf_ratio, epoch)

                for key, value in val_metrics.items():
                    writer.add_scalar(f'Metrics/{key}', value, epoch)

            print(f"\nEpoch {epoch+1}/{config.num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Metrics: {val_metrics}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics
                }, os.path.join(config.checkpoint_dir, 'best_model.pth'))
                print(f"  ✓ 保存最佳模型 (val_loss: {val_loss:.4f})")

        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss
            }, os.path.join(config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # 更新学习率
        scheduler.step()

    # 保存训练历史
    with open(os.path.join(config.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    if writer:
        writer.close()
    print("\n训练完成!")


if __name__ == '__main__':
    main()
