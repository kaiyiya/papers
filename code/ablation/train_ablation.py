"""
消融实验训练脚本
用法:
  python ablation/train_ablation.py --variant full
  python ablation/train_ablation.py --variant no_coord_att
  python ablation/train_ablation.py --variant no_hierarchical
  python ablation/train_ablation.py --variant no_coverage_loss
  python ablation/train_ablation.py --variant lstm_baseline
"""
import os
import sys
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(project_root)

from ablation_config import AblationConfig
from ablation_models import create_model
from simple_dataset import create_dataloaders
from losses.soft_dtw_loss import SoftDTWLoss

VARIANTS = ['full', 'no_coord_att', 'no_hierarchical', 'no_coverage_loss', 'lstm_baseline']


class CoverageLoss(nn.Module):
    def __init__(self, x_weight=2.0, y_weight=1.0):
        super().__init__()
        self.x_weight = x_weight
        self.y_weight = y_weight

    def forward(self, scanpaths):
        x = scanpaths[:, :, 0]
        y = scanpaths[:, :, 1]
        x_loss = -((x.std(dim=1) + (x.max(dim=1)[0] - x.min(dim=1)[0])) / 2).mean()
        y_loss = -((y.std(dim=1) + (y.max(dim=1)[0] - y.min(dim=1)[0])) / 2).mean()
        return self.x_weight * x_loss + self.y_weight * y_loss


def train(variant, run=0):
    config = AblationConfig()
    device = config.device
    print(f"Variant: {variant}  run: {run}  |  Device: {device}")

    checkpoint_dir = os.path.join('ablation', variant, f'run{run}', 'checkpoints')
    log_dir        = os.path.join('ablation', variant, f'run{run}', 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 数据
    train_loader, val_loader = create_dataloaders(config)
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}")

    # 模型
    model = create_model(config, variant=variant).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # 损失
    path_criterion = SoftDTWLoss(gamma=0.1, normalize=True)
    use_coverage = (variant != 'no_coverage_loss')
    coverage_criterion = CoverageLoss(config.x_coverage_weight, config.y_coverage_weight)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)

    best_val_loss = float('inf')
    history = []

    for epoch in range(config.num_epochs):
        # ── 训练 ──
        model.train()
        tf_ratio = max(0.0, 0.8 * (0.95 ** epoch))
        train_losses = []

        for batch in tqdm(train_loader, desc=f"[{variant}] Epoch {epoch+1}/{config.num_epochs}", leave=False):
            images      = batch['image'].to(device)
            gt          = batch['scanpath'].to(device)
            salmaps     = batch.get('saliency_map')
            if salmaps is None:
                continue
            salmaps = salmaps.to(device)

            pred = model(images, salmaps, gt, tf_ratio)
            loss = path_criterion(pred, gt)
            if use_coverage:
                loss = loss + config.coverage_loss_weight * coverage_criterion(pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = np.mean(train_losses)

        # ── 验证 ──
        val_loss = train_loss  # 默认
        if (epoch + 1) % config.val_interval == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    images  = batch['image'].to(device)
                    gt      = batch['scanpath'].to(device)
                    salmaps = batch.get('saliency_map')
                    if salmaps is None:
                        continue
                    salmaps = salmaps.to(device)
                    pred = model(images, salmaps, None, 0.0)
                    val_losses.append(path_criterion(pred, gt).item())
            val_loss = np.mean(val_losses)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'variant': variant,
                    'model_state_dict': model.state_dict(),
                    'val_loss': val_loss,
                }, os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"  [Epoch {epoch+1}] train={train_loss:.4f}  val={val_loss:.4f}  ✓ saved")
            else:
                print(f"  [Epoch {epoch+1}] train={train_loss:.4f}  val={val_loss:.4f}")

        history.append({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss})
        scheduler.step()

    # 保存训练历史
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(log_dir, f'history_{ts}.json'), 'w') as f:
        json.dump({'variant': variant, 'history': history}, f, indent=2)

    print(f"\n[{variant}] run{run} Training done. Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=VARIANTS)
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()
    train(args.variant, args.run)
