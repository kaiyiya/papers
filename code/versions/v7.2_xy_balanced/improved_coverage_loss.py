"""
V7.2改进的覆盖率损失函数

同时优化X和Y方向，X方向权重更高（因为360度全景）
"""
import torch
import torch.nn as nn


class ImprovedCoverageLoss(nn.Module):
    """改进的覆盖率损失 - X和Y方向都优化，X权重更高"""
    def __init__(self, x_weight=2.0, y_weight=1.0):
        super().__init__()
        self.x_weight = x_weight  # X方向权重（360度，应该更高）
        self.y_weight = y_weight  # Y方向权重

    def forward(self, scanpaths):
        """
        Args:
            scanpaths: [B, T, 2] 归一化坐标
        Returns:
            loss: 标量损失值
        """
        B, T, _ = scanpaths.shape

        # X方向（水平）- 360度全景，应该大范围探索
        x_coords = scanpaths[:, :, 0]  # [B, T]
        x_std = x_coords.std(dim=1).mean()
        x_range = (x_coords.max(dim=1)[0] - x_coords.min(dim=1)[0]).mean()
        x_loss = -(x_std + x_range) / 2.0  # 负号：越大越好

        # Y方向（垂直）- 适中即可
        y_coords = scanpaths[:, :, 1]  # [B, T]
        y_std = y_coords.std(dim=1).mean()
        y_range = (y_coords.max(dim=1)[0] - y_coords.min(dim=1)[0]).mean()
        y_loss = -(y_std + y_range) / 2.0  # 负号：越大越好

        # 总损失：X方向权重更高
        total_loss = self.x_weight * x_loss + self.y_weight * y_loss

        return total_loss

    def get_stats(self, scanpaths):
        """获取统计信息（用于监控）"""
        x_coords = scanpaths[:, :, 0]
        y_coords = scanpaths[:, :, 1]

        x_std = x_coords.std(dim=1).mean().item()
        x_range = (x_coords.max(dim=1)[0] - x_coords.min(dim=1)[0]).mean().item()
        y_std = y_coords.std(dim=1).mean().item()
        y_range = (y_coords.max(dim=1)[0] - y_coords.min(dim=1)[0]).mean().item()

        return {
            'x_std': x_std,
            'x_range': x_range,
            'x_coverage': x_range * 100,  # 转换为百分比
            'y_std': y_std,
            'y_range': y_range,
            'y_coverage': y_range * 100,  # 转换为百分比
            'xy_ratio': (x_range / y_range) if y_range > 0 else 0  # X/Y比例
        }
