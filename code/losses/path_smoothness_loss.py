"""
路径平滑损失 - 减少不自然的跳跃

通过惩罚过大的加速度变化，使路径更加平滑自然
"""
import torch
import torch.nn as nn


class PathSmoothnessLoss(nn.Module):
    """
    路径平滑损失

    计算路径的加速度（二阶差分），惩罚过大的加速度变化
    这有助于生成更自然、更平滑的扫视路径
    """

    def __init__(self):
        super().__init__()

    def forward(self, scanpath):
        """
        Args:
            scanpath: [B, T, 2] 扫视路径（归一化坐标）

        Returns:
            smoothness_loss: 标量，路径平滑损失
        """
        # 计算相邻点之间的距离（一阶差分）
        # step_distances: [B, T-1]
        step_distances = torch.norm(
            scanpath[:, 1:] - scanpath[:, :-1],
            dim=-1
        )

        # 计算加速度（二阶差分）
        # acceleration: [B, T-2]
        acceleration = torch.abs(
            step_distances[:, 1:] - step_distances[:, :-1]
        )

        # 惩罚过大的加速度变化
        smoothness_loss = acceleration.mean()

        return smoothness_loss


class PathVelocityConsistencyLoss(nn.Module):
    """
    路径速度一致性损失（可选）

    惩罚速度的剧烈变化，使路径速度更加一致
    """

    def __init__(self):
        super().__init__()

    def forward(self, scanpath):
        """
        Args:
            scanpath: [B, T, 2] 扫视路径

        Returns:
            velocity_loss: 标量，速度一致性损失
        """
        # 计算速度（一阶差分）
        velocities = scanpath[:, 1:] - scanpath[:, :-1]

        # 计算速度的标准差
        velocity_std = velocities.std(dim=1).mean()

        return velocity_std
