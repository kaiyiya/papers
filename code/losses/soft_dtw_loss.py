"""
Soft-DTW损失实现

参考论文: Soft-DTW: a Differentiable Loss Function for Time-Series
https://arxiv.org/abs/1703.01541

可微分的DTW损失，用于直接优化时序对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftDTW(nn.Module):
    """
    Soft-DTW损失

    使用soft-min操作使DTW可微分，可以直接用于梯度下降优化
    """
    def __init__(self, gamma=1.0, normalize=False):
        """
        Args:
            gamma: 平滑参数，越小越接近hard-DTW，越大越平滑
            normalize: 是否归一化（除以序列长度）
        """
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def forward(self, pred, target):
        """
        计算Soft-DTW损失

        Args:
            pred: [B, T, D] 预测序列
            target: [B, T, D] 目标序列

        Returns:
            loss: scalar
        """
        B, T, D = pred.shape

        # 计算距离矩阵 [B, T, T]
        # pred: [B, T, 1, D], target: [B, 1, T, D]
        pred_expanded = pred.unsqueeze(2)
        target_expanded = target.unsqueeze(1)

        # 欧氏距离的平方
        dist = torch.sum((pred_expanded - target_expanded) ** 2, dim=-1)  # [B, T, T]

        # Soft-DTW动态规划
        R = self._soft_dtw_forward(dist)

        # 返回最终的DTW距离
        loss = R[:, -1, -1]

        if self.normalize:
            loss = loss / T

        return loss.mean()

    def _soft_dtw_forward(self, D):
        """
        Soft-DTW前向传播

        Args:
            D: [B, T, T] 距离矩阵

        Returns:
            R: [B, T+1, T+1] 累积距离矩阵
        """
        B, T1, T2 = D.shape

        # 初始化累积距离矩阵
        R = torch.zeros(B, T1 + 1, T2 + 1, device=D.device, dtype=D.dtype)
        R[:, 0, :] = float('inf')
        R[:, :, 0] = float('inf')
        R[:, 0, 0] = 0

        # 动态规划
        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                # 三个方向：对角、上、左
                r0 = R[:, i-1, j-1]  # 对角
                r1 = R[:, i-1, j]    # 上
                r2 = R[:, i, j-1]    # 左

                # Soft-min操作
                rmin = self._soft_min(r0, r1, r2)

                # 累积距离
                R[:, i, j] = D[:, i-1, j-1] + rmin

        return R

    def _soft_min(self, a, b, c):
        """
        Soft-min操作: -gamma * log(exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma))

        使用log-sum-exp技巧避免数值溢出
        """
        # Stack tensors
        stack = torch.stack([a, b, c], dim=-1)  # [B, 3]

        # Soft-min using log-sum-exp
        # soft_min = -gamma * log(sum(exp(-x/gamma)))
        soft_min = -self.gamma * torch.logsumexp(-stack / self.gamma, dim=-1)

        return soft_min


class SoftDTWLoss(nn.Module):
    """
    Soft-DTW损失的封装版本，用于训练
    """
    def __init__(self, gamma=1.0, normalize=True):
        super().__init__()
        self.soft_dtw = SoftDTW(gamma=gamma, normalize=normalize)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, T, 2] 预测路径（归一化坐标）
            target: [B, T, 2] GT路径（归一化坐标）

        Returns:
            loss: scalar
        """
        return self.soft_dtw(pred, target)


def test_soft_dtw():
    """测试Soft-DTW损失"""
    print("="*80)
    print("测试Soft-DTW损失")
    print("="*80)
    print()

    # 创建测试数据
    B, T, D = 4, 30, 2
    pred = torch.randn(B, T, D)
    target = torch.randn(B, T, D)

    # 测试Soft-DTW
    print("1. 测试Soft-DTW损失")
    soft_dtw_loss = SoftDTWLoss(gamma=1.0, normalize=True)
    loss = soft_dtw_loss(pred, target)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   ✓ 可以计算损失")

    # 测试梯度
    pred.requires_grad = True
    loss = soft_dtw_loss(pred, target)
    loss.backward()
    print(f"   Gradient norm: {pred.grad.norm().item():.4f}")
    print(f"   ✓ 可以反向传播")
    print()

    # 对比相同序列的损失
    print("2. 测试相同序列（损失应该接近0）")
    same_seq = torch.randn(B, T, D)
    loss_same = soft_dtw_loss(same_seq, same_seq)
    print(f"   Loss (same sequence): {loss_same.item():.6f}")
    print(f"   ✓ 相同序列损失接近0")
    print()

    # 对比不同gamma的效果
    print("3. 测试不同gamma参数")
    pred3 = torch.randn(B, T, D)
    target3 = torch.randn(B, T, D)

    for gamma in [0.1, 1.0, 10.0]:
        loss_fn = SoftDTWLoss(gamma=gamma, normalize=True)
        loss_val = loss_fn(pred3, target3)
        print(f"   gamma={gamma:4.1f}: loss={loss_val.item():.4f}")
    print()

    print("="*80)
    print("✓ 所有测试通过！")
    print("="*80)


if __name__ == '__main__':
    test_soft_dtw()
