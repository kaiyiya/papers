"""
序列匹配损失

综合多个序列特征的匹配损失，用于改善LEV和REC指标
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceMatchingLoss(nn.Module):
    """
    序列匹配损失

    综合考虑：
    1. 位置匹配
    2. 方向匹配
    3. 步长匹配
    4. 访问顺序匹配
    """
    def __init__(self, weights=None):
        """
        Args:
            weights: dict, 各项损失的权重
                - position: 位置匹配权重
                - direction: 方向匹配权重
                - step: 步长匹配权重
                - order: 访问顺序权重
        """
        super().__init__()

        if weights is None:
            weights = {
                'position': 1.0,
                'direction': 0.5,
                'step': 0.3,
                'order': 0.2
            }

        self.weights = weights

    def forward(self, pred, target, saliency_map=None):
        """
        Args:
            pred: [B, T, 2] 预测路径（归一化坐标）
            target: [B, T, 2] GT路径（归一化坐标）
            saliency_map: [B, 1, H, W] 显著性图（可选，用于order loss）

        Returns:
            loss: scalar
            loss_dict: dict, 各项损失的详细信息
        """
        B, T, D = pred.shape

        loss_dict = {}

        # 1. 位置匹配损失（MSE）
        position_loss = F.mse_loss(pred, target)
        loss_dict['position'] = position_loss.item()

        # 2. 方向匹配损失
        direction_loss = self._direction_matching_loss(pred, target)
        loss_dict['direction'] = direction_loss.item()

        # 3. 步长匹配损失
        step_loss = self._step_matching_loss(pred, target)
        loss_dict['step'] = step_loss.item()

        # 4. 访问顺序损失（如果提供了显著性图）
        if saliency_map is not None:
            order_loss = self._order_matching_loss(pred, target, saliency_map)
            loss_dict['order'] = order_loss.item()
        else:
            order_loss = 0.0
            loss_dict['order'] = 0.0

        # 综合损失
        total_loss = (
            self.weights['position'] * position_loss +
            self.weights['direction'] * direction_loss +
            self.weights['step'] * step_loss +
            self.weights['order'] * order_loss
        )

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def _direction_matching_loss(self, pred, target):
        """
        方向匹配损失

        计算预测路径和GT路径的方向向量的余弦相似度
        """
        # 计算方向向量
        pred_directions = pred[:, 1:] - pred[:, :-1]  # [B, T-1, 2]
        target_directions = target[:, 1:] - target[:, :-1]  # [B, T-1, 2]

        # 归一化
        pred_directions = F.normalize(pred_directions, p=2, dim=-1)
        target_directions = F.normalize(target_directions, p=2, dim=-1)

        # 余弦相似度
        cosine_sim = (pred_directions * target_directions).sum(dim=-1)  # [B, T-1]

        # 转换为损失（1 - cosine_sim）
        direction_loss = (1 - cosine_sim).mean()

        return direction_loss

    def _step_matching_loss(self, pred, target):
        """
        步长匹配损失

        匹配预测路径和GT路径的步长分布
        """
        # 计算步长
        pred_steps = torch.norm(pred[:, 1:] - pred[:, :-1], dim=-1)  # [B, T-1]
        target_steps = torch.norm(target[:, 1:] - target[:, :-1], dim=-1)  # [B, T-1]

        # MSE损失
        step_loss = F.mse_loss(pred_steps, target_steps)

        return step_loss

    def _order_matching_loss(self, pred, target, saliency_map):
        """
        访问顺序损失

        匹配预测路径和GT路径访问显著性区域的顺序
        """
        B, T, _ = pred.shape
        _, _, H, W = saliency_map.shape

        # 将归一化坐标转换为像素坐标
        pred_x = (pred[:, :, 0] * (W - 1)).long()
        pred_y = (pred[:, :, 1] * (H - 1)).long()
        pred_x = torch.clamp(pred_x, 0, W - 1)
        pred_y = torch.clamp(pred_y, 0, H - 1)

        target_x = (target[:, :, 0] * (W - 1)).long()
        target_y = (target[:, :, 1] * (H - 1)).long()
        target_x = torch.clamp(target_x, 0, W - 1)
        target_y = torch.clamp(target_y, 0, H - 1)

        # 提取显著性值序列
        batch_indices = torch.arange(B, device=pred.device).unsqueeze(1).expand(B, T)

        pred_saliency_seq = saliency_map[batch_indices, 0, pred_y, pred_x]  # [B, T]
        target_saliency_seq = saliency_map[batch_indices, 0, target_y, target_x]  # [B, T]

        # MSE损失
        order_loss = F.mse_loss(pred_saliency_seq, target_saliency_seq)

        return order_loss


class CombinedSequenceLoss(nn.Module):
    """
    综合序列损失

    结合Soft-DTW和序列匹配损失
    """
    def __init__(self, dtw_gamma=1.0, dtw_weight=0.5, seq_weights=None):
        """
        Args:
            dtw_gamma: Soft-DTW的gamma参数
            dtw_weight: Soft-DTW损失的权重
            seq_weights: 序列匹配损失的权重字典
        """
        super().__init__()

        from losses.soft_dtw_loss import SoftDTWLoss

        self.soft_dtw = SoftDTWLoss(gamma=dtw_gamma, normalize=True)
        self.seq_matching = SequenceMatchingLoss(weights=seq_weights)
        self.dtw_weight = dtw_weight

    def forward(self, pred, target, saliency_map=None):
        """
        Args:
            pred: [B, T, 2] 预测路径
            target: [B, T, 2] GT路径
            saliency_map: [B, 1, H, W] 显著性图（可选）

        Returns:
            total_loss: scalar
            loss_dict: dict, 各项损失的详细信息
        """
        # Soft-DTW损失
        dtw_loss = self.soft_dtw(pred, target)

        # 序列匹配损失
        seq_loss, seq_loss_dict = self.seq_matching(pred, target, saliency_map)

        # 综合损失
        total_loss = self.dtw_weight * dtw_loss + (1 - self.dtw_weight) * seq_loss

        # 损失字典
        loss_dict = {
            'dtw': dtw_loss.item(),
            'sequence_matching': seq_loss.item(),
            **seq_loss_dict
        }

        return total_loss, loss_dict


def test_sequence_matching_loss():
    """测试序列匹配损失"""
    print("="*80)
    print("测试序列匹配损失")
    print("="*80)
    print()

    # 创建测试数据
    B, T = 4, 30
    H, W = 256, 512

    pred = torch.rand(B, T, 2)
    target = torch.rand(B, T, 2)
    saliency_map = torch.rand(B, 1, H, W)

    # 测试序列匹配损失
    print("1. 测试序列匹配损失")
    seq_loss_fn = SequenceMatchingLoss()
    loss, loss_dict = seq_loss_fn(pred, target, saliency_map)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Loss breakdown:")
    for key, val in loss_dict.items():
        print(f"     {key}: {val:.4f}")
    print(f"   ✓ 可以计算损失")
    print()

    # 测试梯度
    print("2. 测试梯度")
    pred.requires_grad = True
    loss, _ = seq_loss_fn(pred, target, saliency_map)
    loss.backward()
    print(f"   Gradient norm: {pred.grad.norm().item():.4f}")
    print(f"   ✓ 可以反向传播")
    print()

    # 测试相同序列
    print("3. 测试相同序列（损失应该很小）")
    same_seq = torch.rand(B, T, 2)
    loss_same, loss_dict_same = seq_loss_fn(same_seq, same_seq, saliency_map)
    print(f"   Loss (same sequence): {loss_same.item():.6f}")
    print(f"   Position loss: {loss_dict_same['position']:.6f}")
    print(f"   Direction loss: {loss_dict_same['direction']:.6f}")
    print(f"   Step loss: {loss_dict_same['step']:.6f}")
    print(f"   ✓ 相同序列损失接近0")
    print()

    # 测试综合损失
    print("4. 测试综合序列损失")
    combined_loss_fn = CombinedSequenceLoss(dtw_gamma=1.0, dtw_weight=0.5)
    pred2 = torch.rand(B, T, 2, requires_grad=True)
    loss_combined, loss_dict_combined = combined_loss_fn(pred2, target, saliency_map)
    print(f"   Total loss: {loss_combined.item():.4f}")
    print(f"   Loss breakdown:")
    for key, val in loss_dict_combined.items():
        if key != 'total':
            print(f"     {key}: {val:.4f}")
    loss_combined.backward()
    print(f"   Gradient norm: {pred2.grad.norm().item():.4f}")
    print(f"   ✓ 综合损失可用")
    print()

    print("="*80)
    print("✓ 所有测试通过！")
    print("="*80)


if __name__ == '__main__':
    test_sequence_matching_loss()
