"""
V7.1: 平衡优化模型 - 分层采样 + 覆盖率损失

核心改进:
1. 分层采样策略: 前40%步数高探索(70%)，后60%步数低探索(25%)
2. 覆盖率损失函数: 显式优化Y方向覆盖率和标准差
3. 保留V7优势: CoordAttention, 自适应探索, Y方向增强
4. 目标: 在保持LEV/REC优势的同时，大幅提升覆盖率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    """坐标注意力 - 增强Y方向感知"""
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()

        # 分别处理H和W方向
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)

        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h
        return out


class HighExplorationSampler(nn.Module):
    """高探索率采样器 - V7核心改进"""
    def __init__(self,
                 temperature=0.12,
                 max_step_size=0.18,
                 momentum=0.45,
                 base_exploration_rate=0.35,  # 基础探索率提升到35%
                 y_boost_factor=2.0):  # Y方向增强因子
        super().__init__()
        self.temperature = temperature
        self.max_step_size = max_step_size
        self.momentum = momentum
        self.base_exploration_rate = base_exploration_rate
        self.y_boost_factor = y_boost_factor

    def get_adaptive_exploration_rate(self, step, seq_len, visited_positions):
        """V7.1改进: 分层采样策略"""
        # 分两个阶段
        if step < seq_len * 0.4:  # 前40%步数 (前10步)
            # 第一阶段: 粗略扫描，建立覆盖
            scheduled_rate = 0.70  # 70%探索率
        else:  # 后60%步数 (后15步)
            # 第二阶段: 精细注视，保证质量
            scheduled_rate = 0.25  # 25%探索率

        # 覆盖率感知: 如果访问位置过于集中，提升探索率
        if len(visited_positions) > 3:
            positions = torch.stack(visited_positions[-5:], dim=0)  # 最近5个位置
            y_std = positions[:, :, 1].std(dim=0).mean().item()  # Y方向标准差

            # 如果Y方向标准差小于0.1，说明过于集中，提升探索率
            if y_std < 0.1:
                coverage_boost = 0.15
            elif y_std < 0.15:
                coverage_boost = 0.10
            else:
                coverage_boost = 0.0

            return min(0.80, scheduled_rate + coverage_boost)  # 最高80%

        return scheduled_rate

    def sample_from_saliency(self, saliency_map, exploration_rate, y_boost=False):
        """从显著性图采样位置 - 支持Y方向增强"""
        B, _, H, W = saliency_map.shape

        if torch.rand(1).item() < exploration_rate:
            # 探索模式：均匀随机采样
            x = torch.rand(B, device=saliency_map.device)

            if y_boost:
                # Y方向增强: 使用更宽的分布
                y = torch.rand(B, device=saliency_map.device)
                # 将Y值推向边缘 (0.0-0.3 或 0.7-1.0)
                mask = torch.rand(B, device=saliency_map.device) < 0.5
                y = torch.where(mask, y * 0.3, 0.7 + y * 0.3)
            else:
                y = torch.rand(B, device=saliency_map.device)

            return torch.stack([x, y], dim=-1)

        # 利用模式：显著性引导
        saliency_flat = saliency_map.view(B, -1)
        probs = F.softmax(saliency_flat / self.temperature, dim=-1)
        indices = torch.multinomial(probs, num_samples=1).squeeze(-1)

        y = (indices // W).float() / max(H - 1, 1)
        x = (indices % W).float() / max(W - 1, 1)

        return torch.stack([x, y], dim=-1)

    def sample_with_momentum(self, saliency_map, prev_pos, prev_direction, step, seq_len, visited_positions):
        """带动量的采样 - 自适应探索"""
        # 获取自适应探索率
        exploration_rate = self.get_adaptive_exploration_rate(step, seq_len, visited_positions)

        # Y方向增强: 每3步强制一次Y方向探索
        y_boost = (step % 3 == 0) and (step > 0)

        # 基础显著性采样（带自适应探索）
        base_pos = self.sample_from_saliency(saliency_map, exploration_rate, y_boost=y_boost)

        if prev_direction is not None and torch.rand(1).item() > exploration_rate:
            # 只在非探索模式下使用动量
            momentum_pos = prev_pos + prev_direction * self.max_step_size
            momentum_pos = torch.clamp(momentum_pos, 0.0, 1.0)
            sampled_pos = (1 - self.momentum) * base_pos + self.momentum * momentum_pos
        else:
            sampled_pos = base_pos

        return sampled_pos

    def apply_step_constraint(self, current_pos, prev_pos):
        """限制步长"""
        step_vector = current_pos - prev_pos
        step_size = torch.norm(step_vector, dim=-1, keepdim=True)
        scale = torch.clamp(step_size / self.max_step_size, min=1.0)
        return prev_pos + step_vector / scale


class EnhancedGlanceNetwork(nn.Module):
    """增强的全局特征提取 - 添加CoordAttention"""
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CoordAtt(128, 128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            CoordAtt(256, 256),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(256, d_model)

    def forward(self, x):
        feat = self.encoder(x).view(x.size(0), -1)
        return self.fc(feat)


class EnhancedFocusNetwork(nn.Module):
    """增强的局部特征提取 - 添加CoordAttention"""
    def __init__(self, d_model):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            CoordAtt(128, 128),
            nn.AdaptiveAvgPool2d(4)
        )
        self.fc = nn.Linear(128 * 16, d_model)

    def extract_patch(self, image, position):
        """提取局部patch"""
        B, _, H, W = image.shape
        cx = (position[:, 0] * W).long()
        cy = (position[:, 1] * H).long()

        patches = []
        patch_size = min(H, W) // 4

        for b in range(B):
            x1 = max(0, cx[b] - patch_size // 2)
            x2 = min(W, x1 + patch_size)
            y1 = max(0, cy[b] - patch_size // 2)
            y2 = min(H, y1 + patch_size)

            patch = image[b:b+1, :, y1:y2, x1:x2]
            patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
            patches.append(patch)

        return torch.cat(patches, dim=0)

    def forward(self, image, position):
        patch = self.extract_patch(image, position)
        feat = self.encoder(patch).view(patch.size(0), -1)
        return self.fc(feat)


class V71BalancedModel(nn.Module):
    """V7.1: 平衡优化模型"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = config.seq_len

        # 1. 高探索率采样器 (V7.1改进)
        self.sampler = HighExplorationSampler()

        # 2. 增强的Glance - 全局上下文 + CoordAttention
        self.glance = EnhancedGlanceNetwork(config.d_model)

        # 3. 增强的Focus - 局部细节 + CoordAttention
        self.focus = EnhancedFocusNetwork(config.d_model)

        # 4. Mamba - 序列建模
        self.mamba = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=4,
            expand=2
        )

        # 5. 位置编码
        self.pos_encoder = nn.Sequential(
            nn.Linear(2, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )

        # 6. 位置解码器
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, images, saliency_map, gt_scanpaths=None, teacher_forcing_ratio=0.0):
        """
        V7.1改进流程：
        1. 分层采样: 前40%步数70%探索，后60%步数25%探索
        2. 覆盖率感知: 实时监控Y标准差，动态调整
        3. Y方向增强: 每3步强制探索
        4. 保持V7优势: CoordAttention, momentum, 平滑度
        """
        B, _, H, W = saliency_map.shape

        # 全局特征（带CoordAttention）
        global_feat = self.glance(images)

        # 初始位置：显著性最大值
        max_idx = saliency_map.view(B, -1).argmax(dim=1)
        y_start = (max_idx // W).float() / max(H - 1, 1)
        x_start = (max_idx % W).float() / max(W - 1, 1)
        prev_pos = torch.stack([x_start, y_start], dim=-1)
        prev_direction = None

        all_features = []
        all_positions = []
        visited_positions = []  # 用于覆盖率感知

        for t in range(self.seq_len):
            # Teacher forcing
            use_tf = gt_scanpaths is not None and torch.rand(1).item() < teacher_forcing_ratio

            if use_tf:
                current_pos = gt_scanpaths[:, t]
            else:
                if t == 0:
                    current_pos = prev_pos
                else:
                    # V7.1高探索率采样（分层+自适应+Y方向增强）
                    current_pos = self.sampler.sample_with_momentum(
                        saliency_map, prev_pos, prev_direction, t, self.seq_len, visited_positions
                    )
                    current_pos = self.sampler.apply_step_constraint(current_pos, prev_pos)

            # 记录访问位置（用于覆盖率感知）
            visited_positions.append(current_pos)

            # 提取局部特征（带CoordAttention）
            local_feat = self.focus(images, current_pos)

            # 位置编码
            pos_feat = self.pos_encoder(current_pos)

            # 融合特征
            combined = global_feat + local_feat + pos_feat
            all_features.append(combined)
            all_positions.append(current_pos)

            # 更新方向
            if t > 0:
                direction = current_pos - prev_pos
                norm = torch.norm(direction, dim=-1, keepdim=True)
                prev_direction = direction / (norm + 1e-8)

            prev_pos = current_pos

        # Mamba序列建模
        features = torch.stack(all_features, dim=1)
        mamba_out = self.mamba(features)

        # 解码
        scanpaths = self.decoder(mamba_out)

        return scanpaths


def create_model(config):
    return V71BalancedModel(config)
