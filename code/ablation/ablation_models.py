"""
消融实验模型定义
包含5个变体：
  full            - 完整V7.2模型（基准）
  no_coord_att    - 去掉CoordAttention，用普通Conv替代
  no_hierarchical - 去掉分层采样，用固定探索率
  no_coverage_loss- 去掉Coverage Loss
  lstm_baseline   - 用LSTM替代Mamba
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ─── 共享模块 ─────────────────────────────────────────────────────────────────

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6


class CoordAtt(nn.Module):
    """坐标注意力"""
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, 1)
        self.bn1   = nn.BatchNorm2d(mip)
        self.relu  = nn.ReLU()
        self.conv_h = nn.Conv2d(mip, oup, 1)
        self.conv_w = nn.Conv2d(mip, oup, 1)

    def forward(self, x):
        identity = x
        B, C, H, W = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.relu(self.bn1(self.conv1(y)))
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        return identity * self.conv_h(x_h).sigmoid() * self.conv_w(x_w).sigmoid()


# ─── Glance网络（两个版本） ───────────────────────────────────────────────────

class GlanceNetwork(nn.Module):
    """带CoordAttention的全局特征提取"""
    def __init__(self, d_model, use_coord_att=True):
        super().__init__()
        layers = [
            nn.Conv2d(3, 64, 7, stride=2, padding=3), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        ]
        if use_coord_att:
            layers.append(CoordAtt(128, 128))
        layers += [
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        ]
        if use_coord_att:
            layers.append(CoordAtt(256, 256))
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(256, d_model)

    def forward(self, x):
        return self.fc(self.encoder(x).view(x.size(0), -1))


class FocusNetwork(nn.Module):
    """带CoordAttention的局部特征提取"""
    def __init__(self, d_model, use_coord_att=True):
        super().__init__()
        layers = [
            nn.Conv2d(3, 64, 5, padding=2), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        ]
        if use_coord_att:
            layers.append(CoordAtt(128, 128))
        layers.append(nn.AdaptiveAvgPool2d(4))
        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(128 * 16, d_model)

    def extract_patch(self, image, position):
        B, _, H, W = image.shape
        cx = (position[:, 0] * W).long()
        cy = (position[:, 1] * H).long()
        patch_size = min(H, W) // 4
        patches = []
        for b in range(B):
            x1 = max(0, cx[b].item() - patch_size // 2)
            x2 = min(W, x1 + patch_size)
            y1 = max(0, cy[b].item() - patch_size // 2)
            y2 = min(H, y1 + patch_size)
            patch = image[b:b+1, :, y1:y2, x1:x2]
            patch = F.interpolate(patch, size=(64, 64), mode='bilinear', align_corners=False)
            patches.append(patch)
        return torch.cat(patches, dim=0)

    def forward(self, image, position):
        patch = self.extract_patch(image, position)
        return self.fc(self.encoder(patch).view(patch.size(0), -1))


# ─── 采样器（两个版本） ───────────────────────────────────────────────────────

class HierarchicalSampler(nn.Module):
    """分层采样（V7.2完整版）"""
    def __init__(self, temperature=0.12, max_step_size=0.18, momentum=0.45):
        super().__init__()
        self.temperature = temperature
        self.max_step_size = max_step_size
        self.momentum = momentum

    def get_exploration_rate(self, step, seq_len):
        if step < seq_len * 0.4:
            return 0.60   # 第一阶段
        return 0.30       # 第二阶段

    def sample_from_saliency(self, saliency_map, exploration_rate):
        B, _, H, W = saliency_map.shape
        if torch.rand(1).item() < exploration_rate:
            return torch.rand(B, 2, device=saliency_map.device)
        saliency_flat = saliency_map.view(B, -1)
        probs = F.softmax(saliency_flat / self.temperature, dim=-1)
        indices = torch.multinomial(probs, 1).squeeze(-1)
        y = (indices // W).float() / max(H - 1, 1)
        x = (indices % W).float() / max(W - 1, 1)
        return torch.stack([x, y], dim=-1)

    def sample(self, saliency_map, prev_pos, prev_direction, step, seq_len):
        rate = self.get_exploration_rate(step, seq_len)
        base_pos = self.sample_from_saliency(saliency_map, rate)
        if prev_direction is not None and torch.rand(1).item() > rate:
            momentum_pos = torch.clamp(prev_pos + prev_direction * self.max_step_size, 0, 1)
            base_pos = (1 - self.momentum) * base_pos + self.momentum * momentum_pos
        return base_pos

    def apply_step_constraint(self, current_pos, prev_pos):
        step_vec = current_pos - prev_pos
        step_size = torch.norm(step_vec, dim=-1, keepdim=True)
        scale = torch.clamp(step_size / self.max_step_size, min=1.0)
        return prev_pos + step_vec / scale


class FixedRateSampler(nn.Module):
    """固定探索率采样（消融：去掉分层采样）"""
    def __init__(self, temperature=0.12, max_step_size=0.18, momentum=0.45, exploration_rate=0.45):
        super().__init__()
        self.temperature = temperature
        self.max_step_size = max_step_size
        self.momentum = momentum
        self.exploration_rate = exploration_rate  # 固定探索率（取两阶段均值）

    def sample_from_saliency(self, saliency_map):
        B, _, H, W = saliency_map.shape
        if torch.rand(1).item() < self.exploration_rate:
            return torch.rand(B, 2, device=saliency_map.device)
        saliency_flat = saliency_map.view(B, -1)
        probs = F.softmax(saliency_flat / self.temperature, dim=-1)
        indices = torch.multinomial(probs, 1).squeeze(-1)
        y = (indices // W).float() / max(H - 1, 1)
        x = (indices % W).float() / max(W - 1, 1)
        return torch.stack([x, y], dim=-1)

    def sample(self, saliency_map, prev_pos, prev_direction, step, seq_len):
        base_pos = self.sample_from_saliency(saliency_map)
        if prev_direction is not None and torch.rand(1).item() > self.exploration_rate:
            momentum_pos = torch.clamp(prev_pos + prev_direction * self.max_step_size, 0, 1)
            base_pos = (1 - self.momentum) * base_pos + self.momentum * momentum_pos
        return base_pos

    def apply_step_constraint(self, current_pos, prev_pos):
        step_vec = current_pos - prev_pos
        step_size = torch.norm(step_vec, dim=-1, keepdim=True)
        scale = torch.clamp(step_size / self.max_step_size, min=1.0)
        return prev_pos + step_vec / scale


# ─── 序列建模（Mamba vs LSTM） ────────────────────────────────────────────────

class LSTMSequenceModel(nn.Module):
    """LSTM序列建模（消融：替代Mamba）"""
    def __init__(self, d_model):
        super().__init__()
        self.lstm = nn.LSTM(d_model, d_model, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


# ─── 统一主模型 ───────────────────────────────────────────────────────────────

class AblationModel(nn.Module):
    """
    消融实验统一模型，通过variant参数控制：
      'full'             - 完整V7.2
      'no_coord_att'     - 去掉CoordAttention
      'no_hierarchical'  - 固定探索率
      'no_coverage_loss' - 同full，Coverage Loss在训练脚本中控制
      'lstm_baseline'    - LSTM替代Mamba
    """
    def __init__(self, config, variant='full'):
        super().__init__()
        self.seq_len = config.seq_len
        self.variant = variant

        use_coord_att = (variant != 'no_coord_att')
        use_hierarchical = (variant != 'no_hierarchical')

        self.glance = GlanceNetwork(config.d_model, use_coord_att=use_coord_att)
        self.focus  = FocusNetwork(config.d_model,  use_coord_att=use_coord_att)

        if use_hierarchical:
            self.sampler = HierarchicalSampler()
        else:
            self.sampler = FixedRateSampler()

        if variant == 'lstm_baseline':
            self.seq_model = LSTMSequenceModel(config.d_model)
        else:
            assert HAS_MAMBA, "mamba_ssm not installed"
            self.seq_model = Mamba(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=4,
                expand=2
            )

        self.pos_encoder = nn.Sequential(
            nn.Linear(2, config.d_model // 2),
            nn.ReLU(),
            nn.Linear(config.d_model // 2, config.d_model)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model // 2, 2),
            nn.Sigmoid()
        )

    def forward(self, images, saliency_map, gt_scanpaths=None, teacher_forcing_ratio=0.0):
        B, _, H, W = saliency_map.shape
        global_feat = self.glance(images)

        max_idx = saliency_map.view(B, -1).argmax(dim=1)
        y_start = (max_idx // W).float() / max(H - 1, 1)
        x_start = (max_idx % W).float() / max(W - 1, 1)
        prev_pos = torch.stack([x_start, y_start], dim=-1)
        prev_direction = None

        all_features = []

        for t in range(self.seq_len):
            use_tf = gt_scanpaths is not None and torch.rand(1).item() < teacher_forcing_ratio
            if use_tf:
                current_pos = gt_scanpaths[:, t]
            else:
                if t == 0:
                    current_pos = prev_pos
                else:
                    current_pos = self.sampler.sample(
                        saliency_map, prev_pos, prev_direction, t, self.seq_len
                    )
                    current_pos = self.sampler.apply_step_constraint(current_pos, prev_pos)

            local_feat = self.focus(images, current_pos)
            pos_feat   = self.pos_encoder(current_pos)
            all_features.append(global_feat + local_feat + pos_feat)

            if t > 0:
                direction = current_pos - prev_pos
                norm = torch.norm(direction, dim=-1, keepdim=True)
                prev_direction = direction / (norm + 1e-8)
            prev_pos = current_pos

        features = torch.stack(all_features, dim=1)
        out = self.seq_model(features)
        return self.decoder(out)


def create_model(config, variant='full'):
    return AblationModel(config, variant=variant)
