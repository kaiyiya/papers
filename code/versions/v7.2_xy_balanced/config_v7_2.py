"""
V7.2配置文件 - X-Y平衡优化模型
改进：同时优化X和Y方向，X方向权重更高
"""
import torch

class Config:
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据
    processed_data_path = 'datasets/Salient360.pkl'
    seq_len = 25

    # 模型
    d_model = 256
    d_state = 256

    # 训练
    batch_size = 128
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4

    # 数据增强
    use_augmentation = True

    # 数据加载
    num_workers = 16
    pin_memory = True
    prefetch_factor = 4

    # 日志
    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    save_interval = 5  # 每5个epoch保存一次（更密集）
    val_interval = 2

    # V7.2特定参数
    # 分层采样（降低探索率）
    phase1_ratio = 0.4  # 前40%步数为第一阶段
    phase1_exploration = 0.60  # 第一阶段探索率60%（V7.1是70%）
    phase2_exploration = 0.30  # 第二阶段探索率30%（V7.1是25%）

    # 改进的覆盖率损失
    coverage_loss_weight = 0.05  # 降低权重（V7.1是0.1）
    x_coverage_weight = 2.0  # X方向权重（360度全景，更重要）
    y_coverage_weight = 1.0  # Y方向权重

    # 早停策略
    early_stop_x_coverage = 0.50  # X覆盖率低于50%时警告
    early_stop_y_coverage = 0.80  # Y覆盖率高于80%时警告
