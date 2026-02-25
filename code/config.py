"""
配置文件
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
    num_workers = 8
    pin_memory = True
    prefetch_factor = 4

    # 日志
    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    save_interval = 10
    val_interval = 2
