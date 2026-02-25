"""
消融实验统一配置
"""
import torch

class AblationConfig:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    processed_data_path = 'datasets/Salient360.pkl'
    seq_len = 25

    d_model = 256
    d_state = 256

    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    weight_decay = 1e-4

    use_augmentation = True
    num_workers = 8
    pin_memory = True
    prefetch_factor = 4

    save_interval = 50   # 只保存最终模型，节省空间
    val_interval = 2

    # V7.2参数（full model使用）
    phase1_ratio = 0.4
    phase1_exploration = 0.60
    phase2_exploration = 0.30
    coverage_loss_weight = 0.05
    x_coverage_weight = 2.0
    y_coverage_weight = 1.0
