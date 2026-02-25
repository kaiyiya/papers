"""
数据加载器
"""
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class Salient360ScanpathDataset(Dataset):
    def __init__(self, data_path, split='train', seq_len=30, augment=True):
        print(f"Loading {split} data from {data_path}...")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        raw_data = data_dict[split]
        self.seq_len = seq_len
        self.augment = augment and (split == 'train')  # 仅在训练时增强

        # 方案3：展开数据集，每条路径作为独立样本
        # 这样解决了"同一图像每次对应不同GT"的问题
        print(f"Expanding dataset: each scanpath becomes an independent sample...")
        self.samples = []
        for key in raw_data.keys():
            sample = raw_data[key]
            scanpaths = sample['scanpaths_2d']

            # 处理 object 类型
            if scanpaths.dtype == np.object_:
                scanpaths_list = []
                for i in range(len(scanpaths)):
                    sp = np.array(scanpaths[i], dtype=np.float32)
                    if sp.shape[1] == 3:
                        sp = sp[:, :2]
                    scanpaths_list.append(sp)
                scanpaths = np.array(scanpaths_list, dtype=np.float32)

            # 确保是 2D 坐标
            if scanpaths.shape[2] == 3:
                scanpaths = scanpaths[:, :, :2]

            # 为每条路径创建一个独立样本
            for scanpath_idx in range(len(scanpaths)):
                self.samples.append({
                    'key': key,
                    'image': sample['image'],
                    'saliency_map': sample.get('saliency_map') or sample.get('salmap'),
                    'scanpath': scanpaths[scanpath_idx].copy(),
                    'scanpath_idx': scanpath_idx
                })

        print(f"Loaded {len(self.samples)} samples from {len(raw_data)} images (augmentation: {self.augment})")
        print(f"  Average {len(self.samples)/len(raw_data):.1f} scanpaths per image")

    def __len__(self):
        return len(self.samples)

    def normalize_scanpath(self, scanpath):
        """改进的坐标归一化 - 专门处理经纬度坐标"""
        # 检查是否已经归一化
        if scanpath.min() >= 0.0 and scanpath.max() <= 1.0:
            return scanpath

        # 判断是否为经纬度坐标
        # 经纬度特征：X在[-180,180]，Y在[-90,90]
        x_min, x_max = scanpath[:, 0].min(), scanpath[:, 0].max()
        y_min, y_max = scanpath[:, 1].min(), scanpath[:, 1].max()

        # 更可靠的判断：检查是否在经纬度范围内（允许一定容差）
        # 经度范围：[-180, 180]，纬度范围：[-90, 90]
        if -180.1 <= x_min and x_max <= 180.1 and -90.1 <= y_min and y_max <= 90.1:
            # 经纬度归一化
            scanpath[:, 0] = (scanpath[:, 0] + 180.0) / 360.0
            scanpath[:, 1] = (scanpath[:, 1] + 90.0) / 180.0
        else:
            # 其他坐标系统（像素等）
            scanpath_min = scanpath.min(axis=0, keepdims=True)
            scanpath_max = scanpath.max(axis=0, keepdims=True)
            scanpath_range = scanpath_max - scanpath_min
            scanpath_range[scanpath_range < 1e-8] = 1.0
            scanpath = (scanpath - scanpath_min) / scanpath_range

        # 验证归一化结果
        scanpath = np.clip(scanpath, 0.0, 1.0)
        return scanpath

    def __getitem__(self, idx):
        sample = self.samples[idx]
        key = sample['key']

        # 加载图像
        img = sample['image']
        if isinstance(img, torch.Tensor):
            image = img.float()
        else:
            image = torch.from_numpy(img).float() if isinstance(img, np.ndarray) else torch.tensor(img, dtype=torch.float32)

        # 加载显著性图（如果存在）
        saliency_map = None
        sal = sample['saliency_map']
        if sal is not None:
            if isinstance(sal, torch.Tensor):
                saliency_map = sal.float()
            else:
                saliency_map = torch.from_numpy(sal).float() if isinstance(sal, np.ndarray) else torch.tensor(sal, dtype=torch.float32)
            # 确保是(1, H, W)格式
            if saliency_map.ndim == 2:
                saliency_map = saliency_map.unsqueeze(0)

            # 归一化显著性图到[0, 1]范围
            sal_min = saliency_map.min()
            sal_max = saliency_map.max()
            if sal_max > sal_min:
                saliency_map = (saliency_map - sal_min) / (sal_max - sal_min)
            else:
                # 如果显著性图是常数，设为均匀分布
                saliency_map = torch.ones_like(saliency_map) * 0.5

        # 使用预先分配的固定路径（不再随机选择）
        scanpath = sample['scanpath'].copy()

        # 只取前2维（经纬度），忽略第3维（时间戳等）
        if scanpath.shape[1] > 2:
            scanpath = scanpath[:, :2]

        # 归一化坐标到[0, 1]范围（改进的归一化逻辑）
        scanpath = self.normalize_scanpath(scanpath)

        scanpath = torch.from_numpy(scanpath).float()

        # 数据增强（仅训练时）
        if self.augment:
            if saliency_map is not None:
                image, scanpath, saliency_map = self._apply_augmentation_with_saliency(image, scanpath, saliency_map)
            else:
                image, scanpath = self._apply_augmentation(image, scanpath)

        if scanpath.shape[0] > self.seq_len:
            scanpath = scanpath[:self.seq_len]
        elif scanpath.shape[0] < self.seq_len:
            pad_length = self.seq_len - scanpath.shape[0]
            last_point = scanpath[-1:]
            scanpath = torch.cat([scanpath, last_point.repeat(pad_length, 1)], dim=0)

        # 确保在[0,1]范围内（安全边界）
        scanpath = torch.clamp(scanpath, 0.0, 1.0)

        # 验证数据完整性
        image, scanpath, saliency_map = self.validate_data(image, scanpath, saliency_map, key)

        # 返回数据
        result = {'image': image, 'scanpath': scanpath, 'key': key}
        if saliency_map is not None:
            result['saliency_map'] = saliency_map
        return result

    def validate_data(self, image, scanpath, saliency_map, key):
        """验证数据完整性"""
        # 检查NaN/Inf
        if torch.isnan(image).any() or torch.isinf(image).any():
            raise ValueError(f"图像包含NaN/Inf: {key}")
        if torch.isnan(scanpath).any() or torch.isinf(scanpath).any():
            raise ValueError(f"扫描路径包含NaN/Inf: {key}")

        # 检查坐标范围
        if (scanpath < 0).any() or (scanpath > 1).any():
            print(f"⚠️ 警告：扫描路径超出[0,1]范围: {key}")
            print(f"  范围: x=[{scanpath[:, 0].min():.4f}, {scanpath[:, 0].max():.4f}], "
                  f"y=[{scanpath[:, 1].min():.4f}, {scanpath[:, 1].max():.4f}]")
            scanpath = torch.clamp(scanpath, 0.0, 1.0)

        # 检查路径长度
        if scanpath.shape[0] == 0:
            raise ValueError(f"扫描路径为空: {key}")

        return image, scanpath, saliency_map

    def _apply_augmentation_with_saliency(self, image, scanpath, saliency_map):
        """
        应用数据增强（包含显著性图）
        对于360度全景图，支持：
        1. 水平翻转（左右镜像）
        2. 水平循环移位（利用360度连续性）
        """
        # 50%概率水平翻转
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])  # 水平翻转图像 (C, H, W)
            saliency_map = torch.flip(saliency_map, dims=[2])  # 水平翻转显著性图
            scanpath[:, 0] = 1.0 - scanpath[:, 0]  # X坐标镜像

        # 50%概率水平循环移位（利用360度连续性）
        if torch.rand(1).item() < 0.5:
            shift_ratio = torch.rand(1).item()  # 随机移位比例[0, 1]

            # 图像循环移位
            _, _, W = image.shape
            shift_pixels = int(W * shift_ratio)
            image = torch.roll(image, shifts=shift_pixels, dims=2)
            saliency_map = torch.roll(saliency_map, shifts=shift_pixels, dims=2)

            # 扫描路径X坐标相应移位
            scanpath[:, 0] = (scanpath[:, 0] + shift_ratio) % 1.0

        return image, scanpath, saliency_map

    def _apply_augmentation(self, image, scanpath):
        """
        应用数据增强
        对于360度全景图，支持：
        1. 水平翻转（左右镜像）
        2. 水平循环移位（利用360度连续性）
        """
        # 50%概率水平翻转
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])  # 水平翻转图像 (C, H, W)
            scanpath[:, 0] = 1.0 - scanpath[:, 0]  # X坐标镜像

        # 50%概率水平循环移位（利用360度连续性）
        if torch.rand(1).item() < 0.5:
            shift_ratio = torch.rand(1).item()  # 随机移位比例[0, 1]

            # 图像循环移位
            _, _, W = image.shape
            shift_pixels = int(W * shift_ratio)
            image = torch.roll(image, shifts=shift_pixels, dims=2)

            # 扫描路径X坐标相应移位
            scanpath[:, 0] = (scanpath[:, 0] + shift_ratio) % 1.0

        return image, scanpath

def create_dataloaders(config):
    train_dataset = Salient360ScanpathDataset(
        config.processed_data_path,
        'train',
        config.seq_len,
        augment=config.use_augmentation
    )
    test_dataset = Salient360ScanpathDataset(
        config.processed_data_path,
        'test',
        config.seq_len,
        augment=False  # 测试时不增强
    )

    # 获取prefetch_factor（如果配置中有）
    prefetch_factor = getattr(config, 'prefetch_factor', 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
        prefetch_factor=prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        prefetch_factor=prefetch_factor if config.num_workers > 0 else None,
        persistent_workers=True if config.num_workers > 0 else False
    )

    return train_loader, test_loader
