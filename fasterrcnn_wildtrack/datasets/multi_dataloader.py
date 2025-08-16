import torch
import logging
import numpy as np
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader as TorchDataLoader
import sys
import os
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)
paths_to_add = [project_root, current_dir]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
try:
    from wildtrack2_datasets import WildtrackDataset
    from mot_datasets import MOTDataset
except ImportError as e:
    print(f"导入数据集模块失败: {e}")
    print("请确保 wildtrack2_datasets.py 和 mot_datasets.py 文件存在")
try:
    sys.path.insert(0, os.path.join(project_root, 'utils'))
    from utils import collate_fn
except ImportError:
    print("警告: 无法导入 collate_fn，将使用默认的 collate 函数")
    collate_fn = None

@dataclass
class DatasetStage(Enum):
    """数据集阶段枚举"""
    WILDTRACK = "Wildtrack2"
    MOT = "MOT"

@dataclass
class DataStageConfig:
    """单个数据阶段的配置"""
    stage: DatasetStage
    batch_size: int
    data_aug_conf: Dict
    dataset_params: Dict
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle_train: bool = True
    stage_name: str = ""


@dataclass
class MultiDatasetConfig:
    """多数据集配置"""
    stages: List[DataStageConfig]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    global_seed: int = 42

class TransformSubset(torch.utils.data.Subset):
    """带变换的数据集子集"""
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        try:
            img, target, img_name, mask = self.dataset[self.indices[idx]]
            if self.transform is not None:
                img, target, mask = self.transform(img, target, mask)
            return img, target, img_name, mask
        except Exception as e:
            logging.error(f"Error loading image at index {idx}: {str(e)}")
            raise

def get_augmentation_transforms(train=True, data_aug_conf=None):
    """
    获取数据增强变换列表
    
    Args:
        train (bool): 是否为训练模式
        data_aug_conf (dict): 数据增强配置参数
    
    Returns:
        list: 数据增强变换列表
    """
    transforms = []
    if train:
        if data_aug_conf is not None:
            resize_dims, crop = sample_augmentation(train, data_aug_conf)
            transforms.extend([
                T.Resize(size=resize_dims, antialias=True),
                T.RandomCrop(size=data_aug_conf['final_dim']),
            ])
        transforms.extend([
            T.RandomShortestSize(min_size=800, max_size=1333, antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomIoUCrop(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5,
                                           max_aspect_ratio=2.0, sampler_options={'min_iou': 0.3})], p=0.7),
            T.RandomZoomOut(fill={0: (124, 116, 104)}, side_range=(1.0, 1.5), p=0.3)
        ])
    else:
        if data_aug_conf is not None:
            target_size = data_aug_conf['final_dim']  # 直接使用目标尺寸
            transforms.append(T.Resize(size=target_size, antialias=True))
    
    return transforms

def get_transform(train, data_aug_conf=None):
    """
    获取完整的数据变换管道
    
    Args:
        train (bool): 是否为训练模式
        data_aug_conf (dict): 数据增强配置参数
    
    Returns:
        T.Compose: 变换组合
    """
    # 获取数据增强变换
    transforms = get_augmentation_transforms(train, data_aug_conf)
    
    # 添加标准化和类型转换
    transforms.extend([
        T.ToDtype(torch.float, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        T.ToPureTensor()])
    
    return T.Compose(transforms)

def sample_augmentation(is_train, data_aug_conf):
    """
    采样数据增强参数
    
    Args:
        is_train (bool): 是否为训练模式
        data_aug_conf (dict): 数据增强配置参数
    
    Returns:
        tuple: (resize_dims, crop)
    """
    fH, fW = data_aug_conf['final_dim']
    if is_train:
        resize = np.random.uniform(*data_aug_conf['resize_lim'])
        resize_dims = (int(fW * resize), int(fH * resize))
        newW, newH = resize_dims
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)
        crop_offset = int(data_aug_conf['resize_lim'][0] * data_aug_conf['final_dim'][0])
        crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
        crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    else:  # validation/test
        resize_dims = (fW, fH)
        crop_h = 0
        crop_w = 0
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    return resize_dims, crop

class MultiStageDataLoader:
    """多阶段数据加载器 - 专注于数据加载管理"""
    def __init__(self, config: MultiDatasetConfig, verbose: bool = True):
        self.config = config
        self.current_stage_idx = 0
        self.verbose = verbose
        self.logger = self._setup_logger()
        self._set_seed(config.global_seed) # 设置随机种子（仅用于数据增强的一致性）
        self._dataset_cache = {} # 缓存原始数据集
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('MultiStageDataLoader')
        if self.verbose:
            logger.setLevel(logging.INFO)
            ch_level = logging.INFO
        else:
            logger.setLevel(logging.ERROR)
            ch_level = logging.ERROR
        
        for handler in logger.handlers[:]: # 清除现有处理器
            logger.removeHandler(handler)
        ch = logging.StreamHandler()
        ch.setLevel(ch_level)  # 使用一致的级别
        formatter = logging.Formatter('%(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        return logger
    
    def _set_seed(self, seed: int):
        """设置随机种子（仅用于数据相关操作）"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def get_current_stage_config(self) -> DataStageConfig:
        """获取当前阶段配置"""
        if self.current_stage_idx >= len(self.config.stages):
            raise IndexError("所有数据阶段已完成")
        return self.config.stages[self.current_stage_idx]
    
    def get_stage_config(self, stage_idx: int) -> DataStageConfig:
        """获取指定阶段配置"""
        if stage_idx >= len(self.config.stages):
            raise IndexError(f"阶段索引 {stage_idx} 超出范围")
        return self.config.stages[stage_idx]

    def _initialize_raw_datasets(self, stage_config: DataStageConfig):
        """初始化原始数据集（不带变换）"""
        cache_key = f"{stage_config.stage.value}_{hash(str(stage_config.dataset_params))}"
        
        self.logger.info(f"正在初始化数据集阶段: {stage_config.stage.value}")
        self.logger.info(f"阶段类型: {type(stage_config.stage)}")
        self.logger.info(f"阶段值: {stage_config.stage}")
        self.logger.info(f"数据集参数: {stage_config.dataset_params}")
        
        if cache_key in self._dataset_cache:
            self.logger.info(f"从缓存中获取数据集: {cache_key}")
            return self._dataset_cache[cache_key]
        
        try:
            # 使用字符串比较而不是枚举比较
            if stage_config.stage.value == "Wildtrack2":
                self.logger.info("执行 Wildtrack 数据集初始化分支")
                # 检查路径是否存在
                root_path = stage_config.dataset_params['root']
                if not os.path.exists(root_path):
                    raise FileNotFoundError(f"Wildtrack 数据集路径不存在: {root_path}")
                
                # 创建原始 Wildtrack 数据集
                raw_dataset = WildtrackDataset(
                    root=root_path,
                    transforms=None
                )
                self._dataset_cache[cache_key] = raw_dataset
                self.logger.info(f"Wildtrack 原始数据集初始化完成: {len(raw_dataset)} 样本")
                return raw_dataset
                
            elif stage_config.stage.value == "MOT":
                self.logger.info("执行 MOT 数据集初始化分支")
                # 检查路径是否存在
                root_path = stage_config.dataset_params['root']
                if not os.path.exists(root_path):
                    raise FileNotFoundError(f"MOT 数据集根路径不存在: {root_path}")
                
                # 创建 MOT 训练和测试原始数据集
                datasets = stage_config.dataset_params.get('datasets', ['MOT17', 'MOT20'])
                
                self.logger.info(f"正在初始化 MOT 数据集，根路径: {root_path}")
                self.logger.info(f"数据集列表: {datasets}")
                
                train_dataset_raw = MOTDataset(
                    root=root_path,
                    datasets=datasets,
                    split='train',
                    transforms=None
                )
                
                test_dataset_raw = MOTDataset(
                    root=root_path,
                    datasets=datasets,
                    split='test',
                    transforms=None
                )
                
                raw_datasets = {
                    'train': train_dataset_raw,
                    'test': test_dataset_raw
                }
                self._dataset_cache[cache_key] = raw_datasets
                
                self.logger.info(f"MOT 原始数据集初始化完成:")
                self.logger.info(f"  - 训练分割: {len(train_dataset_raw)} 样本")
                self.logger.info(f"  - 测试分割: {len(test_dataset_raw)} 样本")
                return raw_datasets
                
            else:
                raise ValueError(f"不支持的数据集阶段: {stage_config.stage}")
        
        except Exception as e:
            self.logger.error(f"初始化 {stage_config.stage.value} 数据集失败: {e}")
            raise
    
    def _create_dataset_splits(self, stage_config: DataStageConfig):
        """创建数据集分割索引"""
        raw_datasets = self._initialize_raw_datasets(stage_config)
        
        # 添加调试信息
        self.logger.info(f"创建数据集分割 - 阶段: {stage_config.stage.value}")
        self.logger.info(f"原始数据集类型: {type(raw_datasets)}")
        
        # 使用字符串比较而不是枚举比较
        if stage_config.stage.value == "Wildtrack2":
            self.logger.info("执行 Wildtrack 数据集分割逻辑")
            # Wildtrack 数据集分割逻辑
            total_size = len(raw_datasets)
            indices = torch.randperm(total_size).tolist()
            
            train_size = int(0.8 * total_size)
            val_size = int((total_size - train_size) * 0.5)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]
            
            self.logger.info(f"Wildtrack 数据集分割: 训练{len(train_indices)}, 验证{len(val_indices)}, 测试{len(test_indices)}")
            
            return {
                'train': (raw_datasets, train_indices),
                'val': (raw_datasets, val_indices),
                'test': (raw_datasets, test_indices)
            }
            
        elif stage_config.stage.value == "MOT":
            self.logger.info("执行 MOT 数据集分割逻辑")
            # MOT 数据集分割逻辑（参考 wildtrack2_dataloader.py）
            train_dataset_raw = raw_datasets['train']
            test_dataset_raw = raw_datasets['test']
            
            self.logger.info(f"MOT 原始数据集大小 - 训练: {len(train_dataset_raw)}, 测试: {len(test_dataset_raw)}")
            
            # 训练集：使用完整的train split
            train_indices = list(range(len(train_dataset_raw)))
            
            # 测试集分割：将test split分为验证集和测试集
            test_total_size = len(test_dataset_raw)
            test_indices = torch.randperm(test_total_size).tolist()
            
            # 验证集和测试集各占test split的一半
            val_size = test_total_size // 2
            val_indices = test_indices[:val_size]
            final_test_indices = test_indices[val_size:]
            
            self.logger.info(f"MOT 数据集分割:")
            self.logger.info(f"  - 训练: {len(train_indices)} 样本 (来自train split)")
            self.logger.info(f"  - 验证: {len(val_indices)} 样本 (来自test split)")
            self.logger.info(f"  - 测试: {len(final_test_indices)} 样本 (来自test split)")
            
            return {
                'train': (train_dataset_raw, train_indices),
                'val': (test_dataset_raw, val_indices),
                'test': (test_dataset_raw, final_test_indices)
            }
        
        else:
            raise ValueError(f"不支持的数据集阶段: {stage_config.stage}")
    
    def create_dataset(self, stage_config: DataStageConfig, split: str = 'train') -> torch.utils.data.Dataset:
        """
        根据阶段配置创建数据集
        
        Args:
            stage_config (DataStageConfig): 阶段配置
            split (str): 数据集分割 ('train', 'val', 'test')
        
        Returns:
            torch.utils.data.Dataset: 数据集实例
        """
        # 获取数据集分割
        dataset_splits = self._create_dataset_splits(stage_config)
        raw_dataset, indices = dataset_splits[split]
        
        # 创建变换
        is_train = (split == 'train')
        transform = get_transform(train=is_train, data_aug_conf=stage_config.data_aug_conf)
        
        # 创建带变换的子数据集
        dataset = TransformSubset(raw_dataset, indices, transform=transform)
        
        self.logger.info(f"创建 {stage_config.stage.value} {split} 数据集: {len(dataset)} 样本")
        
        return dataset
    
    def create_dataloader(self, stage_config: DataStageConfig, split: str = 'train') -> TorchDataLoader:
        """
        创建数据加载器
        
        Args:
            stage_config (DataStageConfig): 阶段配置
            split (str): 数据集分割 ('train', 'val', 'test')
        
        Returns:
            TorchDataLoader: 数据加载器
        """
        dataset = self.create_dataset(stage_config, split)
        
        # 准备 DataLoader 参数
        is_train = (split == 'train')
        dataloader_kwargs = {
            'batch_size': stage_config.batch_size,
            'shuffle': is_train and stage_config.shuffle_train,
            'num_workers': stage_config.num_workers,
            'pin_memory': stage_config.pin_memory,
            'drop_last': is_train and stage_config.drop_last,
            'persistent_workers': True if stage_config.num_workers > 0 else False
        }
        
        # 如果有自定义的 collate_fn，则使用它
        if collate_fn is not None:
            dataloader_kwargs['collate_fn'] = collate_fn
        
        dataloader = TorchDataLoader(dataset, **dataloader_kwargs)
        
        self.logger.info(f"创建数据加载器 - 阶段: {stage_config.stage.value}, "
                        f"分割: {split}, 数据集大小: {len(dataset)}, 批次大小: {stage_config.batch_size}")
        
        return dataloader
    
    def get_current_dataloaders(self) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        Get train, validation, and test dataloaders for current stage
        
        Returns:
            Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]: 
            (train_loader, val_loader, test_loader)
        """
        current_config = self.get_current_stage_config()
        self.logger.info(f"Creating dataloaders for current stage: {current_config.stage.value}")
        self.logger.info(f"Stage name: {current_config.stage_name}")
        
        try:
            train_loader = self.create_dataloader(current_config, 'train')
            val_loader = self.create_dataloader(current_config, 'val') 
            test_loader = self.create_dataloader(current_config, 'test')
            self.logger.info(f"✓ Current stage dataloaders created successfully:")
            self.logger.info(f"  - Train: {len(train_loader)} batches, {len(train_loader.dataset)} samples")
            self.logger.info(f"  - Validation: {len(val_loader)} batches, {len(val_loader.dataset)} samples")
            self.logger.info(f"  - Test: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
            return train_loader, val_loader, test_loader
        except Exception as e:
            self.logger.error(f"✗ Failed to create dataloaders for current stage: {e}")
            raise
    
    def get_current_stage_name(self) -> str:
        """Get current stage name for logging purposes"""
        current_config = self.get_current_stage_config()
        return current_config.stage_name

    def get_current_stage_type(self) -> str:
        """Get current stage type (wildtrack/mot)"""
        current_config = self.get_current_stage_config()
        return current_config.stage.value
    
    def get_dataloaders_by_stage(self, stage_idx: int) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
        """
        获取指定阶段的训练、验证和测试数据加载器
        
        Args:
            stage_idx (int): 阶段索引
            
        Returns:
            Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]: (训练加载器, 验证加载器, 测试加载器)
        """
        original_idx = self.current_stage_idx
        self.current_stage_idx = stage_idx
        try:
            return self.get_current_dataloaders()
        finally:
            self.current_stage_idx = original_idx
    
    def next_stage(self) -> bool:
        """
        切换到下一个数据阶段
        
        Returns:
            bool: 是否成功切换到下一阶段
        """
        self.current_stage_idx += 1
        if self.current_stage_idx >= len(self.config.stages):
            self.logger.info("所有数据阶段已完成")
            return False
        
        stage_config = self.get_current_stage_config()
        self.logger.info(f"切换到数据阶段 {self.current_stage_idx + 1}: {stage_config.stage.value}")
        return True
    
    def reset_to_stage(self, stage_idx: int):
        """重置到指定阶段"""
        if stage_idx >= len(self.config.stages):
            raise IndexError(f"阶段索引 {stage_idx} 超出范围")
        self.current_stage_idx = stage_idx
        self.logger.info(f"重置到数据阶段 {stage_idx + 1}")
    
    def get_stage_info(self) -> Dict:
        """获取当前阶段信息"""
        if self.current_stage_idx >= len(self.config.stages):
            return {"stage": "completed", "stage_idx": self.current_stage_idx}
        
        stage_config = self.get_current_stage_config()
        return {
            "stage": stage_config.stage.value,
            "stage_name": stage_config.stage_name,
            "stage_idx": self.current_stage_idx,
            "total_stages": len(self.config.stages),
            "batch_size": stage_config.batch_size,
            "num_workers": stage_config.num_workers
        }
    
    def get_all_stages_info(self) -> List[Dict]:
        """获取所有阶段信息"""
        stages_info = []
        for idx, stage_config in enumerate(self.config.stages):
            stages_info.append({
                "stage_idx": idx,
                "stage": stage_config.stage.value,
                "stage_name": stage_config.stage_name,
                "batch_size": stage_config.batch_size,
                "dataset_params": stage_config.dataset_params
            })
        return stages_info


def create_default_data_config() -> MultiDatasetConfig:
    """创建默认的多数据集配置"""
    # Wildtrack 数据阶段配置
    wildtrack_stage = DataStageConfig(
        stage=DatasetStage.WILDTRACK,
        batch_size=2,
        data_aug_conf={
            'final_dim': [800, 600],  # (H, W)
            'resize_lim': [0.8, 1.2],
        },
        dataset_params={
            'root': '/home/s-jiang/Documents/datasets/Wildtrack2',
        },
        num_workers=8,
        pin_memory=True,
        drop_last=False,  # 改为 False，避免丢弃不完整的批次
        shuffle_train=True,
        stage_name="wildtrack_stage"
    )
    
    # MOT 数据阶段配置
    mot_stage = DataStageConfig(
        stage=DatasetStage.MOT,
        batch_size=4,  # 减小批次大小
        data_aug_conf={
            'final_dim': [800, 600],  # (H, W)
            'resize_lim': [0.8, 1.2],
        },
        dataset_params={
            'root': '/home/s-jiang/Documents/datasets/',
            'datasets': ['MOT17', 'MOT20'],  # 先只用一个数据集测试
        },
        num_workers=4,  # 减少worker数量
        pin_memory=True,
        drop_last=False,  # 改为 False
        shuffle_train=True,
        stage_name="mot_stage"
    )
    
    config = MultiDatasetConfig(
        stages=[wildtrack_stage, mot_stage],
        device="cuda" if torch.cuda.is_available() else "cpu",
        global_seed=42
    )
    
    return config

def create_multi_stage_config(batch_size: int, num_workers: int, datasets_root: str, 
                            wildtrack_params: dict = None, mot_params: dict = None) -> MultiDatasetConfig:
    """
    Create multi-stage data configuration with customizable parameters
    
    Args:
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        datasets_root: Root directory containing datasets
        wildtrack_params: Wildtrack dataset specific parameters (optional)
        mot_params: MOT dataset specific parameters (optional)
    
    Returns:
        MultiDatasetConfig: Configuration for multi-stage training
    """
    # Default Wildtrack parameters
    default_wildtrack_params = {
        'root': os.path.join(datasets_root, 'Wildtrack2'),
        'train_ratio': 0.8  # 80% for training, 20% for validation
    }
    
    # Default MOT parameters
    default_mot_params = {
        'root': datasets_root,
        'datasets': ['MOT17', 'MOT20']  # Use available datasets
    }
    
    # Merge user provided parameters with defaults
    wildtrack_config = {**default_wildtrack_params, **(wildtrack_params or {})}
    mot_config = {**default_mot_params, **(mot_params or {})}
    
    # Wildtrack stage configuration
    wildtrack_stage = DataStageConfig(
        stage=DatasetStage.WILDTRACK,
        batch_size=batch_size,
        data_aug_conf={
            'final_dim': [800, 600],
            'resize_lim': [0.8, 1.2],
            'bot_pct_lim': [0.0, 0.05],
            'rot_lim': [-3.0, 3.0],
            'rand_flip': True,
        },
        dataset_params=wildtrack_config,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle_train=True,
        stage_name="wildtrack_stage")
    
    # MOT stage configuration  
    mot_stage = DataStageConfig(
        stage=DatasetStage.MOT,
        batch_size= 4,#batch_size,
        data_aug_conf={
            'final_dim': [800, 600],
            'resize_lim': [0.8, 1.2],  # More conservative augmentation for fine-tuning
            'bot_pct_lim': [0.0, 0.05],
            'rot_lim': [-3.0, 3.0],
            'rand_flip': True,
        },
        dataset_params=mot_config,
        num_workers=int(num_workers/10),
        pin_memory=True,
        drop_last=True,
        shuffle_train=True,
        stage_name="mot_stage")
    
    config = MultiDatasetConfig(
        stages=[wildtrack_stage, mot_stage],
        device="cuda" if torch.cuda.is_available() else "cpu",
        global_seed=42)
    
    return config

def test_mot_dataset():
    """测试 MOT 数据集初始化"""
    try:
        print("=== MOT 数据集检查 ===")
        
        root_path = '/home/s-jiang/Documents/datasets/'
        datasets = ['MOT16']
        
        # 检查数据集路径
        for dataset in datasets:
            dataset_path = os.path.join(root_path, dataset)
            if not os.path.exists(dataset_path):
                print(f"✗ {dataset} 路径不存在: {dataset_path}")
                return False
        
        # 测试数据集创建
        from mot_datasets import MOTDataset
        train_dataset = MOTDataset(
            root=root_path,
            datasets=datasets,
            split='train',
            transforms=None
        )
        
        print(f"✓ MOT 数据集检查通过 - {len(train_dataset)} 训练样本")
        return True
        
    except Exception as e:
        print(f"✗ MOT 数据集检查失败: {e}")
        return False

def example_usage():
    """使用示例 - 展示如何在外部训练脚本中使用"""
    try:
        print("=== 多阶段数据加载器初始化 ===")
        
        # 创建数据配置
        data_config = create_default_data_config()
        
        # 创建多阶段数据加载器
        multi_data_loader = MultiStageDataLoader(data_config)
        
        # 获取所有阶段信息
        print("数据阶段配置:")
        for stage_info in multi_data_loader.get_all_stages_info():
            print(f"  阶段 {stage_info['stage_idx'] + 1}: {stage_info['stage']} - 批次大小: {stage_info['batch_size']}")
        
        # 测试所有阶段
        for stage_idx in range(len(data_config.stages)):
            stage_config = multi_data_loader.get_stage_config(stage_idx)
            stage_name = stage_config.stage.value
            
            print(f"\n=== {stage_name.upper()} 阶段 ===")
            
            try:
                train_loader, val_loader, test_loader = multi_data_loader.get_dataloaders_by_stage(stage_idx)
                
                print(f"✓ {stage_name} 阶段初始化成功")
                print(f"  训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}, 测试批次: {len(test_loader)}")
                
                # 简单的数据加载测试
                if len(train_loader) > 0:
                    sample_batch = next(iter(train_loader))
                    print(f"  数据格式: {len(sample_batch)} 个组件")
                
            except Exception as e:
                print(f"✗ {stage_name} 阶段失败: {e}")
                return False
        
        print("\n=== 初始化完成 ===")
        print("✓ 所有数据阶段初始化成功！")
        print("✓ 数据加载器可以正常使用")
        
        return True
        
    except Exception as e:
        print(f"✗ 初始化失败: {e}")
        return False

if __name__ == "__main__":
    example_usage()
