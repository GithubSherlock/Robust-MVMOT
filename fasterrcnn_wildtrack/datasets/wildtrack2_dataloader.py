import torch
import logging
import numpy as np
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader as TorchDataLoader
import sys
import os
current_file = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file)
project_root = os.path.dirname(current_dir)
paths_to_add = [project_root, current_dir]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)
from wildtrack2_datasets import WildtrackDataset
from mot_datasets import MOTDataset
sys.path.insert(0, os.path.join(project_root, 'utils'))
from utils import collate_fn


def sample_augmentation(is_train, data_aug_conf):
    """
    采样数据增强参数
    
    根据训练/验证模式生成不同的图像变换参数。
    训练时使用随机缩放和裁剪，验证时使用固定变换。
    
    Args:
        is_train (bool): 是否为训练模式
        data_aug_conf (dict): 数据增强配置参数
    
    Returns:
        tuple: (resize_dims, crop)
            - resize_dims: 缩放后的尺寸 (W, H)
            - crop: 裁剪参数 (x, y, x+W, y+H)
    """
    fH, fW = data_aug_conf['final_dim']
    if is_train:
        resize = np.random.uniform(*data_aug_conf['resize_lim'])
        resize_dims = (int(fW * resize), int(fH * resize))
        newW, newH = resize_dims

        # center it
        crop_h = int((newH - fH) / 2)
        crop_w = int((newW - fW) / 2)

        crop_offset = int(data_aug_conf['resize_lim'][0] * data_aug_conf['final_dim'][0])
        crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
        crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    else:  # validation/test
        # do a perfect resize
        resize_dims = (fW, fH)
        crop_h = 0
        crop_w = 0
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    return resize_dims, crop


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
                T.RandomCrop(size=data_aug_conf['final_dim']),])
        
        transforms.extend([
            T.RandomShortestSize(min_size=600, max_size=1000, antialias=True),  # Base scale adjustment
            T.RandomHorizontalFlip(p=0.5),  # Random Horizontal Flip
            # T.RandomApply([T.RandomIoUCrop(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5,
            #                                max_aspect_ratio=2.0, sampler_options={'min_iou': 0.3})], p=0.7),  # Random trimming
            T.RandomZoomOut(fill={0: (124, 116, 104)}, side_range=(1.0, 1.5), p=0.3)  # Random Expanded View
        ])
    else:
        # 验证/测试模式的固定变换
        if data_aug_conf is not None:
            resize_dims, crop = sample_augmentation(train, data_aug_conf)
            transforms.append(T.Resize(size=resize_dims, antialias=True))
    
    return transforms


def get_transform(train, data_aug_conf=None):
    """_summary_
    Ensure that horizontal flip is applied to the image, target and mask at the same time
    
    Args:
        train (bool): If True, apply data augmentation for training.
        data_aug_conf (dict): 数据增强配置参数，包含 'final_dim' 和 'resize_lim'
    Returns:
        transforms (callable): A callable that applies the transformations.
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

class TransformSubset(torch.utils.data.Subset):
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


class DataLoader:
    """
    通用数据加载器类
    支持任意数据集模块，默认为WildtrackDataset
    提供训练、验证和测试数据加载功能
    """
    def __init__(self, root_path, batch_size=4, num_workers=4, data_aug_conf=None, dataset_class=None,
                 **dataset_kwargs):
        """
        初始化DataLoader
        
        Args:
            root_path (str): 数据集根路径
            batch_size (int): 批次大小
            num_workers (int): 工作进程数
            data_aug_conf (dict): 数据增强配置
            dataset_class (class): 数据集类，默认为WildtrackDataset
            **dataset_kwargs: 传递给数据集类的额外参数
        """
        if dataset_class is None:
            dataset_class = WildtrackDataset
            
        self.dataset_class = dataset_class
        self.root_path = root_path
        self.dataset_kwargs = dataset_kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if data_aug_conf is None: # 设置默认数据增强配置
            self.data_aug_conf = {
                'final_dim': [800, 600],  # 最终输出尺寸 (W, H)
                'resize_lim': [0.8, 1.2]}  # 缩放范围
        else:
            self.data_aug_conf = data_aug_conf
            
        self._initialize_datasets()  # 初始化数据集
        self._create_splits()        # 创建数据集分割
        self._create_dataloaders()   # 创建数据加载器
    
    def _initialize_datasets(self):
        """根据数据集类型初始化训练和测试数据集"""
        if self.dataset_class.__name__ == 'MOTDataset':
            # 对于MOTDataset，分别创建训练集和测试集
            datasets = self.dataset_kwargs.get('datasets', ['MOT16'])
            
            # 创建训练数据集 (用于训练和验证)
            self.train_dataset_raw = self.dataset_class(
                root=self.root_path, 
                datasets=datasets, 
                split='train', 
                transforms=None,
                sample_ratio=1.0,  # 禁用内部采样
                max_samples=None   # 禁用样本限制
            )
            
            # 创建测试数据集 (用于验证和测试)
            self.test_dataset_raw = self.dataset_class(
                root=self.root_path, 
                datasets=datasets, 
                split='test', 
                transforms=None,
                sample_ratio=1.0,  # 禁用内部采样
                max_samples=None   # 禁用样本限制
            )
            logging.info(f"MOTDataset initialized:")
            logging.info(f"  - Train split size: {len(self.train_dataset_raw)}")
            logging.info(f"  - Test split size: {len(self.test_dataset_raw)}")
        else:
            # 对于其他数据集，使用原有逻辑
            self.dataset = self.dataset_class(root=self.root_path, transforms=None)
            logging.info(f"{self.dataset_class.__name__} initialized with {len(self.dataset)} samples")
    
    def _validate_data_sources(self):
        """
        验证数据来源的正确性
        """
        if self.dataset_class.__name__ == 'MOTDataset':
            # 验证数据来源
            checks = {
                '训练集': self.train_dataset.dataset is self.train_dataset_raw,
                '验证集': self.val_dataset.dataset is self.train_dataset_raw,
                '测试集': self.test_dataset.dataset is self.test_dataset_raw
            }
            
            print(f"\n🔍 数据来源验证:")
            for name, is_correct in checks.items():
                status = "✅ 正确" if is_correct else "❌ 错误"
                print(f"   - {name}: {status}")
            
            if not all(checks.values()):
                logging.warning("⚠️ 数据来源验证失败！请检查分割逻辑。")
    
    def _create_splits(self):
        """创建训练、验证和测试数据集分割"""
        if self.dataset_class.__name__ == 'MOTDataset':
            # 🎯 MOTDataset的新分割逻辑
            train_total_size = len(self.train_dataset_raw)
            test_total_size = len(self.test_dataset_raw)
            
            # 📊 新的分割策略
            # 1. 从训练集采样50%
            train_sample_ratio = self.dataset_kwargs.get('train_sample_ratio', 0.5)
            train_sample_size = int(train_total_size * train_sample_ratio)
            
            # 2. 从采样的训练集中分出10%作为验证集
            val_ratio_from_train = self.dataset_kwargs.get('val_ratio_from_train', 0.1)
            val_size = int(train_sample_size * val_ratio_from_train)
            # actual_train_size = train_sample_size - val_size
            
            # 3. 测试集数量与验证集相同
            test_size = val_size
            
            # 🎲 生成随机索引
            # 训练集索引（用于训练+验证）
            train_all_indices = torch.randperm(train_total_size)[:train_sample_size].tolist()
            
            # 从训练集索引中分割出验证集
            val_indices = train_all_indices[:val_size]
            actual_train_indices = train_all_indices[val_size:]
            
            # 测试集索引（从原始测试集中采样）
            test_indices = torch.randperm(test_total_size)[:test_size].tolist()
            
            # 🏗️ 创建数据集子集
            self.train_dataset = TransformSubset(self.train_dataset_raw, actual_train_indices)
            self.val_dataset = TransformSubset(self.train_dataset_raw, val_indices)  # ✅ 来自训练集
            self.test_dataset = TransformSubset(self.test_dataset_raw, test_indices)
            
            # 🆕 存储索引信息用于统计
            self._train_indices = actual_train_indices
            self._val_indices = val_indices
            self._test_indices = test_indices
            
            # 📈 打印分割信息
            logging.info(f"MOT数据集分割完成:")
            logging.info(f"  - 原始训练集: {train_total_size}")
            logging.info(f"  - 原始测试集: {test_total_size}")
            logging.info(f"  - 训练集采样: {train_sample_size}/{train_total_size} ({train_sample_ratio:.1%})")
            logging.info(f"  - 最终训练集: {len(actual_train_indices)} (来自训练集)")
            logging.info(f"  - 验证集: {len(val_indices)} (来自训练集, {val_ratio_from_train:.1%})")
            logging.info(f"  - 测试集: {len(test_indices)} (来自测试集, 与验证集数量相同)")
            
            # 🔍 验证数据来源
            self._validate_data_sources()
            
        else:
            # 其他数据集的分割逻辑保持不变
            total_size = len(self.dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            indices = torch.randperm(total_size)
            train_indices = indices[:train_size].tolist()
            val_indices = indices[train_size:train_size + val_size].tolist()
            test_indices = indices[train_size + val_size:].tolist()
            
            self.train_dataset = TransformSubset(self.dataset, train_indices)
            self.val_dataset = TransformSubset(self.dataset, val_indices)
            self.test_dataset = TransformSubset(self.dataset, test_indices)
            
            self._train_indices = train_indices
            self._val_indices = val_indices
            self._test_indices = test_indices
            
            logging.info(f"数据集分割: 训练集{train_size}, 验证集{val_size}, 测试集{test_size}")
    
    def _create_dataloaders(self):
        """Create data loaders for train, validation, and test datasets"""
        self.train_loader = TorchDataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True,
            collate_fn=collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False)
        
        self.val_loader = TorchDataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False)
        
        self.test_loader = TorchDataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False,
            collate_fn=collate_fn, 
            num_workers=self.num_workers,
            pin_memory=True, 
            persistent_workers=True if self.num_workers > 0 else False)
    
    def get_dataloaders(self):
        """
        获取训练、验证和测试数据加载器
        
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_info(self):
        """
        获取数据集信息
        增加数据来源信息
        """
        info = {
            'dataset_class': self.dataset_class.__name__,
            'root_path': self.root_path,
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'data_aug_conf': self.data_aug_conf
        }
        
        if self.dataset_class.__name__ == 'MOTDataset':
            # 计算比例
            train_raw_size = len(self.train_dataset_raw)
            test_raw_size = len(self.test_dataset_raw)
            
            train_size = len(self.train_dataset.indices) if hasattr(self.train_dataset, 'indices') else 0
            val_size = len(self.val_dataset.indices) if hasattr(self.val_dataset, 'indices') else 0
            test_size = len(self.test_dataset.indices) if hasattr(self.test_dataset, 'indices') else 0
            
            # MOTDataset 相关信息
            info.update({
                'train_raw_size': train_raw_size,
                'test_raw_size': test_raw_size,
                'train_size': train_size,
                'val_size': val_size,
                'test_size': test_size,
                'train_sample_ratio': self.dataset_kwargs.get('train_sample_ratio', 0.5),
                'val_ratio_from_train': self.dataset_kwargs.get('val_ratio_from_train', 0.1),
                # 🆕 数据来源信息
                'train_source': 'train_dataset_raw',
                'val_source': 'train_dataset_raw (从训练集分割)',
                'test_source': 'test_dataset_raw',
                # 计算实际比例
                'val_ratio_of_total_train': val_size / train_raw_size if train_raw_size > 0 else 0,
                'test_ratio_of_total_test': test_size / test_raw_size if test_raw_size > 0 else 0
            })
        else:
            # 其他数据集信息
            if hasattr(self, 'dataset'):
                total_size = len(self.dataset)
                info.update({
                    'total_size': total_size,
                    'train_size': len(self.train_dataset.indices) if hasattr(self.train_dataset, 'indices') else 0,
                    'val_size': len(self.val_dataset.indices) if hasattr(self.val_dataset, 'indices') else 0,
                    'test_size': len(self.test_dataset.indices) if hasattr(self.test_dataset, 'indices') else 0
                })
        
        return info
    
    def print_dataset_info(self):
        """
        打印数据集信息
        增强数据来源显示
        """
        try:
            info = self.get_dataset_info()
            
            print(f"\n📊 {info['dataset_class']} DataLoader Information:")
            print(f"   - 数据集类型: {info['dataset_class']}")
            print(f"   - 根路径: {info['root_path']}")
            print(f"   - 批次大小: {info['batch_size']}")
            print(f"   - 工作进程: {info['num_workers']}")
            
            if info['dataset_class'] == 'MOTDataset':
                print(f"\n📈 MOT数据集统计:")
                print(f"   - 原始训练集: {info.get('train_raw_size', 'N/A')}")
                print(f"   - 原始测试集: {info.get('test_raw_size', 'N/A')}")
                print(f"   - 最终训练集: {info.get('train_size', 'N/A')} ({info.get('train_source', 'N/A')})")
                print(f"   - 验证集: {info.get('val_size', 'N/A')} ({info.get('val_source', 'N/A')})")
                print(f"   - 测试集: {info.get('test_size', 'N/A')} ({info.get('test_source', 'N/A')})")
                
                print(f"\n📊 采样比例:")
                print(f"   - 训练集采样比例: {info.get('train_sample_ratio', 'N/A'):.1%}")
                print(f"   - 验证集占训练集比例: {info.get('val_ratio_from_train', 'N/A'):.1%}")
                print(f"   - 验证集占总训练集比例: {info.get('val_ratio_of_total_train', 0):.1%}")
                print(f"   - 测试集占总测试集比例: {info.get('test_ratio_of_total_test', 0):.1%}")
            else:
                print(f"\n📈 数据集统计:")
                print(f"   - 总大小: {info.get('total_size', 'N/A')}")
                print(f"   - 训练集: {info.get('train_size', 'N/A')}")
                print(f"   - 验证集: {info.get('val_size', 'N/A')}")
                print(f"   - 测试集: {info.get('test_size', 'N/A')}")
            
            print(f"\n🔧 数据增强配置:")
            print(f"   - 最终尺寸: {info['data_aug_conf']['final_dim']}")
            print(f"   - 缩放范围: {info['data_aug_conf']['resize_lim']}")
            
        except Exception as e:
            print(f"❌ 打印数据集信息时出错: {str(e)}")
            print("🔍 可用属性:", [attr for attr in dir(self) if not attr.startswith('_')])

def create_wildtrack_dataloaders(root_path, batch_size=4, num_workers=4, data_aug_conf=None, dataset_class=None):
    """
    创建数据加载器（向后兼容函数）
    
    Args:
        root_path (str): 数据集根路径
        batch_size (int): 批次大小
        num_workers (int): 工作进程数
        data_aug_conf (dict): 数据增强配置
        dataset_class (class): 数据集类，默认为WildtrackDataset
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataloader = DataLoader(root_path, batch_size, num_workers, data_aug_conf, dataset_class)
    return dataloader.get_dataloaders()

if __name__ == "__main__":
    def wildtrack2_dataloader_test():
        root_path = "/home/s-jiang/Documents/datasets/Wildtrack2"
        try:
            # Methode 1：use default dataset: WildtrackDataset
            dataloader = DataLoader(root_path, batch_size=2, num_workers=2)
            dataloader.print_dataset_info()
            # Methode 2：Use custom dataset class
            # from some_other_dataset import SomeDataset
            # dataloader = DataLoader(root_path, batch_size=2, num_workers=2, dataset_class=SomeDataset)
            train_loader, val_loader, test_loader = dataloader.get_dataloaders()
            # Test data loading
            print("\n=== Testing data loading ===")
            for batch_idx, (images, targets, img_names, masks) in enumerate(train_loader):
                print(f"Batch {batch_idx}: {len(images)} images loaded")
                print(f"Image shapes: {[img.shape for img in images]}")
                print(f"Image names: {img_names}")
                if batch_idx == 0:
                    break     
        except Exception as e:
            print(f"Error: {e}")
            print("Please check the dataset path and configuration.")
            import traceback
            traceback.print_exc()
    def mot_dataloader_test():
        root_path = "/home/s-jiang/Documents/datasets"
        try:
            print("=== Testing MOTDataset ===")
            mot_dataloader = DataLoader(
                root_path=root_path,
                batch_size=8, 
                num_workers=8,
                dataset_class=MOTDataset,
                datasets=['MOT16', 'MOT17', 'MOT20'],  # 可以根据需要添加更多数据集
                split='train'  # 这个参数在新逻辑中不会影响分割策略
            )
            mot_dataloader.print_dataset_info()
            
            train_loader, val_loader, test_loader = mot_dataloader.get_dataloaders()
            
            # Test data loading
            print("\n=== Testing MOT data loading ===")
            for batch_idx, (images, targets, img_names, masks) in enumerate(train_loader):
                print(f"Train Batch {batch_idx}: {len(images)} images loaded")
                if batch_idx == 0:
                    break
                    
            for batch_idx, (images, targets, img_names, masks) in enumerate(val_loader):
                print(f"Val Batch {batch_idx}: {len(images)} images loaded")
                if batch_idx == 0:
                    break
                    
        except Exception as e:
            print(f"Error testing MOTDataset: {e}")
            import traceback
            traceback.print_exc()
        
    # wildtrack2_dataloader_test()
    mot_dataloader_test()