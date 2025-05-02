import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Subset, DataLoader
from .wildtrack2_datasets import WildtrackDataset
from utils import utils

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

class TransformSubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices, transform=None):
        super().__init__(dataset, indices)
        self.transform = transform

    def __getitem__(self, idx):
        img, target = self.dataset[self.indices[idx]]
        # 打印原始图像信息
        print(f"Original image - type: {type(img)}")
        if isinstance(img, torch.Tensor):
            print(f"Original tensor - dtype: {img.dtype}, shape: {img.shape}")
        if self.transform is not None:
            # 逐步应用转换并打印中间状态
            for t in self.transform.transforms:
                img = t(img)
                print(f"After {t.__class__.__name__} - dtype: {img.dtype}, "
                      f"shape: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")
        return img, target

def create_wildtrack_dataloaders(root_path, batch_size=4, num_workers=4):
    """创建训练、验证和测试数据加载器"""
    # 创建完整数据集
    dataset = WildtrackDataset(root=root_path, transforms=None)
    
    # 按80-10-10比例划分数据集
    total_size = len(dataset)
    indices = torch.randperm(total_size).tolist()
    
    train_size = int(0.8 * total_size)
    val_size = int((total_size - train_size) * 0.5)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 创建数据集子集
    train_dataset = TransformSubset(dataset, train_indices, transform=get_transform(train=True))
    val_dataset = TransformSubset(dataset, val_indices, transform=get_transform(train=False))
    test_dataset = TransformSubset(dataset, test_indices, transform=get_transform(train=False))
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=utils.collate_fn, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=utils.collate_fn, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=utils.collate_fn, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader
