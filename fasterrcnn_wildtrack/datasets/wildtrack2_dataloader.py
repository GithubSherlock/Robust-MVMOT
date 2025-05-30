import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
from .wildtrack2_datasets import WildtrackDataset
from utils.utils import collate_fn


def get_transform(train):
    """_summary_
    Ensure that horizontal flip is applied to the image, target and mask at the same time
    
    Args:
        train (bool): If True, apply data augmentation for training.
    Returns:
        transforms (callable): A callable that applies the transformations.
    """
    transforms = []
    if train:
        transforms.extend([ # Data augmentation
            # T.RandomShortestSize(min_size=800, max_size=1333, antialias=True), # Base scale adjustment
            T.RandomHorizontalFlip(p=0.5), # Random Horizontal Flip
            # T.RandomApply([T.RandomIoUCrop(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5,
            #                                max_aspect_ratio=2.0,sampler_options={'min_iou': 0.3})], p=0.7), # Random trimming
            # T.RandomPhotometricDistort(brightness=(0.8, 1.2), contrast=(0.8, 1.2),saturation=(0.8, 1.2),
            #                            hue=(-0.1, 0.1), p=0.5), # Random photometric transformation
            # T.RandomApply([T.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.3), # Random blur
            # T.RandomZoomOut(fill={0: (124, 116, 104)}, side_range=(1.0, 1.5), p=0.3) # Random Expanded View
            ])
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
            raise KeyError(f"Error loading image at index {idx}: {str(e)}")

def create_wildtrack_dataloaders(root_path, batch_size=4, num_workers=4):
    """
    Creating training, validation and test data loaders
    and dividing the dataset on an 80-10-10 scale
    """
    dataset = WildtrackDataset(root=root_path, transforms=None)
    total_size = len(dataset)
    indices = torch.randperm(total_size).tolist()
    
    train_size = int(0.8 * total_size)
    val_size = int((total_size - train_size) * 0.5)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = TransformSubset(dataset, train_indices, transform=get_transform(train=True))
    val_dataset = TransformSubset(dataset, val_indices, transform=get_transform(train=False))
    test_dataset = TransformSubset(dataset, test_indices, transform=get_transform(train=False))
    
    train_loader = DataLoader( # Creating the Data Loaders
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True)
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True)
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn, 
        num_workers=num_workers,
        pin_memory=True, 
        persistent_workers=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    pass
