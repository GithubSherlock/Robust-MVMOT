import os
import json
import torch
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T

class WildtrackDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_dir = os.path.join(root, "ds", "img")
        self.ann_dir = os.path.join(root, "ds", "ann")
        self.mask_dir = os.path.join(root, "ds", "mask")
        self.imgs = [f for f in sorted(os.listdir(self.img_dir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        self.cam_mask_map = {}
        for mask_file in os.listdir(self.mask_dir):
            if mask_file.endswith('.png'):
                cam_id = mask_file.split('.')[0]
                self.cam_mask_map[cam_id] = os.path.join(self.mask_dir, mask_file)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)
        cam_id = img_name.split('_')[0]  # Obtain camera ID from the image name
        if cam_id not in self.cam_mask_map:
            raise KeyError(f"Camera ID {cam_id} not found in mask mapping")
        mask = read_image(self.cam_mask_map[cam_id], mode=ImageReadMode.GRAY)
        if mask.dtype == torch.uint8:
            mask = mask.float() / 255.0

        ann_path = os.path.join(self.ann_dir, self.imgs[idx] + ".json") # Load the corresponding annotation file
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        boxes = [] # Extract bounding boxes and labels for all pedestrian targets
        person_ids = []
        position_ids = []
        for obj in ann_data['objects']:
            if obj['classTitle'] == 'pedestrian':
                exterior = obj['points']['exterior'] # Get bounding box coordinates
                boxes.append([
                    exterior[0][0],  # xmin
                    exterior[0][1],  # ymin
                    exterior[1][0],  # xmax
                    exterior[1][1]])   # ymax
                for tag in obj['tags']: # Get person_id and position_id
                    if tag['name'] == 'person id':
                        person_ids.append(tag['value'])
                    elif tag['name'] == 'position id':
                        position_ids.append(tag['value'])
        
        
        if len(boxes) == 0: # Check that the boxes are empty before converting to tensor
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        num_objs = boxes.shape[0]  # Use shape[0] instead of len(boxes)
        
        labels = torch.ones((num_objs,), dtype=torch.int64) # Creation of labels etc. based on target quantities
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        if len(person_ids) == 0: # person_ids and position_ids also need to handle null cases
            person_ids = torch.zeros((0,), dtype=torch.int64)
        else:
            person_ids = torch.as_tensor(person_ids, dtype=torch.int64)
            
        if len(position_ids) == 0:
            position_ids = torch.zeros((0,), dtype=torch.int64)
        else:
            position_ids = torch.as_tensor(position_ids, dtype=torch.int64)

        boxes = torch.as_tensor(boxes, dtype=torch.float32) # Transform the boxes to tensor
        num_objs = len(boxes)
        labels = torch.ones((num_objs,), dtype=torch.int64) # All targets are pedestrian (label=1)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) # Calculate the area of the bounding box
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64) # Assuming all instances are not crowd
        person_ids = torch.as_tensor(person_ids, dtype=torch.int64) # Transform the person_id and position_id to tensor
        position_ids = torch.as_tensor(position_ids, dtype=torch.int64)

        
        img = tv_tensors.Image(img) # Constructing the target dictionary, i. e. the ground truth
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd,
            "person_ids": person_ids,
            "position_ids": position_ids,
            "mask": mask}

        if self.transforms is not None: # Apply the transforms
            img, target, mask = self.transforms(img, target, mask)
        if img.dtype == torch.uint8: # Add type checking and conversion
            img = img.float() / 255.0

        return img, target, img_name, mask

    def __len__(self):
        return len(self.imgs)
    
def _print_dataset_info(dataset):
    """Print dataset information"""
    img, target, img_name, mask = dataset[1] # Get first sample
    # Print sample information
    print(f"Image name: {img_name}")
    print(f"Image size: {img.shape}")
    print(f"Mask size: {mask.shape}")
    print("\nTarget info:")
    print(f"- Number of bounding box: {len(target['boxes'])}")
    print(f"- Bounding box coordinate: {target['boxes'].tolist()}")
    print(f"- Person IDs: {target['person_ids'].tolist()}")
    print(f"- Position IDs: {target['position_ids'].tolist()}")
    # Print related mask info
    print("\nMask info:")
    print(f"- Mask type: {mask.dtype}")
    print(f"- The range of Mask size: [{mask.min()}, {mask.max()}]")
    print(f"- Percentage of labelled regions: {(mask > 0).sum().item() / mask.numel():.2%}")
    # Print dataset size
    print(f"\nTotal sample size of the dataset: {len(dataset)}")

if __name__ == "__main__":
    ROOT_DIR = "/home/s-jiang/Documents/datasets/Wildtrack2"
    PRINT_DATASET_INFO = False
    dataset = WildtrackDataset(ROOT_DIR)
    while PRINT_DATASET_INFO:
        _print_dataset_info(dataset)
        PRINT_DATASET_INFO = False