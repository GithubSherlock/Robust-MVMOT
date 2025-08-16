import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.io import read_image
from torchvision.io import ImageReadMode
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T

class WildtrackDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None, use_preprocess=False, context_size=20,
                 mask_method='gaussian', mask_params=None):
        self.root = root
        self.transforms = transforms
        self.use_preprocess = use_preprocess
        self.context_size = context_size
        self.mask_method = mask_method
        self.mask_params = mask_params or {}
        self.img_dir = os.path.join(root, "ds", "img")
        self.ann_dir = os.path.join(root, "ds", "ann")
        self.mask_dir = os.path.join(root, "ds", "mask")
        self.img_filtered_dir = os.path.join(root, "ds", "img_" + f"{mask_method}")
        self.imgs = [f for f in sorted(os.listdir(self.img_dir))
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]
        self.cam_mask_map = {}
        for mask_file in os.listdir(self.mask_dir):
            if mask_file.endswith('.png'):
                cam_id = mask_file.split('.')[0]
                self.cam_mask_map[cam_id] = os.path.join(self.mask_dir, mask_file)
        # 如果启用预处理，检查并处理所有图像
        if use_preprocess:
            os.makedirs(self.img_filtered_dir, exist_ok=True)
            self._preprocess_all_images()
    
    def _preprocess_all_images(self):
        """预处理所有需要处理的图像"""
        # 获取需要处理的图像列表
        to_process = []
        for img_name in self.imgs:
            filtered_path = os.path.join(self.img_filtered_dir, img_name)
            if not os.path.exists(filtered_path):
                to_process.append(img_name)
        if not to_process:
            return
        
        print(f"预处理 {len(to_process)} 张图像...")
        for img_name in tqdm(to_process, desc="预处理图像"):
            # 加载原始图像
            img_path = os.path.join(self.img_dir, img_name)
            img = read_image(img_path)
            img = img.float() / 255.0
            
            # 加载mask
            cam_id = img_name.split('_')[0]
            mask = read_image(self.cam_mask_map[cam_id], mode=ImageReadMode.GRAY)
            mask = (mask > 0).float()
            
            # 加载标注
            ann_path = os.path.join(self.ann_dir, img_name + ".json")
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
            
            # 构建target
            boxes = []
            for obj in ann_data['objects']:
                if obj['classTitle'] == 'pedestrian':
                    exterior = obj['points']['exterior']
                    boxes.append([
                        exterior[0][0],  # xmin
                        exterior[0][1],  # ymin
                        exterior[1][0],  # xmax
                        exterior[1][1]])   # ymax
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target = {"boxes": boxes}
            
            # 预处理图像
            self.preprocess_image(img, target, mask, img_name)

    def preprocess_image(self, img, target, mask, img_name):
        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.squeeze().numpy()
        result = img_np.copy()

        # 生成过渡权重
        transition_width = self.mask_params.get('transition_width', 10)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (transition_width, transition_width))
        
        # 创建过渡区域
        dilated_mask = cv2.dilate(mask_np.astype(np.uint8), kernel)
        eroded_mask = cv2.erode(mask_np.astype(np.uint8), kernel)
        transition_mask = dilated_mask - eroded_mask
        
        # 生成平滑的权重图
        weight_map = cv2.GaussianBlur(mask_np.astype(np.float32), 
                                    (transition_width*2+1, transition_width*2+1), 
                                    transition_width/3)
        
        # 根据不同方法处理非掩码区域
        if self.mask_method == 'gaussian':
            # 增强高斯模糊
            kernel_size = self.mask_params.get('kernel_size', 15)
            sigma = self.mask_params.get('sigma', 15)
            blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
            blend_mask = np.expand_dims(weight_map, axis=-1)
            result = img_np * blend_mask + blurred * (1 - blend_mask)
            
        elif self.mask_method == 'solid':
            # 固定颜色填充
            color = self.mask_params.get('color', [0.5, 0.5, 0.5])
            color_img = np.full_like(img_np, color)
            blend_mask = np.expand_dims(weight_map, axis=-1)
            result = img_np * blend_mask + color_img * (1 - blend_mask)
            
        elif self.mask_method == 'noise':
            noise_level = self.mask_params.get('noise_level', 0.3)
            noise = np.random.normal(0.5, noise_level, img_np.shape)
            noise = np.clip(noise, 0, 1)
            # 使用权重图进行平滑混合
            blend_mask = np.expand_dims(weight_map, axis=-1)
            result = img_np * blend_mask + noise * (1 - blend_mask)
            
        elif self.mask_method == 'combined':
            kernel_size = self.mask_params.get('kernel_size', 20)
            sigma = self.mask_params.get('sigma', 15)
            noise_level = self.mask_params.get('noise_level', 0.2)
            
            # 1. 强化高斯模糊
            blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
            # 2. 添加噪声
            noise = np.random.normal(0.5, noise_level, img_np.shape)
            blurred = np.clip(blurred + noise, 0, 1)
            
            # 使用权重图进行平滑混合
            blend_mask = np.expand_dims(weight_map, axis=-1)
            result = img_np * blend_mask + blurred * (1 - blend_mask)

        # 处理目标框区域
        boxes = target['boxes'].numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            fit_x = int((x1 + x2) / 2)
            fit_y = int(y2)
            
            fit_x = min(max(0, fit_x), mask_np.shape[1] - 1)
            fit_y = min(max(0, fit_y), mask_np.shape[0] - 1)
            
            if mask_np[fit_y, fit_x] == 1:
                cx1 = max(0, x1 - self.context_size)
                cy1 = max(0, y1 - self.context_size)
                cx2 = min(img_np.shape[1], x2 + self.context_size)
                cy2 = min(img_np.shape[0], y2 + self.context_size)
                result[cy1:cy2, cx1:cx2] = img_np[cy1:cy2, cx1:cx2]
        
        # 保存处理后的图像
        output_path = os.path.join(self.img_filtered_dir, img_name)
        cv2.imwrite(output_path, (result * 255).astype(np.uint8)[..., ::-1])
        
        return torch.from_numpy(result).permute(2, 0, 1)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = read_image(img_path)
        img = img.float() / 255.0
        cam_id = img_name.split('_')[0]  # Obtain camera ID from the image name
        if cam_id not in self.cam_mask_map:
            raise KeyError(f"Camera ID {cam_id} not found in mask mapping")
        mask = read_image(self.cam_mask_map[cam_id], mode=ImageReadMode.GRAY)
        mask = (mask > 0).float()
        
        img = tv_tensors.Image(img) # Constructing the target dictionary, i. e. the ground truth
        mask = tv_tensors.Mask(mask)
        # if mask.dtype == torch.uint8:
        #     mask = mask.float() / 255.0

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

        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd,
            "person_ids": person_ids,
            "position_ids": position_ids,
            "cam_id": cam_id,
            'mask': mask}

        if self.transforms is not None: # Apply the transforms
            img, target, mask = self.transforms(img, target, mask)
        # if img.dtype == torch.uint8: # Add type checking and conversion
        #     img = img.float() / 255.0

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
    PRINT_DATASET_INFO = False  # Set to True to print dataset info
    dataset = WildtrackDataset(
        root=ROOT_DIR,
        use_preprocess=False,
        context_size=10,
        mask_method='gaussian',
        mask_params={
            'kernel_size': 35,
            'sigma': 35,
            'noise_level': 0.2,
            'contrast': 0.2,
            'color': [0.5, 0.5, 0.5]})
    while PRINT_DATASET_INFO:
        _print_dataset_info(dataset)
        PRINT_DATASET_INFO = False
    