import os
import json
import torch
from torchvision.io import read_image
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import torchvision.transforms.v2 as T

class WildtrackDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # 获取img和ann目录
        self.img_dir = os.path.join(root, "ds", "img")
        self.ann_dir = os.path.join(root, "ds", "ann")
        # 获取所有图像文件名并排序
        self.imgs = [f for f in sorted(os.listdir(self.img_dir)) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('.')]

    def __getitem__(self, idx):
        # 加载图像
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        img = read_image(img_path)
        
        # 加载对应的标注文件
        ann_path = os.path.join(self.ann_dir, self.imgs[idx] + ".json")
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        # 提取所有行人目标的边界框和标签
        boxes = []
        person_ids = []
        position_ids = []
        
        for obj in ann_data['objects']:
            if obj['classTitle'] == 'pedestrian':
                # 获取边界框坐标
                exterior = obj['points']['exterior']
                boxes.append([
                    exterior[0][0],  # xmin
                    exterior[0][1],  # ymin
                    exterior[1][0],  # xmax
                    exterior[1][1]   # ymax
                ])
                
                # 获取person_id和position_id
                for tag in obj['tags']:
                    if tag['name'] == 'person id':
                        person_ids.append(tag['value'])
                    elif tag['name'] == 'position id':
                        position_ids.append(tag['value'])
        
        # 在转换为tensor之前，检查boxes是否为空
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        
        num_objs = boxes.shape[0]  # 使用shape[0]替代len(boxes)
        
        # 根据目标数量创建对应的标签等
        labels = torch.ones((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # person_ids 和 position_ids 也需要处理空的情况
        if len(person_ids) == 0:
            person_ids = torch.zeros((0,), dtype=torch.int64)
        else:
            person_ids = torch.as_tensor(person_ids, dtype=torch.int64)
            
        if len(position_ids) == 0:
            position_ids = torch.zeros((0,), dtype=torch.int64)
        else:
            position_ids = torch.as_tensor(position_ids, dtype=torch.int64)

        # 转换为tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)
        
        # 所有目标都是行人类别(label=1)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        
        # 计算边界框面积
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # 假设所有实例都不是crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        # 将person_id和position_id转换为tensor
        person_ids = torch.as_tensor(person_ids, dtype=torch.int64)
        position_ids = torch.as_tensor(position_ids, dtype=torch.int64)

        # 构建target字典
        img = tv_tensors.Image(img)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(img)),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd,
            "person_ids": person_ids,
            "position_ids": position_ids
        }

        # 应用transforms
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        # 添加类型检查和转换
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        return img, target

    def __len__(self):
        return len(self.imgs)