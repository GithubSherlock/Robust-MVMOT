import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F

from datasets.wildtrack2_dataloader import create_wildtrack_dataloaders

class GroundTruthVisualizer:
    def __init__(self, data_root, device='cuda', vis_params=None):
        self.device = device
        self.data_root = data_root
        self.vis_params = vis_params or {
            'box_color': (0, 255, 0),
            'thickness': 3,
            'alpha': 0.6,
            'show_labels': True,
            'text_color': (255, 255, 255),
            'text_thickness': 2,
            'text_size': 0.8
        }
    
    def _draw_boxes_on_image(self, image, boxes, labels=None, person_ids=None, position_ids=None):
        """
        在图像上绘制边界框
        """
        if isinstance(image, torch.Tensor):
            # Convert tensor to numpy array
            if image.dim() == 3 and image.shape[0] == 3:  # (C, H, W)
                image = image.permute(1, 2, 0)
            # 确保数值范围在[0,1]或[0,255]
            if image.max() <= 1.0:
                image = (image * 255).clamp(0, 255).byte().cpu().numpy()
            else:
                image = image.clamp(0, 255).byte().cpu().numpy()
        
        # Ensure image is in BGR format for OpenCV
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Create overlay for transparency effect
        overlay = image.copy()
        
        if len(boxes) == 0:
            return image
        
        for i, box in enumerate(boxes):
            if isinstance(box, torch.Tensor):
                x1, y1, x2, y2 = box.int().tolist()
            else:
                x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w-1, x2), min(h-1, y2)
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), 
                         self.vis_params['box_color'], 
                         self.vis_params['thickness'])
            
            # Add label if available
            if self.vis_params['show_labels']:
                label_parts = []
                
                # 添加类别标签
                if labels is not None:
                    label_parts.append("Person")  # Person
                
                # # 添加person_id
                # if person_ids is not None and len(person_ids) > i:
                #     if isinstance(person_ids[i], torch.Tensor):
                #         pid = int(person_ids[i].item())
                #     else:
                #         pid = int(person_ids[i])
                #     label_parts.append(f"P{pid}")
                
                # # 添加position_id
                # if position_ids is not None and len(position_ids) > i:
                #     if isinstance(position_ids[i], torch.Tensor):
                #         pos_id = int(position_ids[i].item())
                #     else:
                #         pos_id = int(position_ids[i])
                #     label_parts.append(f"Pos{pos_id}")
                
                label_text = " ".join(label_parts) if label_parts else f"Box_{i}"
                
                # Calculate text size and position
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 
                    self.vis_params['text_size'], self.vis_params['text_thickness'])
                
                # Draw text background
                cv2.rectangle(overlay, (x1, y1 - text_height - baseline - 5), 
                             (x1 + text_width, y1), self.vis_params['box_color'], -1)
                
                # Draw text
                cv2.putText(overlay, label_text, (x1, y1 - baseline - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, self.vis_params['text_size'],
                           self.vis_params['text_color'], self.vis_params['text_thickness'])
        
        # Apply transparency
        result = cv2.addWeighted(image, 1 - self.vis_params['alpha'], 
                                overlay, self.vis_params['alpha'], 0)
        
        return result
    
    def visualize_dataset_split(self, data_loader, split_name, save_dir):
        """
        可视化单个数据集分割（训练/验证/测试）的真值标注
        """
        split_save_dir = os.path.join(save_dir, split_name)
        os.makedirs(split_save_dir, exist_ok=True)
        
        print(f"开始可视化 {split_name} 集，保存至: {split_save_dir}")
        
        total_images = 0
        total_boxes = 0
        processed_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"Processing {split_name}")):
                try:
                    # 解包数据：[imgs_list, targets_list, img_names_list, masks_list]
                    if len(batch_data) != 4:
                        continue
                    
                    imgs_list, targets_list, img_names_list, masks_list = batch_data
                    
                    # 处理每个样本
                    for sample_idx, (img, target, img_name, mask) in enumerate(zip(imgs_list, targets_list, img_names_list, masks_list)):
                        
                        # 获取真值标注
                        if isinstance(target, dict):
                            gt_boxes = target.get('boxes', torch.empty(0, 4))
                            gt_labels = target.get('labels', None)
                            person_ids = target.get('person_ids', None)
                            position_ids = target.get('position_ids', None)
                        else:
                            continue
                        
                        # 转换boxes为CPU tensor
                        if isinstance(gt_boxes, torch.Tensor):
                            gt_boxes = gt_boxes.cpu()
                        else:
                            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
                        
                        # 跳过没有标注框的图像
                        if len(gt_boxes) == 0:
                            continue
                        
                        # 确保图像是tensor格式
                        if isinstance(img, torch.Tensor):
                            img_tensor = img.cpu()
                        else:
                            continue
                        
                        # 绘制边界框
                        vis_image = self._draw_boxes_on_image(
                            img_tensor, gt_boxes, gt_labels, person_ids, position_ids
                        )
                        
                        # 保存可视化结果
                        base_name = os.path.splitext(img_name)[0]
                        save_path = os.path.join(split_save_dir, f"{base_name}.jpg")
                        
                        success = cv2.imwrite(save_path, vis_image)
                        if success:
                            total_images += 1
                            total_boxes += len(gt_boxes)
                    
                    processed_batches += 1
                    
                    # 每处理50个批次显示一次进度
                    # if processed_batches % 50 == 0:
                    #     print(f"  {split_name}: 已处理 {processed_batches} 个批次，保存了 {total_images} 张图像")
                
                except Exception as e:
                    # 静默处理错误，继续处理下一个批次
                    continue
        
        print(f"{split_name} 集可视化完成!")
        print(f"  处理图像: {total_images}")
        print(f"  标注框数: {total_boxes}")
        print(f"  平均每张图像标注框数: {total_boxes/total_images:.2f}" if total_images > 0 else "  无有效图像")
        
        return total_images, total_boxes
    
    def visualize_all_datasets(self, dataloaders, save_dir):
        """
        可视化所有数据集分割的真值标注
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 数据集分割名称
        split_names = ['train', 'val', 'test']
        
        total_all_images = 0
        total_all_boxes = 0
        
        print("=" * 60)
        print("开始可视化整个数据集的真值标注")
        print("=" * 60)
        
        for i, (data_loader, split_name) in enumerate(zip(dataloaders, split_names)):
            print(f"\n[{i+1}/3] 处理 {split_name.upper()} 集...")
            print(f"预计批次数: {len(data_loader)}")
            
            images_count, boxes_count = self.visualize_dataset_split(
                data_loader, split_name, save_dir
            )
            
            total_all_images += images_count
            total_all_boxes += boxes_count
        
        print("\n" + "=" * 60)
        print("整个数据集可视化完成!")
        print("=" * 60)
        print(f"总处理图像: {total_all_images}")
        print(f"总标注框数: {total_all_boxes}")
        print(f"平均每张图像标注框数: {total_all_boxes/total_all_images:.2f}" if total_all_images > 0 else "无有效图像")
        print(f"结果保存在: {save_dir}")
        print("  ├── train/     (训练集可视化结果)")
        print("  ├── val/       (验证集可视化结果)")
        print("  └── test/      (测试集可视化结果)")

def main(BATCH_SIZE: int = 4, NUM_WORKERS: int = 8):
    """
    主函数：可视化整个数据集（训练集、验证集、测试集）的真值标注
    """
    data_root = "/home/s-jiang/Documents/datasets/Wildtrack2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 可视化参数配置
    vis_params = {
        'box_color': (0, 255, 0),      # 绿色边界框
        'thickness': 3,                 # 边界框线条粗细
        'alpha': 1,                  # 透明度
        'show_labels': True,           # 显示标签
        'text_color': (255, 255, 255), # 白色文字
        'text_thickness': 2,           # 文字粗细
        'text_size': 0.8               # 文字大小
    }
    
    # 创建数据加载器
    print("正在加载数据集...")
    try:
        dataloaders = create_wildtrack_dataloaders(
            data_root,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS
        )
        
        print(f"数据加载器创建成功，批次大小: {BATCH_SIZE}")
        print(f"数据集分割数量: {len(dataloaders)}")
        
        # 显示每个分割的信息
        split_names = ['train', 'val', 'test']
        for i, (loader, name) in enumerate(zip(dataloaders, split_names)):
            print(f"  {name.upper()} 集: {len(loader)} 个批次")
        
    except Exception as e:
        print(f"创建数据加载器时出错: {e}")
        return
    
    # 创建可视化器
    visualizer = GroundTruthVisualizer(
        data_root=data_root,
        device=device,
        vis_params=vis_params
    )
    
    # 设置保存目录
    save_dir = "gt_results"
    
    # 开始可视化整个数据集
    try:
        visualizer.visualize_all_datasets(dataloaders, save_dir)
    except Exception as e:
        print(f"可视化过程中出现错误: {str(e)}")
        return

if __name__ == "__main__":
    # 运行主函数，处理整个数据集（训练集、验证集、测试集）
    main(BATCH_SIZE=4, NUM_WORKERS=8)
