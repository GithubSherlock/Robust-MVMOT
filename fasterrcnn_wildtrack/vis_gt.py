import os
import json
import torch
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torchmetrics.detection as detection_metrics
from models.backbone import _base_model_
from datasets.wildtrack2_dataloader import create_wildtrack_dataloaders
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class DetectionPredictor:
    def __init__(self, model_path, device='cuda', vis_params=None, use_pretrained=False):
        self.device = device
        self.vis_params = vis_params or {
            'box_color': (255, 0, 0),
            'thickness': 2,
            'alpha': 0.6,
            'show_labels': True,  # 控制是否显示标签
            'text_color': (255, 255, 255),  # 文本颜色(白色)
            'text_thickness': 1,  # 文本粗细
            'text_size': 0.6  # 文本大小
        }
        self.use_pretrained = use_pretrained
        self.model = self._load_model(model_path)
        self.metrics = detection_metrics.MeanAveragePrecision().to(device)
        
    def _load_model(self, model_path):
        try:
            if not self.use_pretrained:
                if not os.path.exists(model_path):
                    response = input("Local model does not exist. Is the PyTorch raw FasterRCNN model used? (y/n): ")
                    if response.lower() == 'y' or response.lower() == 'yes' or response.lower() == 'Y':
                        print("Loading the PyTorch pretrained FasterRCNN model...")
                        model = _base_model_(pretrained=True)
                        model = model.to(self.device)
                        model.eval()
                        return model
                    else:
                        raise RuntimeError("User selects termination procedure")
                
                # 加载本地模型
                model = _base_model_(pretrained=False)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
                
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict)
            else:
                # 直接使用PyTorch原始模型
                model = _base_model_(pretrained=True)
                
            model = model.to(self.device)
            model.eval()
            return model
                
        except Exception as e:
            print(f"Error message: {str(e)}")
            raise RuntimeError(f"Model loading failure: {str(e)}")

    def visualize_detection(self, image, boxes, scores, labels=None):
        image = image.cpu()
        boxes = boxes.cpu()
        
        # 转换为RGB格式
        image = F.convert_image_dtype(image, torch.uint8)
        image_np = image.permute(1, 2, 0).numpy().copy()
        
        # 转换为OpenCV格式以添加文本
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 绘制边界框和标签
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.int().tolist()
            
            # 绘制边界框
            cv2.rectangle(image_np, (x1, y1), (x2, y2), 
                        self.vis_params['box_color'], 
                        self.vis_params['thickness'])
            
            # 如果需要显示标签
            if self.vis_params['show_labels']:
                if self.use_pretrained:
                    class_name = 'person' if labels[i] == 1 else f'other_{labels[i]}'
                else:
                    class_name = 'person'
                label_text = f"{class_name}: {score:.2f}"
                
                # 计算文本大小
                (text_width, text_height), _ = cv2.getTextSize(
                    label_text, 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.vis_params['text_size'], 
                    self.vis_params['text_thickness']
                )
                
                # 绘制文本背景
                cv2.rectangle(image_np, 
                            (x1, y1 - text_height - 4),
                            (x1 + text_width, y1),
                            self.vis_params['box_color'],
                            -1)
                
                # 绘制文本
                cv2.putText(image_np, label_text,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.vis_params['text_size'],
                        self.vis_params['text_color'],
                        self.vis_params['text_thickness'])
        
        # 转换回PIL格式
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_np)

    def predict_and_evaluate(self, test_loader, save_dir):
        results_dir = Path(save_dir) / f"vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(test_loader, desc="Predicting"):
                images = [img.to(self.device) for img in images]
                predictions = self.model(images)

                # 如果使用预训练模型，过滤非人类目标
                if self.use_pretrained:
                    for pred in predictions:
                        keep_mask = pred['labels'] == 1  # COCO数据集中1表示人类
                        pred['boxes'] = pred['boxes'][keep_mask]
                        pred['labels'] = pred['labels'][keep_mask]
                        pred['scores'] = pred['scores'][keep_mask]
                
                # 保存预测结果和目标
                for pred, target, image in zip(predictions, targets, images):
                    all_predictions.append(pred)
                    all_targets.append({
                        'boxes': target['boxes'].to(self.device),
                        'labels': target['labels'].to(self.device)
                    })
                    
                    # 可视化并保存结果
                    vis_img = self.visualize_detection(
                        image,
                        pred['boxes'],
                        pred['scores']
                    )
                    
                    # 保存图像
                    img_name = f"pred_{len(all_predictions)}.png"
                    vis_img.save(results_dir / img_name)

        # 计算指标
        metrics = self.metrics(all_predictions, all_targets)
        
        # 生成报告
        self._generate_report(metrics, results_dir)
        
        return metrics

    def _generate_report(self, metrics, results_dir):
        report = {
            'map': metrics['map'].item(),
            'map_50': metrics['map_50'].item(),
            'map_75': metrics['map_75'].item(),
            'mar_1': metrics['mar_1'].item(),
            'mar_10': metrics['mar_10'].item()
        }
        
        with open(results_dir / 'report.json', 'w') as f:
            json.dump(report, f, indent=4)
            print("Report saved at:", results_dir / 'report.json')

def main():
    # 参数设置
    model_path = ""  # 替换为实际模型路径
    data_root = "/root/autodl-tmp/project/documents/datasets/Wildtrack2"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_pretrained = False # 是否使用PyTorch原始模型
    
    # 可视化参数
    vis_params = {
        'box_color': (255, 0, 0),
        'thickness': 2,
        'alpha': 0.6,
        'show_labels': True,
        'text_color': (255, 255, 255),
        'text_thickness': 1,
        'text_size': 0.6
    }
    
    # 创建数据加载器
    _, _, test_loader = create_wildtrack_dataloaders(
        data_root,
        batch_size=16,
        num_workers=16
    )
    
    # 创建预测器
    predictor = DetectionPredictor(
        model_path=model_path,
        device=device,
        vis_params=vis_params,
        use_pretrained=use_pretrained 
    )
    
    # 运行预测和评估
    save_dir = "vis_results"
    metrics = predictor.predict_and_evaluate(test_loader, save_dir)
    print("Prediction completed! The results saved at:", save_dir)
    print("mAP:", metrics['map'].item())

if __name__ == "__main__":
    main()
