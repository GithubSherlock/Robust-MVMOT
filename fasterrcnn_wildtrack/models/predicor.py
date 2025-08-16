import os
import cv2
import csv
import json
import torch
import torchmetrics.detection as detection_metrics
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
from torchvision.io import read_image
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from models.backbone import _base_model_, build_fasterrcnn_model

class DetectionPredictor:
    def __init__(self, data_root, model_path, device='cuda', vis_params=None, use_pretrained=False, 
                subset_ratio=1.0, noise_factor=0.0):
        self.device = device
        self.data_root = data_root
        self.subset_ratio = subset_ratio
        self.noise_factor = noise_factor
        self.vis_params = vis_params or {
            'box_color': (0, 255, 0),
            'thickness': 2,
            'alpha': 1,
            'show_labels': True,
            'text_color': (255, 255, 255),
            'text_thickness': 1,
            'text_size': 0.6
        }
        self.use_pretrained = use_pretrained
        self.actual_use_pretrained = use_pretrained
        self.model_path = model_path
        self.model = self._load_model(model_path)
        self.metrics = detection_metrics.MeanAveragePrecision().to(device)
    
    def _add_noise(self, images):
        if self.noise_factor > 0:
            noise = torch.randn_like(images) * self.noise_factor
            noisy_images = images + noise
            return torch.clamp(noisy_images, 0, 1)
        return images
    
    def _filter_by_mask(self, prediction, mask, img_size):
        """
        Filtering the prediction bounding boxes by mask
        Args:
            prediction: model prediction result
            mask: corresponding mask (H, W) or (1, H, W)
            img_size: size of the original image (H, W)
        """
        boxes = prediction['boxes']
        if len(boxes) == 0:
            return prediction
        try:
            if mask.dim() == 3: # Process the mask dim (1, H, W)
                mask = mask.squeeze(0)
            mask = mask.bool() # Comfirm mask is boolean
            if mask.shape != img_size: # Comfirm mask shape
                print(f"Resizing mask from {mask.shape} to {img_size}")
                mask = F.resize(mask.float().unsqueeze(0), img_size).squeeze(0) > 0.5
            
            boxes_int = boxes.round().long() # transform the box coordinates to int
            centers = torch.zeros((len(boxes), 2), device=boxes.device, dtype=torch.long) # Compute the center of all bboxes
            centers[:, 0] = ((boxes_int[:, 0] + boxes_int[:, 2]) / 2).clamp(0, img_size[1] - 1)  # x center
            # centers[:, 1] = ((boxes_int[:, 1] + boxes_int[:, 3]) / 2).clamp(0, img_size[0] - 1)  # y center
            centers[:, 1] = ((boxes_int[:, 3])).clamp(0, img_size[0] - 1)  # ymax
            
            valid_x = (centers[:, 0] >= 0) & (centers[:, 0] < mask.shape[1]) # Verify that the index is in the valid range
            valid_y = (centers[:, 1] >= 0) & (centers[:, 1] < mask.shape[0])
            valid_indices = valid_x & valid_y
            
            if not valid_indices.all():
                print("Warning: Some box centers are outside the mask boundaries")
                centers = centers[valid_indices]
                boxes = boxes[valid_indices]
                prediction['scores'] = prediction['scores'][valid_indices]
                prediction['labels'] = prediction['labels'][valid_indices]
                
            if len(centers) == 0:
                print("Warning: No valid boxes after boundary check")
                prediction['boxes'] = boxes.new_zeros((0, 4))
                prediction['scores'] = boxes.new_zeros(0)
                prediction['labels'] = boxes.new_zeros(0, dtype=torch.long)
                return prediction
            
            valid_mask = mask[centers[:, 1], centers[:, 0]] # Get mask values at the center points
            prediction['boxes'] = boxes[valid_mask] # Filtering the bboxes based on the mask
            prediction['scores'] = prediction['scores'][valid_mask]
            prediction['labels'] = prediction['labels'][valid_mask]
            
        except Exception as e:
            print(f"Error in _filter_by_mask: {str(e)}")
            print(f"Mask shape: {mask.shape}, Image size: {img_size}")
            print(f"Boxes shape: {boxes.shape}")
            if 'centers' in locals():
                print(f"Centers shape: {centers.shape}")
            return prediction
        
        return prediction
        
    def _load_model(self, model_path):
        try:
            if not self.use_pretrained:
                if not os.path.exists(model_path):
                    response = input("Local model does not exist. Is the PyTorch raw FasterRCNN model used? (y/n): ")
                    if response.lower() == 'y' or response.lower() == 'yes':
                        print("Loading the PyTorch pretrained FasterRCNN model...")
                        self.actual_use_pretrained = True
                        model = _base_model_(pretrained=True)
                        model.roi_heads.nms_thresh = 0.5 # Set the IoU threshold for NMS to 0.5
                        model = model.to(self.device)
                        model.eval()
                        return model
                    else:
                        raise RuntimeError("User selects termination procedure")
                
                # Load the model from the specified path
                model = _base_model_(pretrained=False)
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
                # model = build_fasterrcnn_model(weights_name="fasterrcnn_resnet50_fpn", num_classes=2, pretrained=True)
                model.roi_heads.nms_thresh = 0.5
                checkpoint = torch.load(model_path, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                model.load_state_dict(state_dict)
            else:
                model = _base_model_(pretrained=True)
                model.roi_heads.nms_thresh = 0.5 # Set the IoU threshold for NMS to 0.5
            
            model = model.to(self.device)
            model.eval()
            return model
                
        except Exception as e:
            print(f"Error message: {str(e)}")
            raise RuntimeError(f"Model loading failure: {str(e)}")

    def visualize_detection(self, image, boxes, scores, labels=None):
        image = image.cpu()
        boxes = boxes.cpu()
        if labels is not None:
            labels = labels.cpu()
        image = F.convert_image_dtype(image, torch.uint8)
        image_np = image.permute(1, 2, 0).numpy().copy()
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(image_np, (x1, y1), (x2, y2), # Drawing the bounding box
                        self.vis_params['box_color'], 
                        self.vis_params['thickness'])
            if self.vis_params['show_labels']: # Drawing bounding boxes and labels
                if self.actual_use_pretrained and labels is not None:
                    class_name = 'person' if labels[i] == 1 else f'other_{labels[i]}'
                else:
                    class_name = 'person'
                label_text = f"{class_name}: {score:.2f}"
                
                (text_width, text_height), _ = cv2.getTextSize( # Calculate text size
                    label_text, 
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.vis_params['text_size'], 
                    self.vis_params['text_thickness'])
                cv2.rectangle(image_np, # Drawing a text background
                            (x1, y1 - text_height - 4),
                            (x1 + text_width, y1),
                            self.vis_params['box_color'],
                            -1)
                cv2.putText(image_np, label_text, # Drawing the text
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.vis_params['text_size'],
                        self.vis_params['text_color'],
                        self.vis_params['text_thickness'])
        
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_np)

    def predict_and_evaluate(self, test_loader, save_dir):
        results_dir = Path(save_dir) / f"vis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        all_predictions = []
        all_targets = []
        skipped_count = 0
        
        with torch.no_grad():
            for images, targets, img_names, masks in tqdm(test_loader, desc="Predicting"):
                try:
                    valid_indices = [i for i, target in enumerate(targets) if len(target['boxes']) > 0]
                    if not valid_indices:  # 如果这个batch中所有样本都没有标注
                        skipped_count += len(targets)
                        continue

                    images = [self._add_noise(img.to(self.device)) for img in images]
                    masks = [mask.to(self.device) for mask in masks]
                    valid_targets = [targets[i] for i in valid_indices]
                    valid_img_names = [img_names[i] for i in valid_indices]
                    predictions = self.model(images)

                    # Precess each image in the batch
                    for i, (pred, image, img_name, mask) in enumerate(zip(predictions, images, valid_img_names, masks)):
                        try:
                            pred = self._filter_by_mask(pred, mask, image.shape[1:]) # Apply mask filtering
                            confidence_mask = pred['scores'] >= 0.5 # Apply confidence and class filtering
                            class_mask = pred['labels'] == 1
                            keep_mask = confidence_mask & class_mask
                            pred = {k: v[keep_mask] for k, v in pred.items()}
                            
                            all_predictions.append(pred) # Save the prediction results
                            all_targets.append({
                                'boxes': valid_targets[i]['boxes'].to(self.device),
                                'labels': valid_targets[i]['labels'].to(self.device)})
                            
                            vis_img = self.visualize_detection( # Visualize and save images
                                image,
                                pred['boxes'],
                                pred['scores'],
                                pred['labels'])
                            vis_img.save(results_dir / img_name)
                            
                        except Exception as e:
                            print(f"Error processing image {img_name}: {str(e)}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue
            print(f"Skipped {skipped_count} images with no annotations")

            try: # Compute metrics and save the report
                metrics = self.metrics(all_predictions, all_targets)
                self._generate_report(metrics, results_dir)
                self._log_to_csv(metrics, self.model_path, self.data_root)
                return metrics
            except Exception as e:
                print(f"Error calculating metrics: {str(e)}")
                return None

    def _generate_report(self, metrics, results_dir):
        report = {
            'map': metrics['map'].item(),
            'map_50': metrics['map_50'].item(),
            'map_75': metrics['map_75'].item(),
            'map_small': metrics['map_small'].item(),
            'map_medium': metrics['map_medium'].item(),
            'map_large': metrics['map_large'].item(),
            'mar_1': metrics['mar_1'].item(),
            'mar_10': metrics['mar_10'].item(),
            'mar_100': metrics['mar_100'].item(),
            'mar_small': metrics['mar_small'].item(),
            'mar_medium': metrics['mar_medium'].item(),
            'mar_large': metrics['mar_large'].item()
        }
        
        with open(results_dir / 'report.json', 'w') as f:
            json.dump(report, f, indent=4)
            print("Report saved at:", results_dir / 'report.json')

    def _log_to_csv(self, metrics, model_path, data_root):
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        dataset_name = os.path.basename(data_root)
        
        # model_dir = os.path.dirname(model_path)
        if self.actual_use_pretrained: #  or self.use_pretrained
            backbone_type = "resnet50_fpn"
            model_name = "PyTorch_fasterrcnn_resnet50_fpn"
        elif 'model_weights' in str(model_path):
            parent_dir = Path(model_path).parent.name  # A example: 'model_finetuning_0_bs2_nw8'
            if parent_dir.startswith('model_'):
                backbone_type = parent_dir.split('_')[1]  # extact 'finetuning'
            else:
                backbone_type = "Unknown"
            model_name = Path(model_path).name
        else:
            backbone_type = "Unknown"
            model_name = "Unknown"
        

        csv_path = "CSVLogger.csv"
        file_exists = os.path.exists(csv_path)
        
        row_data = [
            current_time,
            dataset_name,
            str(self.actual_use_pretrained),
            backbone_type,
            model_name,
            f"{metrics['map'].item():.4f}",
            f"{metrics['map_50'].item():.4f}",
            f"{metrics['map_75'].item():.4f}",
            f"{metrics['map_small'].item():.4f}",
            f"{metrics['map_medium'].item():.4f}",
            f"{metrics['map_large'].item():.4f}",
            f"{metrics['mar_1'].item():.4f}",
            f"{metrics['mar_10'].item():.4f}",
            f"{metrics['mar_100'].item():.4f}",
            f"{metrics['mar_small'].item():.4f}",
            f"{metrics['mar_medium'].item():.4f}",
            f"{metrics['mar_large'].item():.4f}"
        ]
        
        mode = 'a' if file_exists else 'w' # write into CSV logger file
        with open(csv_path, mode, newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Date', 'Dataset', 'Pretrained', 'Backbone', 'Model',
                    'mAP', 'mAP_50', 'mAP_75', 'mAP_small', 'mAP_medium', 'mAP_large',
                    'mAR_1', 'mAR_10', 'mAR_100', 'mAR_small', 'mAR_medium', 'mAR_large'])
            writer.writerow(row_data)