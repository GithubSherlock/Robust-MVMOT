"""
Object Detection Model Training Engine
Implements training and evaluation loops for object detection models
Author: Shiqi Jiang
Date: 2025-05-22
"""


import math
import sys
import time
import logging
import torch
import torchvision.models.detection.mask_rcnn
from tqdm import tqdm
from utils import utils
from utils.coco_eval import CocoEvaluator
from utils.coco_utils import get_coco_api_from_dataset

def train_one_epoch(model, optimizer, lr_scheduler, data_loader, device,
                    epoch: int, print_freq: int, writer=None, scaler=None):
    """
    Train model for one epoch with progress tracking and logging
    
    Args:
        model: PyTorch detection model
        optimizer: Model optimizer
        data_loader: Training data loader
        device: Training device (CPU/GPU)
        epoch: Current epoch number
        print_freq: Frequency of progress logging
        scaler: Gradient scaler for mixed precision training
        
    Returns:
        MetricLogger: Training metrics
        
    Time Complexity: O(N * B) where N = len(data_loader), B = batch_size
    Space Complexity: O(B + M) where M = model size
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"
    logging.info(f"Starting training epoch {epoch}")

    # Configure learning rate warmup for first epoch
    # lr_scheduler = None
    # if epoch == 0:
    #     warmup_factor = 1.0 / 1000
    #     warmup_iters = min(1000, len(data_loader) - 1)
    #     lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    # Initialize progress bar
    pbar = tqdm(total=len(data_loader), desc=header, leave=True)
    for images, targets, _, masks in data_loader:
        images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # masks = list(mask.to(device) for mask in masks)
        for target, mask in zip(targets, masks):
            target["mask"] = mask
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        ############################################
        # Forward pass with optional mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        ############################################
        # Reduce losses across GPUs
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        # Update all metrics
        metric_logger.update(
            loss=losses_reduced,
            loss_classifier=loss_dict_reduced['loss_classifier'],
            loss_box_reg=loss_dict_reduced['loss_box_reg'],
            lr=optimizer.param_groups[0]["lr"])

        # Check for invalid loss
        if not math.isfinite(loss_value):
            logging.error(f"Loss is {loss_value}, stopping training")
            logging.error(f"Loss dict: {loss_dict_reduced}")
            sys.exit(1)
        # Backward pass and optimization
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        pbar.update(1)
        # Detailed logging
        if pbar.n % print_freq == 0:
            logging.info(f"{header} [{pbar.n}/{len(data_loader)}] "
                        f"Loss: {loss_value:.4f} "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    pbar.close()
    logging.info(f"Epoch [{epoch}] completed. Average loss: {metric_logger.loss.global_avg:.4f}")

    if writer:
        writer.add_scalar('Loss/train', metric_logger.loss.global_avg, epoch)
        writer.add_scalar('Cls Loss/train', metric_logger.loss_classifier.global_avg, epoch)
        writer.add_scalar('Reg. Loss/train', metric_logger.loss_box_reg.global_avg, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    if epoch > 0:
            lr_scheduler.step()
    return metric_logger


def _get_iou_types(model):
    """
    Determine IoU types based on model architecture
    
    Args:
        model: Detection model
        
    Returns:
        list: IoU types to evaluate
        
    Time Complexity: O(1)
    Space Complexity: O(1)
    """
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types

@torch.inference_mode()
def evaluate(model, data_loader, device, global_step=None, writer=None):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    val_loss, val_cls_loss, val_reg_loss = 0.0, 0.0, 0.0 # Initialize loss accumulators
    metric_logger = utils.MetricLogger(delimiter="  ") # Initialize metric logger
    if len(data_loader) == 0:
        logging.warning("Validation dataset is empty!")
        return None, metric_logger
    header = "Validation" if writer else "Test"
    logging.info("Starting evaluation ...")
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, writer=writer)

    """Evaluation loop with progress bar"""
    pbar = tqdm(total=len(data_loader), desc=header, leave=True)
    for i, (images, targets, _, masks) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        # targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # masks = list(mask.to(device) for mask in masks)
        for target, mask in zip(targets, masks):
            target["mask"] = mask
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        ############################################
        model.eval()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model.train()
        loss_dict = model(images, targets) # Compute loss
        model.eval()
        ############################################
        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()
        val_cls_loss += loss_dict['loss_classifier'].item()
        val_reg_loss += loss_dict['loss_box_reg'].item()
        metric_logger.update( # Update metrics
            loss=losses.item(),
            loss_classifier=loss_dict['loss_classifier'].item(),
            loss_box_reg=loss_dict['loss_box_reg'].item())
        
        if i == len(data_loader) - 1: # Update progress bar
            avg_val_loss = val_loss / len(data_loader)
            avg_val_cls = val_cls_loss / len(data_loader)
            avg_val_reg = val_reg_loss / len(data_loader)
            pbar.set_postfix({
                'avg_val_loss': f'{avg_val_loss:.4f}',
                'cls_loss': f'{avg_val_cls:.4f}',
                'reg_loss': f'{avg_val_reg:.4f}',
                'batch': f'{i+1}/{len(data_loader)}'})
        res = {target["image_id"]: output for target, output # Update evaluator
               in zip(targets, outputs) if "image_id" in target}
        if res:
            coco_evaluator.update(res)
        else:
            raise KeyError("Warning: Empty detection results for current batch")
        pbar.update(1)
    pbar.close()

    metric_logger.synchronize_between_processes() # Synchronize and accumulate results
    logging.info("Averaged stats: %s", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    
    avg_val_loss = val_loss / len(data_loader) # Compute average losses
    avg_val_cls = val_cls_loss / len(data_loader)
    avg_val_reg = val_reg_loss / len(data_loader)
    if writer and global_step is not None: # Log metrics to tensorboard
        writer.add_scalar('Loss/val', avg_val_loss, global_step)
        writer.add_scalar('Cls Loss/val', avg_val_cls, global_step)
        writer.add_scalar('Reg. Loss/val', avg_val_reg, global_step)
    torch.set_num_threads(n_threads) # Restore thread settings
    return coco_evaluator, metric_logger
