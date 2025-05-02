"""
Object Detection Model Training Engine
Implements training and evaluation loops for object detection models
Author: Shiqi Jiang
Date: 2025-04-25
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

# def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
#     model.train()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
#     header = f"Epoch: [{epoch}]"
#     logging.info(f"Starting training epoch {epoch}")

#     lr_scheduler = None
#     if epoch == 0:
#         warmup_factor = 1.0 / 1000
#         warmup_iters = min(1000, len(data_loader) - 1)

#         lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)

#     for images, targets in metric_logger.log_every(data_loader, print_freq, header):
#         images = list(image.to(device) for image in images)
#         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
#         with torch.cuda.amp.autocast(enabled=scaler is not None):
#             loss_dict = model(images, targets)
#             losses = sum(loss for loss in loss_dict.values())

#         # reduce losses over all GPUs for logging purposes
#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         loss_value = losses_reduced.item()

#         if not math.isfinite(loss_value):
#             logging.error(f"Loss is {loss_value}, stopping training")
#             logging.error(f"Loss dict: {loss_dict_reduced}")
#             sys.exit(1)

#         optimizer.zero_grad()
#         if scaler is not None:
#             scaler.scale(losses).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             losses.backward()
#             optimizer.step()

#         if lr_scheduler is not None:
#             lr_scheduler.step()

#         metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
#         metric_logger.update(lr=optimizer.param_groups[0]["lr"])
#     logging.info(f"Epoch [{epoch}] completed. Average loss: {metric_logger.loss.global_avg:.4f}")

#     return metric_logger

def train_one_epoch(model, optimizer, data_loader, device, epoch: int, 
                   print_freq: int, scaler=None):
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
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_iters)
    # Initialize progress bar
    pbar = tqdm(total=len(data_loader), desc=header, leave=True)
    for images, targets in data_loader:
        # Move data to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        # Forward pass with optional mixed precision
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        # Reduce losses across GPUs
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
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
        # Update learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()
        # Update metrics
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_value:.4f}', 'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'})
        pbar.update(1)
        # Detailed logging is done every print_freq iteration
        if pbar.n % print_freq == 0:
            logging.info(f"{header} [{pbar.n}/{len(data_loader)}] "
                        f"Loss: {loss_value:.4f} "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")
    pbar.close()
    logging.info(f"Epoch [{epoch}] completed. Average loss: {metric_logger.loss.global_avg:.4f}")
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


# @torch.inference_mode()
# def evaluate(model, data_loader, device):
#     n_threads = torch.get_num_threads()
#     # FIXME remove this and make paste_masks_in_image run on the GPU
#     torch.set_num_threads(1)
#     cpu_device = torch.device("cpu")
#     model.eval()
#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Test:"
#     logging.info("Starting evaluation")

#     coco = get_coco_api_from_dataset(data_loader.dataset)
#     iou_types = _get_iou_types(model)
#     coco_evaluator = CocoEvaluator(coco, iou_types)

#     for images, targets in metric_logger.log_every(data_loader, 100, header):
#         images = list(img.to(device) for img in images)

#         if torch.cuda.is_available():
#             torch.cuda.synchronize()
#         model_time = time.time()
#         outputs = model(images)

#         outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
#         model_time = time.time() - model_time

#         res = {target["image_id"]: output for target, output in zip(targets, outputs)}
#         evaluator_time = time.time()
#         coco_evaluator.update(res)
#         evaluator_time = time.time() - evaluator_time
#         metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

#     # gather the stats from all processes
#     metric_logger.synchronize_between_processes()
#     logging.info("Averaged stats: %s", metric_logger)
#     coco_evaluator.synchronize_between_processes()

#     # accumulate predictions from all images
#     coco_evaluator.accumulate()
#     coco_evaluator.summarize()
#     torch.set_num_threads(n_threads)
#     return coco_evaluator

@torch.inference_mode()
def evaluate(model, data_loader, global_step, device, writer=None):
    """
    Evaluate model performance using COCO metrics
    
    Args:
        model: PyTorch detection model
        data_loader: Validation/test data loader
        device: Evaluation device
        writer: Optional tensorboard writer
        
    Returns:
        CocoEvaluator: Evaluation results
        
    Time Complexity: O(N * B) where N = len(data_loader), B = batch_size
    Space Complexity: O(B + R) where R = results storage size
    """
    # Configure thread settings
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    # Initialize metrics and evaluator
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test"
    logging.info("Starting evaluation")
    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types, writer=writer)
    # Evaluation loop with progress bar
    pbar = tqdm(total=len(data_loader), desc=header, leave=True)
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        # Synchronize CUDA operations
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        # Model inference
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        # Update evaluator
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
         # Update metrics
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        # Update progress bar
        pbar.set_postfix({
            'model_time': f'{model_time:.4f}s',
            'evaluator_time': f'{evaluator_time:.4f}s'
        })
        pbar.update(1)
    pbar.close()
    # Synchronize and accumulate results
    metric_logger.synchronize_between_processes()
    logging.info("Averaged stats: %s", metric_logger)
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    # Restore thread settings
    torch.set_num_threads(n_threads)
    return coco_evaluator
