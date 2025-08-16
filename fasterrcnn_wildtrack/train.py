"""
Object Detection Model Training Script
Supports base model training, fine-tuning and feature freezing modes
Author: Shiqi Jiang
Date: 2025-04-25
"""


import os
import glob
import time
import torch
import logging
import argparse
from torch.utils.tensorboard import SummaryWriter
from models import *
from utils.utils import *
from utils.engine import train_one_epoch, evaluate
from utils.callbacks import CheckpointManager, EarlyStopping, finalize_training
from datasets.wildtrack2_dataloader import DataLoader
from datasets.wildtrack2_datasets import WildtrackDataset
from datasets.mot_datasets import MOTDataset

def setup_gpu_device(gpu_ids):
    if gpu_ids:
        if ',' not in gpu_ids:
            device = torch.device(f"cuda:{gpu_ids}")
            return device, [int(gpu_ids)]
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
            device = torch.device("cuda:0")
            gpu_list = [i for i in range(len(gpu_ids.split(',')))]
            return device, gpu_list
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), []

def format_time(seconds: float) -> str:
    """
    Convert seconds to human readable time format (HH:MM:SS)
    
    Args:
        seconds: Number of seconds to convert
    Returns:
        str: Formatted time string
    Time Complexity: O(1)
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60) 
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def train(model, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, WEIGHTS, BASE_LR, base_dir_name):
    """
    Main training function for object detection models
    
    Args:
        model: Model to train
        device: Training device (CPU/GPU)
        NUM_EPOCHS: Number of training epochs
        train_loader: Training data loader
        val_loader: Validation data loader 
        test_loader: Test data loader
        model_type: Type of model ('standard'/'finetuning'/'freeze')
        base_dir_name: Base directory for saving outputs
        
    Returns:
        model: Trained model
        
    Time Complexity: O(NUM_EPOCHS * (N_train + N_val))
    Space Complexity: O(model_size + batch_size)
    """
    torch.cuda.empty_cache()
    # Create unique output directory
    counter = 0
    dir_name = f"{base_dir_name}_{WEIGHTS}_{counter}"
    while os.path.exists(dir_name):
        counter += 1
        dir_name = f"{base_dir_name}_{WEIGHTS}_{counter}"
    target_dir = os.path.join(dir_name, 'checkpoints')
    os.makedirs(target_dir, exist_ok=True)

    """Configure logging"""
    logging.root.handlers = []  # Clear existing handlers
    log_file = os.path.join(dir_name, 'trainval.log') # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.root.addHandler(file_handler)
    console_handler = logging.StreamHandler() # Setup console handler
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logging.root.addHandler(console_handler)
    logging.root.setLevel(logging.INFO)

    device, gpu_list = setup_gpu_device(gpu_ids)
    # if len(gpu_list) > 1 and torch.cuda.is_available():
    #     model = DataParallel(model)
    logging.info(f'Using device: {device}')
    logging.info(f'GPU IDs: {gpu_ids if gpu_ids else "Not specified"}')
    logging.info(f'Number of GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')
    logging.info(f"Created directory: {target_dir}")

    """Initialize tensorboard writer and checkpoint manager"""
    writer = SummaryWriter(log_dir=dir_name)
    checkpoint_manager = CheckpointManager(dir_name, max_checkpoints=11, save_nodes=False, max_nodes=3)
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, improvement_threshold=5.0, # 5% generalisation loss allowed
                                   warmup_epochs=5, smoothing=True)
    logging.info('='*50)
    logging.info(f"Starting training for {WEIGHTS} model")
    logging.info('='*50)
    logging.info(f"Total epochs: {NUM_EPOCHS}")
    logging.info(f"Base learning rate: {BASE_LR}")

    """Initialize Optimizer, please check avaliable scheduler in model/optimizer.py"""
    optimizer, lr_scheduler = yosinski_optimizer(BASE_LR, model, NUM_EPOCHS, scheduler="CosineAnnealingLR")

    """Training loop"""
    for epoch in range(NUM_EPOCHS):
        logging.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        metrics = train_one_epoch(model, optimizer, lr_scheduler, train_loader, device, epoch, print_freq=len(train_loader), writer=writer)
        writer.add_scalar('Loss/train', metrics.loss.global_avg, epoch)
        writer.add_scalar('Cls Loss/train', metrics.loss_classifier.global_avg, epoch)
        writer.add_scalar('Reg. Loss/train', metrics.loss_box_reg.global_avg, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)

        """Validation phase"""
        coco_evaluator, val_metric_logger = evaluate(model, val_loader, device=device, global_step=epoch, writer=writer)
        coco_evaluator.summarize(global_step=epoch)
        val_loss = val_metric_logger.loss.global_avg
        val_cls_loss = val_metric_logger.loss_classifier.global_avg  
        val_reg_loss = val_metric_logger.loss_box_reg.global_avg
        current_map = coco_evaluator.coco_eval['bbox'].stats[0]
        logging.info(
            f"Epoch {epoch+1} validation: "
            f"Loss: {val_loss:.4f}, "
            f"Cls Loss: {val_cls_loss:.4f}, "
            f"Reg Loss: {val_reg_loss:.4f}, "
            f"mAP: {current_map:.4f}")
        writer.add_scalar('mAP/val', current_map, epoch)

        """Early stopping check"""
        _, should_stop = early_stopping(val_loss, epoch)
        if current_map > checkpoint_manager.best_map:
            checkpoint_path = checkpoint_manager.save_checkpoint(
                model, optimizer, lr_scheduler, epoch, current_map)
            checkpoint_manager.update_checkpoints(checkpoint_path, current_map)
            logging.info(f'mAP improved to {current_map:.4f}, checkpoint saved at epoch {epoch+1}')
        if should_stop:
            logging.info(f'Early stopping triggered at epoch {epoch+1}')
            logging.info(f'Best validation loss: {early_stopping.best_loss:.4f} '
                        f'at epoch {early_stopping.best_epoch+1}')
            break
        # Update learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

    """Final Test Phase"""
    finalize_training(checkpoint_manager) # Handling after training
    best_model_file = glob.glob(os.path.join(dir_name, 'best_map_*.pth'))[0] # Find the best model file
    logging.info('-'*100)
    logging.info('\nLoading best model for final evaluation...') # Load and evaluate the best model
    checkpoint_best = torch.load(best_model_file)
    model.load_state_dict(checkpoint_best['model_state_dict'])
    logging.info(f"Best model was from epoch {checkpoint_best['epoch']} with mAP {checkpoint_best['map']:.4f}")
    logging.info('\nEvaluating best model on test set...') # Evaluate the best model on a test set
    final_evaluator, _ = evaluate(model, test_loader, global_step=NUM_EPOCHS, device=device, writer=None) # Test best model
    final_evaluator.summarize(global_step=NUM_EPOCHS)
    final_map_best = final_evaluator.coco_eval['bbox'].stats[0]
    logging.info(f'Best Model Test mAP: {final_map_best:.4f}')
    writer.add_scalar('mAP/test_best_model', final_map_best, 0) # Record the final test results
    writer.close()
    return model

def main(NUM_EPOCHS: int, BATCH_SIZE: int, NUM_WORKERS: int, NUM_CLASSES: int, WEIGHTS: str, BASE_LR: float, 
         gpu_ids: str, TRAIN: bool, PRETRAINED: bool = True):
    """
    Main entry point for training different model configurations
    
    Args:
        NUM_EPOCHS: Number of training epochs
        _BASE_MODEL_: Whether to train base model
        TRAIN_FINETUNING: Whether to train fine-tuning model
        TRAIN_FREEZE: Whether to train feature-frozen model
        
    Time Complexity: O(NUM_EPOCHS * num_models * (N_train + N_val))
    Space Complexity: O(model_size + batch_size)
    """
    device, gpu_list = setup_gpu_device(gpu_ids)
    # if len(gpu_list) > 1:
    #     BATCH_SIZE *= len(gpu_list)
    dataloader = DataLoader(
        root_path="/home/s-jiang/Documents/datasets/Wildtrack2",  # 数据集根目录
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        data_aug_conf=None,  # 使用默认配置
        dataset_class=WildtrackDataset,   # 使用默认的WildtrackDataset MOTDataset
        datasets=[],    # 或者 ['MOT16', 'MOT17', 'MOT20']
        split=''        # 'train' 或 'test'
    )
    dataloader.print_dataset_info()
    train_loader, val_loader, test_loader = dataloader.get_dataloaders()

    total_training_time = 0
    while TRAIN: # Train feature-frozen model if requested
        start_time = time.time()
        model_freeze = build_fasterrcnn_model(weights_name=WEIGHTS, num_classes=NUM_CLASSES, pretrained=PRETRAINED).to(device)
        logging.info(f"Backbone: {WEIGHTS}")
        logging.info(f"Number of classes: {NUM_CLASSES}")
        logging.info(f"Pretrained: {PRETRAINED}")
        # if len(gpu_list) > 1:
        #     model_freeze = DataParallel(model_freeze)
        model_freeze = train(model_freeze, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, WEIGHTS, BASE_LR,'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        duration = time.time() - start_time
        total_training_time += duration
        logging.info(f'Your model training completed in: {format_time(duration)}')
        TRAIN = False
    
    logging.info('='*50)
    logging.info(f'All models trained, total training time: {format_time(total_training_time)}')
    logging.info('='*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default="3",
                      help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training")
    parser.add_argument('--bz', type=int, default=2, help="Batch size for training")
    parser.add_argument('--nw', type=int, default=8, help="Number of workers for training")
    parser.add_argument('--weights', type=str, default="fasterrcnn_resnet50_fpn",
                      help='Backbone weights to use (e.g., "resnet50"), please check models/backbone.py for more options')
    parser.add_argument('--num_classes', type=int, default=2,
                      help='Number of classes (default is 2 for background + person)')
    parser.add_argument('--lr', type=float, default=1e-3, help="""Base learning rate for training""")
    parser.add_argument('--train', type=bool, default=True, help="Training or not")
    args = parser.parse_args()
    
    main(NUM_EPOCHS=args.epochs,
         BATCH_SIZE=args.bz,
         NUM_WORKERS=args.nw,
         gpu_ids=args.gpu_ids,
         WEIGHTS=args.weights,
         NUM_CLASSES=args.num_classes,
         BASE_LR=args.lr,
         TRAIN=args.train)
