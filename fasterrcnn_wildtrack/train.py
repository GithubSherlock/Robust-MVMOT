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
from datasets.wildtrack2_dataloader import create_wildtrack_dataloaders

def setup_gpu_device(gpu_ids):
    if gpu_ids:
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

def train(model, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, model_type, base_dir_name):
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
    dir_name = f"{base_dir_name}_{model_type}_{counter}"
    while os.path.exists(dir_name):
        counter += 1
        dir_name = f"{base_dir_name}_{model_type}_{counter}"
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
    logging.info(f"Starting training for {model_type} model")
    logging.info('='*50)
    logging.info(f"Total epochs: {NUM_EPOCHS}")

    """Training loop"""
    for epoch in range(NUM_EPOCHS):
        logging.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        base_lr = 1e-3 # Training phase
        optimizer, lr_scheduler = yosinski_optimizer(base_lr, model)
        metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=len(train_loader), writer=writer)
        writer.add_scalar('Loss/train', metrics.loss.global_avg, epoch)
        writer.add_scalar('Cls Loss/train', metrics.loss_classifier.global_avg, epoch)
        writer.add_scalar('Reg. Loss/train', metrics.loss_box_reg.global_avg, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch > 0:
            lr_scheduler.step()

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
            lr_scheduler.step(val_loss)

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

def main(NUM_EPOCHS: int, BATCH_SIZE: int, NUM_WORKERS: int, gpu_ids: str = None, TRAIN_FINETURNING: bool = True,
        TRAIN_FREEZE: bool = False, TRAIN_BBFREEZE: bool = False,TRAIN_SUPERFREEZE: bool = False):
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
    if len(gpu_list) > 1:
        BATCH_SIZE *= len(gpu_list)
    train_loader, val_loader, test_loader = create_wildtrack_dataloaders(
        root_path="/home/s-jiang/Documents/datasets/Wildtrack2",
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    total_training_time = 0

    if TRAIN_FINETURNING: # Train feature-frozen model if requested
        start_time = time.time()
        model_fineturned = build_fasterrcnn_finetuning(num_classes=2).to(device)
        # if len(gpu_list) > 1:
        #     model_freeze = DataParallel(model_freeze)
        model_fineturned = train(model_fineturned, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, 'fineturned', 'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        freeze_duration = time.time() - start_time
        total_training_time += freeze_duration
        logging.info(f'Base model training completed in: {format_time(freeze_duration)}')

    if TRAIN_FREEZE: # Train feature-frozen model if requested
        start_time = time.time()
        model_freeze = build_fasterrcnn_freeze(num_classes=2).to(device)
        # if len(gpu_list) > 1:
        #     model_freeze = DataParallel(model_freeze)
        model_freeze = train(model_freeze, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, 'freeze', 'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        freeze_duration = time.time() - start_time
        total_training_time += freeze_duration
        logging.info(f'Freezed model training completed in: {format_time(freeze_duration)}')
    
    if TRAIN_BBFREEZE: # Train backbone-frozen model if requested
        start_time = time.time()
        model_freeze = build_fasterrcnn_bbfreeze(num_classes=2).to(device)
        # if len(gpu_list) > 1:
        #     model_freeze = DataParallel(model_freeze)
        model_freeze = train(model_freeze, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, 'bbfreeze', 'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        freeze_duration = time.time() - start_time
        total_training_time += freeze_duration
        logging.info(f'Freezed model training completed in: {format_time(freeze_duration)}')

    if TRAIN_SUPERFREEZE: # Train feature-frozen model if requested
        start_time = time.time()
        model_freeze = build_fasterrcnn_superfreeze(num_classes=2).to(device)
        # if len(gpu_list) > 1:
        #     model_freeze = DataParallel(model_freeze)
        model_freeze = train(model_freeze, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, 'superfreeze', 'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        freeze_duration = time.time() - start_time
        total_training_time += freeze_duration
        logging.info(f'Freezed model training completed in: {format_time(freeze_duration)}')
    
    logging.info('='*50)
    logging.info(f'All models trained, total training time: {format_time(total_training_time)}')
    logging.info('='*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default="5",
                      help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    main(NUM_EPOCHS=args.epochs,
         BATCH_SIZE=args.batch_size,
         NUM_WORKERS=args.num_workers,
         gpu_ids=args.gpu_ids,
         TRAIN_FINETURNING=True,
         TRAIN_FREEZE=False,
         TRAIN_BBFREEZE=False,
         TRAIN_SUPERFREEZE=False)
