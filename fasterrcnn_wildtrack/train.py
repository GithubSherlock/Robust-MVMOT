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
from torch.nn.parallel import DataParallel
from tqdm import tqdm
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
    if len(gpu_list) > 1 and torch.cuda.is_available():
        model = DataParallel(model)
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
        metrics = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=len(train_loader))

        """Log training metrics"""
        writer.add_scalar('Loss/train', metrics.loss.global_avg, epoch)
        writer.add_scalar('Cls Loss/train', metrics.loss_classifier.global_avg, epoch)
        writer.add_scalar('Reg. Loss/train', metrics.loss_box_reg.global_avg, epoch)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        if epoch > 0:
            lr_scheduler.step()

        """Validation phase"""
        val_loss, val_cls_loss, val_reg_loss = 0.0, 0.0, 0.0
        model.train()
        with torch.no_grad(): # Validation loop
            val_loop = tqdm(val_loader, desc=f'Validation Epoch [{epoch+1}/{NUM_EPOCHS}]', 
                        total=len(val_loader), leave=False)
            for i, (images, targets, _) in enumerate(val_loop):
                images = list(image.to(device) for image in images) # Move data to device
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in t.items()
                } for t in targets]
                loss_dict = model(images, targets)# Forward pass 
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item() # Accumulate losses
                val_cls_loss += loss_dict['loss_classifier'].item()
                val_reg_loss += loss_dict['loss_box_reg'].item()
                if i == len(val_loader) - 1: # Update progress bar display
                    avg_val_loss = val_loss / len(val_loader)
                    avg_val_cls = val_cls_loss / len(val_loader)
                    avg_val_reg = val_reg_loss / len(val_loader)
                    val_loop.set_postfix({
                        'avg_val_loss': f'{avg_val_loss:.4f}',
                        'cls_loss': f'{avg_val_cls:.4f}',
                        'reg_loss': f'{avg_val_reg:.4f}',
                        'batch': f'{i+1}/{len(val_loader)}'})
        avg_val_loss = val_loss / len(val_loader) # Calculate average validation losses
        avg_val_cls = val_cls_loss / len(val_loader)
        avg_val_reg = val_reg_loss / len(val_loader)
        writer.add_scalar('Loss/val', avg_val_loss, epoch) # Log validation metrics
        writer.add_scalar('Cls Loss/val', avg_val_cls, epoch)
        writer.add_scalar('Reg. Loss/val', avg_val_reg, epoch)

        coco_evaluator = evaluate(model, val_loader, global_step=epoch, device=device, writer=writer)
        coco_evaluator.summarize(global_step=epoch)
        current_map = coco_evaluator.coco_eval['bbox'].stats[0]
        logging.info(f'Validation mAP: {current_map:.4f}')
        writer.add_scalar('mAP/val', current_map, epoch)
        
        """Early stopping check"""
        _, should_stop = early_stopping(avg_val_loss, epoch)
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

    """Final evaluation"""
    finalize_training(checkpoint_manager) # Handling after training
    best_model_file = glob.glob(os.path.join(dir_name, 'best_map_*.pth'))[0] # Find the best model file
    logging.info('-'*100)
    logging.info('\nLoading best model for final evaluation...') # Load and evaluate the best model
    checkpoint_best = torch.load(best_model_file)
    model.load_state_dict(checkpoint_best['model_state_dict'])
    logging.info(f"Best model was from epoch {checkpoint_best['epoch']} with mAP {checkpoint_best['map']:.4f}")
    logging.info('\nEvaluating best model on test set...') # Evaluate the best model on a test set
    final_evaluator = evaluate(model, test_loader, global_step=NUM_EPOCHS, device=device, writer=None) # Test best model
    final_evaluator.summarize(global_step=NUM_EPOCHS)
    final_map_best = final_evaluator.coco_eval['bbox'].stats[0]
    logging.info(f'Best Model Test mAP: {final_map_best:.4f}')
    writer.add_scalar('mAP/test_best_model', final_map_best, 0) # Record the final test results
    writer.close()

    return model

def main(NUM_EPOCHS: int, BATCH_SIZE: int, NUM_WORKERS: int, gpu_ids: str = None, 
         _BASE_MODEL_: bool = True, TRAIN_FINETUNING: bool = False, TRAIN_FREEZE: bool = False):
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
    train_loader, val_loader, test_loader = create_wildtrack_dataloaders( # Create data loaders
        root_path="/home/s-jiang/Documents/datasets/Wildtrack2",
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS) # nw: 12

    total_training_time = 0

    if TRAIN_FREEZE: # Train feature-frozen model if requested
        start_time = time.time()
        model_freeze = build_fasterrcnn_freeze(num_classes=2).to(device)
        if len(gpu_list) > 1:
            model_freeze = DataParallel(model_freeze)
        model_freeze = train(model_freeze, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, 'freeze', 'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        freeze_duration = time.time() - start_time
        total_training_time += freeze_duration
        logging.info(f'Freezed model training completed in: {format_time(freeze_duration)}')
        
    if TRAIN_FINETUNING: # Train fine-tuning model if requested    
        start_time = time.time()
        model_finetuning = build_fasterrcnn_finetuning(num_classes=2).to(device)
        if len(gpu_list) > 1:
            model_finetuning = DataParallel(model_finetuning)
        model_finetuning = train(model_finetuning, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader,'finetuning', 'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        finetuning_duration = time.time() - start_time
        total_training_time += finetuning_duration
        logging.info(f'Fine-tuning model training completed in: {format_time(finetuning_duration)}')

    if _BASE_MODEL_: # Train base model if requested
        start_time = time.time()
        model_base = _base_model_(pretrained=True).to(device)
        if len(gpu_list) > 1:
            model = DataParallel(model)
        model_base = train(model_base, gpu_ids, NUM_EPOCHS, train_loader, val_loader, test_loader, 'standart', 'model')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base_duration = time.time() - start_time
        total_training_time += base_duration
        logging.info(f'Base model training completed in: {format_time(base_duration)}')
    
    logging.info('='*50)
    logging.info(f'All models trained, total training time: {format_time(total_training_time)}')
    logging.info('='*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default="5",
                      help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    
    main(NUM_EPOCHS=args.epochs,
         BATCH_SIZE=args.batch_size,
         NUM_WORKERS=args.num_workers,
         _BASE_MODEL_=False,
         TRAIN_FINETUNING=True,
         TRAIN_FREEZE=True,
         gpu_ids=args.gpu_ids)
