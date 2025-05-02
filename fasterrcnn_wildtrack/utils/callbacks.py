"""
Neural Network Training Callbacks Module
Includes checkpoint management and early stopping mechanisms
Author: Shiqi Jiang
Date: 2025-04-25
"""

import os
import shutil
import logging
import torch


class CheckpointManager:
    """
    Manages model checkpoints during training
    
    Handles saving, loading, and cleaning up model checkpoints while maintaining
    the best performing model based on mAP score.
    
    Attributes:
        base_dir (str): Base directory for saving checkpoints
        max_checkpoints (int): Maximum number of checkpoints to keep
        save_nodes (bool): Whether to save intermediate node models
        max_nodes (int): Maximum number of node models to keep
    """
    def __init__(self, base_dir: str, max_checkpoints: int = 11, save_nodes: bool = False, max_nodes: int = 3):
        """
        Initialize checkpoint manager
        
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.base_dir = base_dir
        self.checkpoints_dir = os.path.join(base_dir, 'checkpoints')
        self.max_checkpoints = max_checkpoints
        self.save_nodes = save_nodes
        self.max_nodes = max_nodes
        self.saved_checkpoints = []
        self.node_checkpoints = []
        self.last_model_path = None
        self.best_model_path = None
        self.best_map = float('-inf')
        os.makedirs(self.checkpoints_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, lr_scheduler, epoch: int, mAP: float) -> str:
        """
        Save model checkpoint and return save path
        
        Args:
            model: PyTorch model
            optimizer: Model optimizer
            lr_scheduler: Learning rate scheduler
            epoch: Current epoch number
            mAP: Mean Average Precision score
            
        Returns:
            str: Path where checkpoint was saved
            
        Time Complexity: O(model_size) - dominated by model state dict saving
        Space Complexity: O(model_size)
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'map': mAP
        }
        checkpoint_name = f'checkpoint_epoch_{epoch}_map_{mAP:.4f}.pth'
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        
        torch.save(checkpoint, checkpoint_path)
        logging.info(f'Saved checkpoint: {checkpoint_name}')
        
        return checkpoint_path
    
    def update_checkpoints(self, checkpoint_path: str, current_map: float):
        """
        Update checkpoint list and manage storage
        
        Args:
            checkpoint_path: Path to new checkpoint
            current_map: Current mAP score
            
        Time Complexity: O(max_checkpoints) for cleanup
        Space Complexity: O(1)
        """
        self.last_model_path = checkpoint_path
        if current_map > self.best_map: # Update best model if current mAP is higher
            self.best_map = current_map
            self.best_model_path = checkpoint_path
            logging.info(f'New best model with mAP: {current_map:.4f}')
        
        self.saved_checkpoints.append(checkpoint_path)
        # Keep only recent checkpoints while preserving best model
        while len(self.saved_checkpoints) > self.max_checkpoints:
            old_path = self.saved_checkpoints.pop(0)
            if os.path.exists(old_path) and old_path != self.best_model_path:
                os.remove(old_path)
        
        if self.save_nodes: # Manage node checkpoints if enabled
            self.node_checkpoints.append(checkpoint_path)
            # 仅保留最新的max_nodes个节点模型
            while len(self.node_checkpoints) > self.max_nodes:
                old_node = self.node_checkpoints.pop(0)
                if os.path.exists(old_node):
                    os.remove(old_node)


class EarlyStopping:
    """
    Early stopping mechanism based on validation loss
    
    Implements Prechelt's early stopping criteria using validation loss
    and generalization loss threshold.

    References:
    - Prechelt, L. (2012). "Early stopping — but when?" Neural Networks: Tricks of the Trade, 53-67.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0,
                 improvement_threshold: float = 5.0, warmup_epochs: int = 0,
                 smoothing: bool = False):
        """
        Initialize early stopping mechanism based on verification loss
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in loss to qualify as improvement
            improvement_threshold: Generalization loss threshold (%), refer to Prechelt's paper
            warmup_epochs: Number of initial epochs to ignore
            smoothing: Whether to use moving average for loss smoothing
            
        Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.improvement_threshold = improvement_threshold
        self.warmup_epochs = warmup_epochs
        self.smoothing = smoothing
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.loss_history = []

    def __call__(self, current_loss: float, epoch: int) -> tuple[bool, bool]:
        """
        Check early stopping conditions
        
        Args:
            current_loss: Current validation loss
            epoch: Current epoch number
            
        Returns:
            tuple: (is_improved, should_stop)
            
        Time Complexity: O(1) or O(window_size) with smoothing
        Space Complexity: O(window_size) with smoothing
        """
        if epoch < self.warmup_epochs:
            return False, False
        if self.smoothing: # Apply loss smoothing if enabled
            self.loss_history.append(current_loss)
            window_size = 5
            if len(self.loss_history) >= window_size:
                smoothed_loss = sum(self.loss_history[-window_size:]) / window_size
            else:
                smoothed_loss = current_loss
        else:
            smoothed_loss = current_loss

        improved = False # Check for improvement
        if (self.best_loss - smoothed_loss) > self.min_delta:
            self.best_loss = smoothed_loss
            self.best_epoch = epoch
            self.counter = 0
            improved = True
        else:
            self.counter += 1

        # Calculate generalization loss(GL)，refer to Prechelt's paper
        gl = 100 * (smoothed_loss / self.best_loss - 1) if self.best_loss != 0 else 0

        # Check stopping criteria
        if self.counter >= self.patience and gl > self.improvement_threshold: 
            self.early_stop = True
            logging.info(f"Early stopping triggered at epoch {epoch}. "
                        f"Best loss: {self.best_loss:.4f} (epoch {self.best_epoch}), "
                        f"GL: {gl:.2f}%")

        return improved, self.early_stop


def finalize_training(checkpoint_manager: CheckpointManager):
    """
    Perform cleanup after training completion
    
    Args:
        checkpoint_manager: Checkpoint manager instance
        
    Time Complexity: O(model_size) for copying best model
    Space Complexity: O(model_size)
    """
    if checkpoint_manager.best_model_path: # Copy the best model only if it exists
        best_checkpoint = torch.load(checkpoint_manager.best_model_path)
        best_map = best_checkpoint['map']
        new_best_name = f'best_map_{best_map:.4f}.pth'
        new_best_path = os.path.join(checkpoint_manager.base_dir, new_best_name)
        shutil.copy(checkpoint_manager.best_model_path, new_best_path)
        logging.info(f'Best model copied to: {new_best_name}')  
    if not checkpoint_manager.save_nodes:
        shutil.rmtree(checkpoint_manager.checkpoints_dir)
        logging.info('Checkpoints directory removed')
    else:
        logging.info('Checkpoints directory retained for node models')
