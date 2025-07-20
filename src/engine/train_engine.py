"""
Refactored training engine with clean interface and reduced complexity.
Uses global configuration from YACS.
"""

import math
import sys
import time
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import get_cfg
from src.optim.mixup import Mixup
from timm.utils import ModelEma
from src import utils
from src.utils.gaze import gaze3d_to_gaze2d, compute_angular_error


class TrainingEngine:
    """
    Clean training engine that encapsulates training logic.
    Uses global configuration to avoid parameter passing.
    """
    
    def __init__(self, model: nn.Module, optimizer, loss_scaler, criterion, 
                                  model_ema=None, mixup_fn=None, log_writer=None, 
                                  device: torch.device = torch.device("cuda")):
        self.cfg = get_cfg()
        self.model = model
        self.device = device
        self.model_without_ddp = model
        
        # Initialize components
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.criterion = criterion
        self.model_ema = model_ema
        self.mixup_fn = mixup_fn
        self.log_writer = log_writer
        
        # Training state
        self.current_epoch = 0
        self.best_metric = -1e8 if self.cfg.TRAINING.VAL_METRIC not in ['loss'] else 1e8
        self.best_epoch = None
        
    def train_one_epoch(self, data_loader: DataLoader, epoch: int, 
                        lr_schedule_values=None, wd_schedule_values=None,
                        num_training_steps_per_epoch=None, start_steps=None) -> Dict[str, float]:
        """
        Train model for one epoch with clean interface.
        """
        self.model.train(True)
        self.current_epoch = epoch
        
        # Setup metrics
        metric_logger = utils.logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', utils.logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Epoch: [{epoch}]'
        print_freq = 10
        
        # Initialize gradients
        self._initialize_gradients()
        
        # Training loop
        for data_iter_step, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # Check if we should stop
            step = data_iter_step // self.cfg.TRAINING.UPDATE_FREQ
            if step >= num_training_steps_per_epoch:
                continue
                
            # Update learning rate and weight decay
            if lr_schedule_values is not None or wd_schedule_values is not None:
                self._update_lr_wd(data_iter_step, lr_schedule_values, wd_schedule_values, 
                                   start_steps, step)
            
            # Process batch
            samples, targets = self._process_batch(batch_data)
            
            # Forward pass
            loss, output = self._forward_pass(samples, targets)
            
            # Check for invalid loss
            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)
            
            # Backward pass
            grad_norm = self._backward_pass(loss, data_iter_step)
            
            # Update metrics
            self._update_metrics(metric_logger, loss, output, targets, grad_norm)
            
            # Log to tensorboard
            self._log_metrics(loss, output, targets, grad_norm)
            
        # Gather stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def _initialize_gradients(self):
        """Initialize gradients based on whether using deepspeed or not."""
        if self.loss_scaler is None:
            self.model.zero_grad()
            if hasattr(self.model, 'micro_steps'):
                self.model.micro_steps = 0
        else:
            self.optimizer.zero_grad()
    
    def _update_lr_wd(self, data_iter_step: int, lr_schedule_values, wd_schedule_values, 
                      start_steps: int, step: int):
        """Update learning rate and weight decay."""
        if data_iter_step % self.cfg.TRAINING.UPDATE_FREQ == 0:
            it = start_steps + step  # global training iteration
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
    
    def _process_batch(self, batch_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batch data and apply mixup if needed."""
        samples, targets, _, _ = batch_data
        samples = samples.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        # Ensure correct data types
        if self.cfg.DATA.DATASET_NAME == 'Gaze360':
            targets = targets.float()  # Regression task
        else:
            targets = targets.long()   # Classification task
        
        # Apply mixup if enabled
        if self.mixup_fn is not None:
            samples, targets = self.mixup_fn(samples, targets)
            
        return samples, targets
    
    def _forward_pass(self, samples: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with task-specific handling."""
        if self.loss_scaler is None:
            # DeepSpeed mode
            samples = samples.float()
            return self._forward_task_specific(samples, targets)
        else:
            # Standard mode with mixed precision
            with torch.cuda.amp.autocast():
                return self._forward_task_specific(samples, targets)
    
    def _forward_task_specific(self, samples: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Task-specific forward pass."""
        outputs = self.model(samples)
        
        if self.cfg.DATA.DATASET_NAME == 'Gaze360':
            if self.cfg.GAZE.USE_L2CS:
                # L2CS mode
                if isinstance(outputs, dict):
                    total_loss, ce_loss, mse_loss, angular_error = self.criterion(outputs, targets, self.cfg)
                    return total_loss, outputs
                else:
                    raise ValueError("L2CS mode requires model to output dict with 'pitch' and 'yaw' keys")
            else:
                # Standard gaze regression
                target_angles = gaze3d_to_gaze2d(targets)
                loss = self.criterion(outputs, target_angles)
                return loss, outputs
        else:
            # Classification task
            loss = self.criterion(outputs, targets)
            return loss, outputs
    
    def _backward_pass(self, loss: torch.Tensor, data_iter_step: int) -> Optional[float]:
        """Backward pass with gradient scaling."""
        loss_value = loss.item()
        
        if self.loss_scaler is None:
            # DeepSpeed mode
            loss /= self.cfg.TRAINING.UPDATE_FREQ
            self.model.backward(loss)
            self.model.step()
            
            if (data_iter_step + 1) % self.cfg.TRAINING.UPDATE_FREQ == 0:
                if self.model_ema is not None:
                    self.model_ema.update(self.model)
            
            grad_norm = None
            loss_scale_value = self._get_loss_scale_for_deepspeed()
        else:
            # Standard mode
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            loss /= self.cfg.TRAINING.UPDATE_FREQ
            
            grad_norm = self.loss_scaler(
                loss, self.optimizer, 
                clip_grad=self.cfg.OPTIMIZATION.CLIP_GRAD,
                parameters=self.model.parameters(), 
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % self.cfg.TRAINING.UPDATE_FREQ == 0
            )
            
            if (data_iter_step + 1) % self.cfg.TRAINING.UPDATE_FREQ == 0:
                self.optimizer.zero_grad()
                if self.model_ema is not None:
                    self.model_ema.update(self.model)
                    
            loss_scale_value = self.loss_scaler.state_dict()["scale"]
        
        torch.cuda.synchronize()
        return grad_norm
    
    def _get_loss_scale_for_deepspeed(self) -> float:
        """Get loss scale for DeepSpeed."""
        optimizer = self.model.optimizer
        return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale
    
    def _update_metrics(self, metric_logger, loss: torch.Tensor, output: torch.Tensor, 
                        targets: torch.Tensor, grad_norm: Optional[float]):
        """Update training metrics."""
        loss_value = loss.item()
        
        # Compute accuracy
        if self.mixup_fn is None:
            if self.cfg.DATA.DATASET_NAME == 'Gaze360':
                # For gaze estimation, compute angular error
                if self.cfg.GAZE.USE_L2CS:
                    # L2CS mode - angular error is computed in criterion
                    class_acc = 0.0  # Placeholder
                else:
                    # Standard regression mode
                    target_angles = gaze3d_to_gaze2d(targets)
                    angular_error = compute_angular_error(output, target_angles)
                    class_acc = angular_error
                    metric_logger.update(angular_error=angular_error)
            else:
                # Classification task
                class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
            
        # Update metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        
        # Learning rate metrics
        min_lr = 10.
        max_lr = 0.
        for group in self.optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        
        # Weight decay
        weight_decay_value = None
        for group in self.optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
    
    def _log_metrics(self, loss: torch.Tensor, output: torch.Tensor, 
                     targets: torch.Tensor, grad_norm: Optional[float]):
        """Log metrics to tensorboard."""
        if self.log_writer is None:
            return
            
        loss_value = loss.item()
        
        # Compute accuracy for logging
        if self.mixup_fn is None:
            if self.cfg.DATA.DATASET_NAME == 'Gaze360':
                if not self.cfg.GAZE.USE_L2CS:
                    target_angles = gaze3d_to_gaze2d(targets)
                    angular_error = compute_angular_error(output, target_angles)
                    self.log_writer.update(angular_error=angular_error, head="loss")
                class_acc = 0.0  # Placeholder for gaze
            else:
                class_acc = (output.max(-1)[-1] == targets).float().mean()
                self.log_writer.update(class_acc=class_acc, head="loss")
        else:
            class_acc = None
            
        # Log metrics
        self.log_writer.update(loss=loss_value, head="loss")
        if class_acc is not None:
            self.log_writer.update(class_acc=class_acc, head="loss")
        
        # Log optimization metrics
        max_lr = max(group["lr"] for group in self.optimizer.param_groups)
        min_lr = min(group["lr"] for group in self.optimizer.param_groups)
        weight_decay_value = None
        for group in self.optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
                break
        
        self.log_writer.update(lr=max_lr, head="opt")
        self.log_writer.update(min_lr=min_lr, head="opt")
        if weight_decay_value is not None:
            self.log_writer.update(weight_decay=weight_decay_value, head="opt")
        self.log_writer.update(grad_norm=grad_norm, head="opt")
        
        self.log_writer.set_step()

