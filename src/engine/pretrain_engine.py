"""
Pretraining engine with clean interface for VideoMAE pretraining.
Uses global configuration from YACS.
"""
import math
import sys
from typing import Optional, Dict, Any, Tuple, Iterable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from einops import rearrange, repeat
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.utils.config import get_cfg
from src.utils import logger
from src import utils


class PretrainEngine:
    """
    Clean pretraining engine that encapsulates VideoMAE pretraining logic.
    Uses global configuration to avoid parameter passing.
    """
    
    def __init__(self, model: nn.Module, optimizer, loss_scaler, 
                 log_writer=None, device: torch.device = torch.device("cuda")):
        self.cfg = get_cfg()
        self.model = model
        self.device = device
        self.model_without_ddp = model
        
        # Initialize components
        self.optimizer = optimizer
        self.loss_scaler = loss_scaler
        self.log_writer = log_writer
        
        # Training state
        self.current_epoch = 0
        
        # Loss functions
        self.loss_func = nn.MSELoss()
        if (self.cfg.PRETRAINING.USE_FRAME_DIFF_AS_TARGET and 
            self.cfg.PRETRAINING.TARGET_DIFF_WEIGHT is not None):
            self.loss_func_diff = nn.MSELoss()
        
    def train_one_epoch(self, data_loader: DataLoader, epoch: int, 
                        lr_schedule_values=None, wd_schedule_values=None,
                        start_steps=None, patch_size: int = 16) -> Dict[str, float]:
        """
        Train model for one epoch with clean interface.
        """
        self.model.train()
        self.current_epoch = epoch
        
        # Setup metrics
        metric_logger = logger.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', logger.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = f'Epoch: [{epoch}]'
        print_freq = 10
        
        # Training loop
        for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # Update learning rate and weight decay
            it = start_steps + step  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                self._update_lr_wd(it, lr_schedule_values, wd_schedule_values)
            
            # Process batch
            videos, bool_masked_pos = self._process_batch(batch)
            
            # Generate targets
            labels = self._generate_targets(videos, bool_masked_pos, patch_size)
            
            # Forward pass
            loss = self._forward_pass(videos, bool_masked_pos, labels)
            
            # Backward pass
            grad_norm = self._backward_pass(loss)
            
            # Update metrics
            self._update_metrics(metric_logger, loss, grad_norm)
            
            # Log metrics
            self._log_metrics(loss, grad_norm)
            
        # Gather stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def _update_lr_wd(self, it: int, lr_schedule_values, wd_schedule_values):
        """Update learning rate and weight decay."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            if lr_schedule_values is not None:
                param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
            if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                param_group["weight_decay"] = wd_schedule_values[it]
    
    def _process_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batch data."""
        videos, bool_masked_pos = batch
        
        # Handle multiple samples
        if self.cfg.AUGMENTATION.NUM_SAMPLE > 1:
            videos = rearrange(videos, 'b c (nt t) h w -> (b nt) c t h w', 
                             nt=self.cfg.AUGMENTATION.NUM_SAMPLE)
            bool_masked_pos = repeat(bool_masked_pos, 'b c -> (b nt) c', 
                                   nt=self.cfg.AUGMENTATION.NUM_SAMPLE)
        
        videos = videos.to(self.device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(self.device, non_blocking=True).flatten(1).to(torch.bool)
        
        return videos, bool_masked_pos
    
    def _generate_targets(self, videos: torch.Tensor, bool_masked_pos: torch.Tensor, 
                         patch_size: int) -> torch.Tensor:
        """Generate reconstruction targets."""
        with torch.no_grad():
            # Normalize videos
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :, None, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :, None, None, None]
            unnorm_videos = videos * std + mean  # in [0, 1]

            if self.cfg.MODEL.NORMALIZE_TARGET:
                videos_squeeze = rearrange(unnorm_videos, 
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', 
                    p0=self.cfg.MODEL.TUBELET_SIZE, p1=patch_size, p2=patch_size)
                videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)
                    ) / (videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
            else:
                videos_patch = rearrange(unnorm_videos, 
                    'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', 
                    p0=self.cfg.MODEL.TUBELET_SIZE, p1=patch_size, p2=patch_size)

            # Handle frame difference as target
            if self.cfg.PRETRAINING.USE_FRAME_DIFF_AS_TARGET:
                videos_patch = self._apply_frame_diff_target(unnorm_videos, patch_size)

            B, _, C = videos_patch.shape
            labels = videos_patch[bool_masked_pos].reshape(B, -1, C)

        return labels
    
    def _apply_frame_diff_target(self, unnorm_videos: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Apply frame difference as target."""
        _, _, t_in, h_in, w_in = unnorm_videos.shape
        t_tokenized = t_in // self.cfg.MODEL.TUBELET_SIZE
        h_tokenized = h_in // patch_size
        w_tokenized = w_in // patch_size
        
        # Convert to patch format
        videos_patch = rearrange(unnorm_videos, 
            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', 
            t=t_tokenized, h=h_tokenized, w=w_tokenized, 
            p0=self.cfg.MODEL.TUBELET_SIZE, p1=patch_size, p2=patch_size)
        
        # Reshape for frame difference computation
        videos_patch = rearrange(videos_patch, 
            'b (t h w) (p0 p1 p2 c) -> b c (t p0) (h p1) (w p2)', 
            t=t_tokenized, h=h_tokenized, w=w_tokenized, 
            p0=self.cfg.MODEL.TUBELET_SIZE, p1=patch_size, p2=patch_size)
        
        # Compute frame differences
        group_size = self.cfg.PRETRAINING.FRAME_DIFF_GROUP_SIZE
        videos_patch_kept = videos_patch[:, :, ::group_size]
        videos_patch_diff = videos_patch - videos_patch_kept.repeat_interleave(group_size, dim=2)
        videos_patch_diff[:, :, ::group_size] = videos_patch[:, :, ::group_size]
        
        # Convert back to token format
        videos_patch = rearrange(videos_patch_diff, 
            'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', 
            t=t_tokenized, h=h_tokenized, w=w_tokenized, 
            p0=self.cfg.MODEL.TUBELET_SIZE, p1=patch_size, p2=patch_size)
        
        return videos_patch
    
    def _forward_pass(self, videos: torch.Tensor, bool_masked_pos: torch.Tensor, 
                     labels: torch.Tensor) -> torch.Tensor:
        """Forward pass with task-specific handling."""
        with torch.cuda.amp.autocast():
            outputs = self.model(videos, bool_masked_pos)
            
            # Compute loss
            if (self.cfg.PRETRAINING.USE_FRAME_DIFF_AS_TARGET and 
                self.cfg.PRETRAINING.TARGET_DIFF_WEIGHT is not None):
                loss = self._compute_weighted_loss(outputs, labels)
            else:
                loss = self.loss_func(input=outputs, target=labels)
        
        return loss
    
    def _compute_weighted_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute weighted loss for frame difference targets."""
        p0 = self.cfg.MODEL.TUBELET_SIZE
        labels_new = rearrange(labels, 'b n (p0 c) -> b n p0 c', p0=p0)
        outputs_new = rearrange(outputs, 'b n (p0 c) -> b n p0 c', p0=p0)
        
        loss_org = self.loss_func(input=outputs_new[:, :, 0], target=labels_new[:, :, 0])
        loss_diff = self.loss_func_diff(input=outputs_new[:, :, 1], target=labels_new[:, :, 1])
        
        weight = self.cfg.PRETRAINING.TARGET_DIFF_WEIGHT
        loss = loss_org * (1 - weight) + loss_diff * weight
        
        return loss
    
    def _backward_pass(self, loss: torch.Tensor) -> float:
        """Backward pass with gradient scaling."""
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        
        self.optimizer.zero_grad()
        # Check if optimizer has second order attribute
        is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
        grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.cfg.OPTIMIZATION.CLIP_GRAD,
                                    parameters=self.model.parameters(), create_graph=is_second_order)
        
        torch.cuda.synchronize()
        
        return grad_norm
    
    def _update_metrics(self, metric_logger, loss: torch.Tensor, grad_norm: float):
        """Update training metrics."""
        loss_value = loss.item()
        loss_scale_value = self.loss_scaler.state_dict()["scale"]
        
        # Update metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        
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
    
    def _log_metrics(self, loss: torch.Tensor, grad_norm: float):
        """Log metrics to tensorboard."""
        if self.log_writer is None:
            return
            
        loss_value = loss.item()
        loss_scale_value = self.loss_scaler.state_dict()["scale"]
        
        # Log metrics
        self.log_writer.update(loss=loss_value, head="loss")
        self.log_writer.update(loss_scale=loss_scale_value, head="opt")
        
        # Log optimization metrics
        max_lr = max(group["lr"] for group in self.optimizer.param_groups)
        min_lr = min(group["lr"] for group in self.optimizer.param_groups)
        weight_decay_value = None
        for group in self.optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        
        self.log_writer.update(lr=max_lr, head="opt")
        self.log_writer.update(min_lr=min_lr, head="opt")
        if weight_decay_value is not None:
            self.log_writer.update(weight_decay=weight_decay_value, head="opt")
        self.log_writer.update(grad_norm=grad_norm, head="opt")
        
        self.log_writer.set_step()
