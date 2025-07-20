import math
import sys
import time
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.config import get_cfg
from src.optim.mixup import Mixup
from timm.utils import ModelEma, accuracy
from src import utils



class ValidationEngine:
    """
    Clean validation engine.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
    
    @torch.no_grad()
    def validate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model on validation set.
        """
        # Setup criterion
        criterion = self._setup_criterion()
        cfg = get_cfg()
        
        metric_logger = utils.logger.MetricLogger(delimiter="  ")
        header = 'Val:'
        self.model.eval()
        
        total_angular_error = 0.0
        num_samples = 0
        
        for batch in metric_logger.log_every(data_loader, 10, header):
            videos, targets = batch[0], batch[1]
            videos = videos.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = self.model(videos)
                
                if cfg.DATA.DATASET_NAME == 'Gaze360':
                    # if cfg.GAZE.USE_L2CS:
                    #     # L2CS validation
                    #     gaze_2d = utils.gaze.gaze3d_to_gaze2d(targets)
                    #     pitch_target = gaze_2d[:, 0]
                    #     yaw_target = gaze_2d[:, 1]
                        
                    #     # Compute losses and metrics
                    #     loss, angular_error = self._compute_l2cs_validation(output, pitch_target, yaw_target)
                    #     acc1 = torch.tensor(0.0)  # Placeholder
                    #     acc5 = torch.tensor(0.0)  # Placeholder
                    # else:
                    #     # Standard gaze regression
                    target_angles = utils.gaze.gaze3d_to_gaze2d(targets)
                    loss = criterion(output, target_angles)
                    angular_error = utils.gaze.compute_angular_error(output, target_angles)
                    acc1 = torch.tensor(0.0)  # Placeholder
                    acc5 = torch.tensor(0.0)  # Placeholder
                else:
                    # Classification task
                    loss = criterion(output, targets)
                    angular_error = torch.tensor(0.0)
                    acc1, acc5 = accuracy(output, targets, topk=(1, 5))
            
            # Update metrics
            if cfg.DATA.DATASET_NAME == 'Gaze360':
                total_angular_error += angular_error * videos.shape[0]
                num_samples += videos.shape[0]
            
            batch_size = videos.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['angular_error'].update(angular_error.item(), n=batch_size)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        # Compute final metrics
        if cfg.DATA.DATASET_NAME == 'Gaze360':
            mean_angular_error = total_angular_error / num_samples
            metric_logger.meters['mean_angle_error'].update(mean_angular_error, n=num_samples)
            print(f'* Mean Angular Error {mean_angular_error:.4f}° loss {metric_logger.loss.global_avg:.6f}')
        else:
            print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
        
        metric_logger.synchronize_between_processes()
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def _setup_criterion(self):
        cfg = get_cfg()
        """Setup validation criterion."""
        if cfg.DATA.DATASET_NAME == 'Gaze360':
            if cfg.GAZE.USE_L2CS:
                # L2CS uses custom criterion
                return utils.gaze.l2cs_criterion
            else:
                return torch.nn.MSELoss()
        else:
            return torch.nn.CrossEntropyLoss()
    
    # def _compute_l2cs_validation(self, output, pitch_target, yaw_target):
    #     """Compute L2CS validation metrics."""
    #     # This is a placeholder - implement actual L2CS validation logic
    #     loss = torch.tensor(0.0)
    #     angular_error = 0.0
    #     return loss, angular_error
