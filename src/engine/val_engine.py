import math
import sys
import time
import numpy as np
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
        
        # For collecting predictions and labels to compute UAR/WAR
        all_predictions = []
        all_targets = []
        
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
                    
                    # Collect predictions and targets for UAR/WAR calculation
                    predictions = output.argmax(dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # Update metrics
            if cfg.DATA.DATASET_NAME == 'Gaze360':
                total_angular_error += angular_error * videos.shape[0]
                num_samples += videos.shape[0]
            
            batch_size = videos.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['angular_error'].update(angular_error.item(), n=batch_size)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
        # Compute UAR, WAR and F1 scores for classification tasks
        uar, war, weighted_f1, micro_f1, macro_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        if cfg.DATA.DATASET_NAME != 'Gaze360' and len(all_predictions) > 0:
            from sklearn.metrics import confusion_matrix, f1_score
            conf_mat = confusion_matrix(y_true=all_targets, y_pred=all_predictions)
            class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
            uar = np.mean(class_acc)  # Unweighted Average Recall
            war = conf_mat.trace() / conf_mat.sum()  # Weighted Average Recall (same as overall accuracy)
            weighted_f1 = f1_score(y_true=all_targets, y_pred=all_predictions, average='weighted')
            micro_f1 = f1_score(y_true=all_targets, y_pred=all_predictions, average='micro')
            macro_f1 = f1_score(y_true=all_targets, y_pred=all_predictions, average='macro')
            
            # Add these metrics to the logger
            metric_logger.meters['uar'].update(uar, n=len(all_predictions))
            metric_logger.meters['war'].update(war, n=len(all_predictions))
            metric_logger.meters['weighted_f1'].update(weighted_f1, n=len(all_predictions))
            metric_logger.meters['micro_f1'].update(micro_f1, n=len(all_predictions))
            metric_logger.meters['macro_f1'].update(macro_f1, n=len(all_predictions))
        
        # Compute final metrics
        if cfg.DATA.DATASET_NAME == 'Gaze360':
            mean_angular_error = total_angular_error / num_samples
            metric_logger.meters['mean_angle_error'].update(mean_angular_error, n=num_samples)
            print(f'* Mean Angular Error {mean_angular_error:.4f}° loss {metric_logger.loss.global_avg:.6f}')
        else:
            print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
            if len(all_predictions) > 0:
                print(f'* UAR {uar:.4f} WAR {war:.4f} Weighted F1 {weighted_f1:.4f} Micro F1 {micro_f1:.4f} Macro F1 {macro_f1:.4f}')
        
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
    
    @torch.no_grad()
    def compute_detailed_metrics(self, data_loader: DataLoader) -> Dict[str, Any]:
        """
        Compute detailed metrics including confusion matrix and per-class accuracies.
        """
        cfg = get_cfg()
        criterion = self._setup_criterion()
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        print("Computing detailed metrics on validation set...")
        
        for batch_idx, batch in enumerate(data_loader):
            videos, targets = batch[0], batch[1]
            videos = videos.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                output = self.model(videos)
                
                if cfg.DATA.DATASET_NAME != 'Gaze360':
                    predictions = output.argmax(dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_outputs.extend(output.cpu().numpy())
        
        if cfg.DATA.DATASET_NAME != 'Gaze360' and len(all_predictions) > 0:
            from sklearn.metrics import confusion_matrix, f1_score, classification_report
            import pandas as pd
            
            # Compute confusion matrix
            conf_mat = confusion_matrix(y_true=all_targets, y_pred=all_predictions)
            
            # Compute per-class accuracies  
            class_acc = conf_mat.diagonal() / conf_mat.sum(axis=1)
            
            # Compute various metrics
            uar = np.mean(class_acc)  # Unweighted Average Recall
            war = conf_mat.trace() / conf_mat.sum()  # Weighted Average Recall
            weighted_f1 = f1_score(y_true=all_targets, y_pred=all_predictions, average='weighted')
            micro_f1 = f1_score(y_true=all_targets, y_pred=all_predictions, average='micro')
            macro_f1 = f1_score(y_true=all_targets, y_pred=all_predictions, average='macro')
            
            # Create classification report
            class_report = classification_report(y_true=all_targets, y_pred=all_predictions, output_dict=True)
            
            detailed_metrics = {
                'confusion_matrix': conf_mat.tolist(),
                'class_accuracies': class_acc.tolist(),
                'uar': uar,
                'war': war,
                'weighted_f1': weighted_f1,
                'micro_f1': micro_f1,
                'macro_f1': macro_f1,
                'classification_report': class_report,
                'predictions': all_predictions,
                'targets': all_targets
            }
            
            # Print detailed results
            print(f"Confusion Matrix:\n{conf_mat}")
            print(f"Class Accuracies: {[f'{acc:.4f}' for acc in class_acc]}")
            print(f"UAR: {uar:.4f}")
            print(f"WAR: {war:.4f}")
            print(f"Weighted F1: {weighted_f1:.4f}")
            print(f"Micro F1: {micro_f1:.4f}")
            print(f"Macro F1: {macro_f1:.4f}")
            
            # Save predictions to CSV if output directory exists
            if cfg.SYSTEM.OUTPUT_DIR and utils.is_main_process():
                pred_df = pd.DataFrame({
                    'target': all_targets,
                    'prediction': all_predictions
                })
                pred_df.to_csv(os.path.join(cfg.SYSTEM.OUTPUT_DIR, 'detailed_predictions.csv'), index=False)
            
            return detailed_metrics
        else:
            return {}
    
    # def _compute_l2cs_validation(self, output, pitch_target, yaw_target):
    #     """Compute L2CS validation metrics."""
    #     # This is a placeholder - implement actual L2CS validation logic
    #     loss = torch.tensor(0.0)
    #     angular_error = 0.0
    #     return loss, angular_error
