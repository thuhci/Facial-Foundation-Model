import math
import sys
import time
import numpy as np
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from src.utils.config import get_cfg
from src.optim.mixup import Mixup
from timm.utils import ModelEma, accuracy
from src import utils
from src.utils.gaze import gaze3d_to_gaze2d, compute_angular_error
import os
import matplotlib.pyplot as plt

import cv2


class ValidationEngine:
    """
    Clean validation engine.
    """
    
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.flag = False
        # Define patch and region parameters based on the problem description
        self.num_frames = 16
        self.img_size = (160, 160)
        self.patch_size = (2, 16, 16)  # [T, H, W]
        self.num_patches_per_frame = (self.img_size[0] // self.patch_size[1]) * (self.img_size[1] // self.patch_size[2])  # 10 * 10 = 100
        self.temporal_seq_len = self.num_frames // self.patch_size[0]  # 16 / 2 = 8
        self.total_patches = self.num_patches_per_frame * self.temporal_seq_len  # 100 * 8 = 800
        self.num_regions = 20
        self.region_size = 40  # Patches per region
        self.region_patch_shape = (2, 2, 10)  # [T, H, W] in patch units
        self.num_heads = 8  # Default number of attention heads

    def compute_patch_attention(self, A1, A2, A3):
        """
        Compute the attention score for each patch based on three attention layers.
        
        Args:
            A1 (torch.Tensor): Local self-attention, shape [B*N, H, S, S] = [B*20, H, 41, 41]
            A2 (torch.Tensor): Global self-attention, shape [B, H, N, N] = [B, H, 20, 20]
            A3 (torch.Tensor): Local-global cross-attention, shape [B, H, N*(S-1), N] = [B, H, 800, 20]
        
        Returns:
            torch.Tensor: Attention scores for each patch, shape [B, 800]
        """
        B = A2.shape[0]  # Batch size
        N = self.num_regions  # Number of regions
        S_minus_1 = self.region_size  # Number of patches per region
        S = S_minus_1 + 1  # Sequence length per region (including messenger token)
        total_patches = N * S_minus_1  # Total patches = 20 * 40 = 800
        
        # Step 1: Compute A1' [B*N, S_minus_1] = [B*20, 40]
        # Extract messenger token's attention to local patches (A1[i][0][j], j=1,...,40)
        A1_prime = A1[:, :, 0, 1:]  # [B*N, H, 40]
        # Normalize to sum to 1 over j=1,...,40
        A1_prime = F.softmax(A1_prime, dim=-1)  # [B*N, H, 40]
        # Average over heads
        A1_prime = A1_prime.mean(dim=1)  # [B*N, 40]
        
        # Step 2: Compute A2' [B, N] = [B, 20]
        # Sum over i to get total attention received by region j
        A2_prime = A2.sum(dim=2)  # [B, H, 20] -> [B, H, 20]
        A2_prime = A2_prime.mean(dim=1)  # [B, 20]
        
        # Step 3: Compute A3' [B, N] = [B, 20]
        # Sum over i to get total attention received by region k
        A3_prime = A3.sum(dim=2)  # [B, H, 800, 20] -> [B, H, 20]
        A3_prime = A3_prime.mean(dim=1)  # [B, 20]
        
        # Step 4: Compute A[i][j] for each patch
        # A[i][j] = A1'[i][j] * sum_k (A2[k][i] * A3'[k])
        A2 = A2.mean(dim=1)  # [B, H, 20, 20] -> [B, 20, 20]
        A = torch.zeros(B, total_patches, device=A1.device)  # [B, 800]
        
        for i in range(N):  # For each region
            # A1'[i][j] for region i, patches j=1,...,40
            A1_i = A1_prime[i::N]  # [B, 40]
            # sum_k (A2[k][i] * A3'[k])
            A2_k_i = A2[:, :, i]  # [B, 20]
            weighted_A2 = A2_k_i * A3_prime  # [B, 20]
            global_weight = weighted_A2.sum(dim=1, keepdim=True)  # [B, 1]
            # Compute A[i][j] for all patches in region i
            A[:, i*S_minus_1:(i+1)*S_minus_1] = A1_i * global_weight  # [B, 40]
        
        return A

    def map_tokens_to_frames(self, patch_attention):
        """
        Map patch attention scores to video frames.
        
        Args:
            patch_attention (torch.Tensor): Attention scores for each patch, shape [B, 800]
        
        Returns:
            torch.Tensor: Heatmap for each frame, shape [B, T, H, W] = [B, 16, 160, 160]
        """
        B = patch_attention.shape[0]
        T, H, W = self.num_frames, self.img_size[0], self.img_size[1]
        patch_t, patch_h, patch_w = self.patch_size
        patches_per_frame = self.num_patches_per_frame  # 10 * 10 = 100
        temporal_seq_len = self.temporal_seq_len  # 8
        
        # Reshape patch attention to [B, T_patch, H_patch, W_patch] = [B, 8, 10, 10]
        patch_attention = patch_attention.view(B, temporal_seq_len, H // patch_h, W // patch_w)  # [B, 8, 10, 10]
        
        # Upsample to frame size [B, 16, 160, 160]
        heatmap = torch.zeros(B, T, H, W, device=patch_attention.device)
        for t in range(temporal_seq_len):  # For each temporal patch
            # Map temporal patch to frames (each temporal patch covers 2 frames)
            frame_start = t * patch_t
            frame_end = min((t + 1) * patch_t, T)
            # Upsample spatial dimensions: [B, 10, 10] -> [B, 160, 160]
            patch_frame = patch_attention[:, t]  # [B, 10, 10]
            patch_frame = patch_frame.unsqueeze(1)  # [B, 1, 10, 10]
            # Interpolate to spatial size [160, 160]
            patch_frame = F.interpolate(patch_frame, size=(H, W), mode='bilinear', align_corners=False)  # [B, 1, 160, 160]
            patch_frame = patch_frame.squeeze(1)  # [B, 160, 160]
            # Assign to corresponding frames
            for f in range(frame_start, frame_end):
                heatmap[:, f] = patch_frame
        
        # Normalize heatmap per frame
        heatmap = (heatmap - heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]) / \
                  (heatmap.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0] - heatmap.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0] + 1e-8)
        return heatmap

    def overlay_heatmap_on_frame(self, frame, heatmap, alpha=0.4):
        """
        Overlay heatmap on a single frame.
        
        Args:
            frame (np.ndarray): Frame of shape [H, W, C]
            heatmap (np.ndarray): Heatmap of shape [H, W]
            alpha (float): Transparency factor for heatmap
        
        Returns:
            np.ndarray: Overlaid frame
        """
        normalized_heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        heatmap_colored = plt.cm.get_cmap('jet')(normalized_heatmap)[:, :, :3]
        overlaid_frame = frame * (1 - alpha) + heatmap_colored * alpha
        return overlaid_frame

    def save_attention_heatmap(self, video, attn_weights, layer_idx=1, head_idx=0, attn_type_idx=0, save_path=None, epoch=0):
        """
        Save attention heatmap overlaid on video frames.
        
        Args:
            video (torch.Tensor): Input video, shape [B, C, T, H, W]
            attn_weights (list): List of attention weights from model [A1, A2, A3]
            layer_idx (int): Layer index to visualize
            head_idx (int): Head index to visualize
            attn_type_idx (int): Attention type index (0: first, 1: second, 2: third)
            save_path (str): Path to save heatmap visualizations
            epoch (int): Current epoch
        """
        cfg = get_cfg()
        B, C, T, H, W = video.shape
        
        # Compute patch attention using all three attention layers
        A1 = attn_weights[layer_idx][0]  # [B*N, H, 41, 41]
        A2 = attn_weights[layer_idx][1]  # [B, H, 20, 20]
        A3 = attn_weights[layer_idx][2]  # [B, H, 800, 20]
        patch_attention = self.compute_patch_attention(A1, A2, A3)  # [B, 800]
        
        # Map patch attention to frame heatmap
        heatmap = self.map_tokens_to_frames(patch_attention)  # [B, 16, 160, 160]
        
        # Process first sample in batch
        video = video[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
        heatmap = heatmap[0].cpu().numpy()  # [T, H, W]
        
        # ImageNet mean and std for denormalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        if save_path and save_path.endswith('.mp4'):
            out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
            out_raw = cv2.VideoWriter(save_path.replace('.mp4', '_raw.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H))
        
        for t in range(0, T, 4):  # Save every 4th frame
            frame = video[t]
            # Denormalize frame
            frame_denorm = frame * std + mean
            frame_denorm = np.clip(frame_denorm, 0, 1)
            frame_heatmap = heatmap[t]
            overlaid_frame = self.overlay_heatmap_on_frame(frame_denorm, frame_heatmap)
            
            if save_path:
                if save_path.endswith('.mp4'):
                    frame_bgr = (overlaid_frame * 255).astype(np.uint8)[..., ::-1]
                    out.write(frame_bgr)
                    frame_raw_bgr = (frame_denorm * 255).astype(np.uint8)[..., ::-1]
                    out_raw.write(frame_raw_bgr)
                else:
                    # Save heatmap overlaid frame
                    plt.figure(figsize=(8, 8))
                    plt.imshow(overlaid_frame)
                    plt.colorbar(label='Attention Weight')
                    plt.title(f'Frame {t+1}, Layer {layer_idx+1}, Combined Attention, Epoch {epoch}')
                    plt.axis('off')
                    plt.savefig(f"{save_path}_frame{t+1}.png", bbox_inches='tight')
                    plt.close()
                    
                    # Save raw frame
                    plt.figure(figsize=(8, 8))
                    plt.imshow(frame_denorm)
                    plt.colorbar(label='Raw Frame Intensity')
                    plt.title(f'Raw Frame {t+1}, Epoch {epoch}')
                    plt.axis('off')
                    plt.savefig(f"{save_path}_raw_frame{t+1}.png", bbox_inches='tight')
                    plt.close()
        
        if save_path and save_path.endswith('.mp4'):
            out.release()
            out_raw.release()
    
    
    @torch.no_grad()
    def _process_batch(self, batch_data: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process batch data and apply mixup if needed."""
        samples, targets, _ = batch_data
        samples = samples.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        
        cfg = get_cfg()
        # print(f"[DEBUG] samples shape: {samples.shape}, targets shape: {targets.shape}")
        # print(f"[DEBUG] cfg.MODEL.KEEP_TEMPORAL_DIM: {cfg.MODEL.KEEP_TEMPORAL_DIM}")
        if cfg.MODEL.KEEP_TEMPORAL_DIM:
            if targets.dim() == 3:
                pass
            elif targets.dim() == 2:
                raise ValueError("KEEP_TEMPORAL_DIM requires 5D dataset")
        elif targets.dim() == 3:
            # print("[DEBUG] targets shape before squeeze:", targets.shape)
            targets = targets[:,-1,:]
        
        # Ensure correct data types
        if cfg.DATA.TASK == 'regression':
            targets = targets.float()  # Regression task
        elif cfg.DATA.TASK == 'classification':
            targets = targets.long()   # Classification task
        elif cfg.DATA.TASK == 'combine':
            cls_tar = targets[..., 0].long()  # First column for classification
            reg_tar = targets[..., 1:].float()  # All but first column for regression
            targets = {
                "cls": cls_tar,
                "reg": reg_tar
            }
        else:
            raise ValueError(f"Unknown task: {cfg.DATA.TASK}")
            
        return samples, targets
    
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
        
        i=0
        for batch in metric_logger.log_every(data_loader, 10, header):
            videos, targets = batch[0], batch[1]
            # videos = videos.to(self.device, non_blocking=True)
            # targets = targets.to(self.device, non_blocking=True)
            
            videos, targets = self._process_batch(batch)
            # print("[DEBUG] videos", videos)
            if not self.flag:
                img_to_show = videos[0,:,0,...].permute(1,2,0).cpu().numpy()
                # print("[DEBUG] img_to_show", img_to_show)
                # cv2.imwrite("debug.jpg", (img_to_show*255).astype(np.uint8))
                self.flag = True
            
            with torch.cuda.amp.autocast():
                output = self.model(videos)
                if i == 0 and cfg.TRAINING.SAVE_ATTENTION:
                    save_path = f"./att6/val_epoch"
                    os.makedirs(save_path, exist_ok=True)
                    # Save combined attention heatmap (using all three attention types)
                    self.save_attention_heatmap(
                        videos,
                        attn_weights,
                        layer_idx=10,
                        head_idx=0,
                        attn_type_idx=0,  # Ignored for combined attention
                        save_path=f"{save_path}_combined",
                        epoch=0
                    )
                    i += 1

                if cfg.DATA.TASK == 'regression':
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
                    if targets.shape[-1] == 3:  # Check if targets are 3D gaze vectors
                        # 3D gaze vector
                        targets = targets.reshape(-1, 3)  # Ensure correct shape
                        targets = utils.gaze.gaze3d_to_gaze2d(targets)
                        targets = targets.reshape(-1, 2)  # Reshape to 2D angles
                    else:
                        targets = targets.reshape(-1, 2)  # Ensure correct shape
                    output = output.reshape(-1, 2)  # Ensure outputs are also 2D angles
                    # targets = utils.gaze.gaze3d_to_gaze2d(targets)
                    # print("[DEBUG] targets shape:", targets.shape, "output shape:", output.shape)
                    # print("[DEBUG] targets:", targets, "output:", output)
                    loss = criterion(output, targets)
                    # print("[DEBUG] loss:", loss.item())
                    angular_error = utils.gaze.compute_angular_error(output, targets)
                    # print("[DEBUG] angular_error:", angular_error.item())
                    acc1 = torch.tensor(0.0)  # Placeholder
                    acc5 = torch.tensor(0.0)  # Placeholder
                elif cfg.DATA.TASK == 'combine':
                    # Combined classification and regression task
                    cls_tar = targets["cls"].long().flatten()
                    reg_tar = targets["reg"].reshape(-1, 2)
                    cls_pred = output['cls'].reshape(-1, cfg.DATA.NUM_CLASSES_CLS)
                    reg_pred = output['reg'].reshape(-1, 2)

                    # Compute losses
                    cls_loss = criterion['cls_criterion'](cls_pred, cls_tar)
                    reg_loss = criterion['reg_criterion'](reg_pred, reg_tar)
                    loss = cfg.TRAINING.COMBINE_LOSS_ALPHA * cls_loss + reg_loss
                    
                    # Compute metrics
                    acc1, acc5 = accuracy(cls_pred, cls_tar, topk=(1, 5))
                    angular_error = utils.gaze.compute_angular_error(reg_pred, reg_tar)
                    # print("[DEBUG] angular err", angular_error)
                    
                    # Collect predictions for combine task
                    cls_predictions = cls_pred.argmax(dim=-1)
                    all_predictions.extend(cls_predictions.cpu().numpy())
                    all_targets.extend(cls_tar.cpu().numpy())
                else:
                    # Classification task
                    loss = criterion(output, targets)
                    angular_error = torch.tensor(0.0)
                    acc1, acc5 = accuracy(output, targets, topk=(1, 5))
                    
                    # Collect predictions and targets for UAR/WAR calculation
                    predictions = output.argmax(dim=-1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
            
            # Update metrics
            if cfg.DATA.TASK == 'regression' or cfg.DATA.TASK == 'combine':
                total_angular_error += angular_error * videos.shape[0]
                num_samples += videos.shape[0]
            
            batch_size = videos.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['angular_error'].update(angular_error.item(), n=batch_size)
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
            # exit(0)
        # Compute UAR, WAR and F1 scores for classification tasks
        uar, war, weighted_f1, micro_f1, macro_f1 = 0.0, 0.0, 0.0, 0.0, 0.0
        if (cfg.DATA.TASK == 'classification' or cfg.DATA.TASK == 'combine') and len(all_predictions) > 0:
            from sklearn.metrics import confusion_matrix, f1_score
            conf_mat = confusion_matrix(y_true=all_targets, y_pred=all_predictions)
            # if (conf_mat.sum(axis=1) == 0).any():
            #     class_acc = None  # Avoid division by zero
            # else:
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
        if (cfg.DATA.TASK == 'regression' or cfg.DATA.TASK == 'combine') and num_samples > 0:
            mean_angular_error = total_angular_error / num_samples
            metric_logger.meters['mean_angle_error'].update(mean_angular_error, n=num_samples)
            if cfg.DATA.TASK == 'regression':
                print(f'* Mean Angular Error {mean_angular_error:.4f}° loss {metric_logger.loss.global_avg:.6f}')
            else:  # combine task
                print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} Angular Error {mean_angular_error:.4f}° loss {metric_logger.loss.global_avg:.6f}')
                if len(all_predictions) > 0:
                    print(f'* UAR {uar:.4f} WAR {war:.4f} Weighted F1 {weighted_f1:.4f} Micro F1 {micro_f1:.4f} Macro F1 {macro_f1:.4f}')
        else:
            print(f'* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}')
            if len(all_predictions) > 0:
                print(f'* UAR {uar:.4f} WAR {war:.4f} Weighted F1 {weighted_f1:.4f} Micro F1 {micro_f1:.4f} Macro F1 {macro_f1:.4f}')
        
        metric_logger.synchronize_between_processes()
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    
    def _setup_criterion(self):
        cfg = get_cfg()
        """Setup validation criterion."""
        if cfg.DATA.TASK == 'regression':
            if cfg.GAZE.USE_L2CS:
                # L2CS uses custom criterion
                return utils.gaze.l2cs_criterion
            else:
                return torch.nn.MSELoss()
        elif cfg.DATA.TASK == 'combine':
            # Combined task uses multiple criteria
            return {
                'cls_criterion': torch.nn.CrossEntropyLoss(),
                'reg_criterion': torch.nn.MSELoss()
            }
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
            videos, targets = self._process_batch(batch)
            
            with torch.cuda.amp.autocast():
                output = self.model(videos)
                
                if cfg.DATA.TASK == 'regression':
                    predictions = output.argmax(dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_outputs.extend(output.cpu().numpy())
                elif cfg.DATA.TASK == 'combine':
                    # For combine task, use classification predictions
                    cls_pred = output['cls']
                    cls_tar = targets["cls"]
                    predictions = cls_pred.argmax(dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(cls_tar.cpu().numpy())
                    all_outputs.extend(cls_pred.cpu().numpy())
                else:
                    # Classification task
                    predictions = output.argmax(dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    all_outputs.extend(output.cpu().numpy())

        if (cfg.DATA.TASK == 'classification' or cfg.DATA.TASK == 'combine') and len(all_predictions) > 0:
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
            if cfg.SYSTEM.OUTPUT_DIR and utils.utils.is_main_process():
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
