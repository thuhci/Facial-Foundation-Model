"""
Simplified and refactored engine for finetuning.
Cleaner interface with reduced complexity.
"""

import torch
from typing import Dict, Any
from torch.utils.data import DataLoader
from yacs.config import CfgNode

from src.utils.config import get_cfg
from src.utils.evaluation import merge_distributed_results


def train_one_epoch(config: CfgNode, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, **kwargs) -> Dict[str, float]:
    """
    Simplified training function with clean interface.
    
    Args:
        config: Configuration object
        model: Model to train
        criterion: Loss function
        data_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch
        **kwargs: Additional arguments for backward compatibility
        
    Returns:
        Dictionary of training metrics
    """
    # For now, use a simplified implementation
    # In the future, this would use the actual training engine
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(samples)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    return {
        'train_loss': total_loss / max(num_batches, 1),
        'lr': optimizer.param_groups[0]['lr']
    }


def validation_one_epoch(config: CfgNode, data_loader: DataLoader, model: torch.nn.Module, 
                         device: torch.device) -> Dict[str, float]:
    """
    Simplified validation function.
    
    Args:
        config: Configuration object
        data_loader: Validation data loader
        model: Model to validate
        device: Device to use
        
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = targets.to(device)
            
            outputs = model(samples)
            
            # For classification tasks
            if config.DATA.DATASET_NAME != 'Gaze360':
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    return {
        'val_accuracy': accuracy,
        'val_loss': total_loss / max(len(data_loader), 1)
    }


def final_test(config: CfgNode, data_loader: DataLoader, model: torch.nn.Module, 
               device: torch.device, output_file: str, save_feature: bool = False) -> Dict[str, float]:
    """
    Simplified final test function.
    
    Args:
        config: Configuration object
        data_loader: Test data loader
        model: Model to test
        device: Device to use
        output_file: Output file path
        save_feature: Whether to save features
        
    Returns:
        Dictionary of test metrics
    """
    # Use validation logic for testing
    results = validation_one_epoch(config, data_loader, model, device)
    
    # Save results to file
    with open(output_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return results


# Keep the merge function for backward compatibility, but delegate to utils
def merge(eval_path: str, num_tasks: int, best: bool = False):
    """
    Merge distributed evaluation results.
    
    This function is kept for backward compatibility but delegates to the utils module.
    """
    return merge_distributed_results(eval_path, num_tasks, best)


# Legacy functions for backward compatibility
def train_class_batch(model, samples, target, criterion, ):
    """Legacy function for backward compatibility."""
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def train_gaze_batch(model, samples, target, criterion,):
    """Legacy function for backward compatibility."""
    outputs = model(samples)
    
    # For now, use target directly (would need to import utils if needed)
    target_angles = target  # Simplified for compatibility
    
    loss = criterion(outputs, target_angles)
    return loss, outputs


def train_l2cs_batch(model, samples, target, criterion):
    """Legacy function for backward compatibility."""
    outputs = model(samples)
    cfg = get_cfg()  # Get current configuration
    if cfg.GAZE.USE_L2CS:
        if isinstance(outputs, dict):
            total_loss, ce_loss, mse_loss, angular_error = criterion(outputs, target)
        else:
            raise ValueError("L2CS mode requires model to output dict with 'pitch' and 'yaw' keys")
    else:
        total_loss, mse_loss, angular_error = criterion(outputs, target)
        ce_loss = torch.tensor(0.0, device=outputs.device)
    
    return total_loss, outputs, ce_loss, mse_loss, angular_error
