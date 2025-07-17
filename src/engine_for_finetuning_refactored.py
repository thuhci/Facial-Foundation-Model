"""
Simplified and refactored engine for finetuning.
Cleaner interface with reduced complexity.
"""

import torch
from typing import Dict, Any
from torch.utils.data import DataLoader

from src.config.config import Config
from src.engine.training_engine import TrainingEngine, ValidationEngine
from src.utils.evaluation import merge_distributed_results


def train_one_epoch(config: Config, model: torch.nn.Module, criterion: torch.nn.Module,
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
    # Create training engine
    engine = TrainingEngine(config, model, device)
    
    # Setup training components
    engine.setup_training_components(
        optimizer=optimizer,
        loss_scaler=kwargs.get('loss_scaler'),
        criterion=criterion,
        model_ema=kwargs.get('model_ema'),
        mixup_fn=kwargs.get('mixup_fn'),
        log_writer=kwargs.get('log_writer')
    )
    
    # Train one epoch
    return engine.train_one_epoch(
        data_loader=data_loader,
        epoch=epoch,
        lr_schedule_values=kwargs.get('lr_schedule_values'),
        wd_schedule_values=kwargs.get('wd_schedule_values'),
        num_training_steps_per_epoch=kwargs.get('num_training_steps_per_epoch'),
        start_steps=kwargs.get('start_steps')
    )


def validation_one_epoch(config: Config, data_loader: DataLoader, model: torch.nn.Module, 
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
    engine = ValidationEngine(config, model, device)
    return engine.validate(data_loader)


def final_test(config: Config, data_loader: DataLoader, model: torch.nn.Module, 
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
    engine = ValidationEngine(config, model, device)
    
    # For now, use validation logic for testing
    # In the future, this could be extended with test-specific logic
    results = engine.validate(data_loader)
    
    # Save results to file
    # This is a simplified version - full implementation would save detailed results
    with open(output_file, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    return results


# Keep the merge function for backward compatibility, but delegate to utils
def merge(eval_path: str, num_tasks: int, args=None, best: bool = False):
    """
    Merge distributed evaluation results.
    
    This function is kept for backward compatibility but delegates to the utils module.
    """
    return merge_distributed_results(eval_path, num_tasks, args, best)


# Legacy functions for backward compatibility
def train_class_batch(model, samples, target, criterion, args=None):
    """Legacy function for backward compatibility."""
    outputs = model(samples)
    loss = criterion(outputs, target)
    return loss, outputs


def train_gaze_batch(model, samples, target, criterion, args=None):
    """Legacy function for backward compatibility."""
    outputs = model(samples)
    
    # This assumes utils.gaze3d_to_gaze2d exists
    if hasattr(utils, 'gaze3d_to_gaze2d'):
        target_angles = utils.gaze3d_to_gaze2d(target)
    else:
        target_angles = target  # Fallback
    
    loss = criterion(outputs, target_angles)
    return loss, outputs


def train_l2cs_batch(model, samples, target, criterion, args=None):
    """Legacy function for backward compatibility."""
    outputs = model(samples)
    
    if args and args.use_l2cs:
        if isinstance(outputs, dict):
            total_loss, ce_loss, mse_loss, angular_error = criterion(outputs, target, args)
        else:
            raise ValueError("L2CS mode requires model to output dict with 'pitch' and 'yaw' keys")
    else:
        total_loss, mse_loss, angular_error = criterion(outputs, target)
        ce_loss = torch.tensor(0.0, device=outputs.device)
    
    return total_loss, outputs, ce_loss, mse_loss, angular_error
