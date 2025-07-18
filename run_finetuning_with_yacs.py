#!/usr/bin/env python3
"""
Clean training script using YACS configuration system.
Refactored version with global configuration management.
"""

import os
import sys
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from src.optim.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from timm.utils import ModelEma

from src.utils.config import get_cfg, merge_config_file, freeze_cfg, load_and_freeze_config
from src.engine.train_engine import TrainingEngine
from src.engine.val_engine import ValidationEngine
from src.utils.evaluation import merge_distributed_results
from src.optim.mixup import Mixup
# from src.optim.optim_factory import LayerDecayValueAssigner
from src.dataset.datasets import build_dataset
from src.utils.utils import NativeScalerWithGradNormCount as NativeScaler
from src.utils.utils import multiple_samples_collate
from src.utils.logger import TensorboardLogger
from src.utils import utils

from src.models import ViT, ViT_pretrain, layers

def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('Training script with YACS configuration', add_help=False)
    
    # Basic arguments
    parser.add_argument('--config', default='', type=str, help='Path to YAML config file', required=True)
    parser.add_argument('--output_dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--dist_eval', action='store_true', help='Distributed evaluation')
    
    
    
    return parser


def create_data_loaders():
    """Create data loaders from global configuration."""
    cfg = get_cfg()
    
    # Create dataset
    dataset_train, nb_classes = build_dataset(is_train=True, test_mode=False)
    dataset_val, nb_classes = build_dataset(is_train=False, test_mode=False)
    
    # Create data loaders
    if cfg.SYSTEM.DISTRIBUTED:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    if cfg.AUGMENTATION.NUM_SAMPLE > 1:
        collate_func = partial(multiple_samples_collate, fold=False)
    else:
        collate_func = None
    
    # Create data loaders
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=True,
        collate_fn=collate_func
    )
    # print(f"Data loader train size: {len(data_loader_train)}")
    
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers= cfg.DATA.NUM_WORKERS,
        pin_memory= cfg.DATA.PIN_MEMORY,
        drop_last=False,
        collate_fn=collate_func
    )
    # print(f"Data loader val size: {len(data_loader_val)}")
    
    return data_loader_train, data_loader_val


def create_model_from_config():
    """Create model from global configuration."""
    cfg = get_cfg()
    
    # Handle model name variations
    model_name = cfg.MODEL.NAME
    if 'no_depth' in model_name and cfg.MODEL.DEPTH is not None:
        print(f"==> Note: use custom model depth={cfg.MODEL.DEPTH}!")
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=cfg.DATA.NUM_CLASSES,
            all_frames=cfg.DATA.NUM_FRAMES * cfg.DATA.NUM_SEGMENTS,
            tubelet_size=cfg.MODEL.TUBELET_SIZE,
            drop_rate=cfg.MODEL.DROP,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
            drop_block_rate=None,
            use_mean_pooling=cfg.MODEL.USE_MEAN_POOLING,
            init_scale=cfg.MODEL.INIT_SCALE,
            attn_type=cfg.MODEL.ATTN_TYPE,
            lg_region_size=cfg.MODEL.LG_REGION_SIZE, lg_first_attn_type=cfg.MODEL.LG_FIRST_ATTN_TYPE,
            lg_third_attn_type=cfg.MODEL.LG_THIRD_ATTN_TYPE,
            lg_attn_param_sharing_first_third=cfg.MODEL.LG_ATTN_PARAM_SHARING_FIRST_THIRD,
            lg_attn_param_sharing_all=cfg.MODEL.LG_ATTN_PARAM_SHARING_ALL,
            lg_classify_token_type=cfg.MODEL.LG_CLASSIFY_TOKEN_TYPE,
            lg_no_second=cfg.MODEL.LG_NO_SECOND, lg_no_third=cfg.MODEL.LG_NO_THIRD,
        )
    else:
        model = create_model(
            model_name,
            pretrained=False,
            num_classes=cfg.DATA.NUM_CLASSES,
            all_frames=cfg.DATA.NUM_FRAMES * cfg.DATA.NUM_SEGMENTS,
            tubelet_size=cfg.MODEL.TUBELET_SIZE,
            drop_rate=cfg.MODEL.DROP,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            attn_drop_rate=cfg.MODEL.ATTN_DROP_RATE,
            use_mean_pooling=cfg.MODEL.USE_MEAN_POOLING,
            init_scale=cfg.MODEL.INIT_SCALE,
            attn_type=cfg.MODEL.ATTN_TYPE,
            lg_region_size=cfg.MODEL.LG_REGION_SIZE, lg_first_attn_type=cfg.MODEL.LG_FIRST_ATTN_TYPE,
            lg_third_attn_type=cfg.MODEL.LG_THIRD_ATTN_TYPE,
            lg_attn_param_sharing_first_third=cfg.MODEL.LG_ATTN_PARAM_SHARING_FIRST_THIRD,
            lg_attn_param_sharing_all=cfg.MODEL.LG_ATTN_PARAM_SHARING_ALL,
            lg_classify_token_type=cfg.MODEL.LG_CLASSIFY_TOKEN_TYPE,
            lg_no_second=cfg.MODEL.LG_NO_SECOND, lg_no_third=cfg.MODEL.LG_NO_THIRD,
        )
    
    return model


def create_optimizer_from_config(model, get_num_layer=None, get_layer_scale=None):
    """Create optimizer from global configuration."""
    cfg = get_cfg()
    
    # Get layer assignments for layer decay
    if get_num_layer is not None and get_layer_scale is not None:
        assigner = LayerDecayValueAssigner(
            list(cfg.OPTIMIZATION.LAYER_DECAY.values()),
            get_num_layer,
            get_layer_scale
        )
    else:
        assigner = None
        
    model_without_ddp = model
    if cfg.SYSTEM.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    skip_weight_decay_list = model.no_weight_decay()
    get_num_layer=assigner.get_layer_id if assigner is not None else None,
    print("get_num_layer", get_num_layer)
    optimizer = create_optimizer(
        model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=assigner.get_layer_id if assigner is not None else None, 
        get_layer_scale=assigner.get_scale if assigner is not None else None)
    
    return optimizer


def create_criterion_from_config():
    """Create loss criterion from global configuration."""
    cfg = get_cfg()
    
    if cfg.DATA.DATASET_NAME == 'Gaze360':
        if cfg.GAZE.USE_L2CS:
            from src.utils.gaze.l2cs_criterion import L2CSCriterion
            criterion = L2CSCriterion()
        else:
            criterion = torch.nn.MSELoss()
    else:
        # Classification task
        mixup_active = cfg.AUGMENTATION.MIXUP_ALPHA > 0 or cfg.AUGMENTATION.CUTMIX_ALPHA > 0
        if mixup_active:
            # Mixup mode
            criterion = SoftTargetCrossEntropy()
        elif cfg.AUGMENTATION.SMOOTHING > 0:
            # Label smoothing
            criterion = LabelSmoothingCrossEntropy(smoothing=cfg.AUGMENTATION.SMOOTHING)
        else:
            # Standard cross entropy
            criterion = torch.nn.CrossEntropyLoss()
    
    return criterion


def create_mixup_from_config():
    """Create mixup function from global configuration."""
    cfg = get_cfg()
    
    mixup_active = cfg.AUGMENTATION.MIXUP > 0 or cfg.AUGMENTATION.CUTMIX > 0
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=cfg.AUGMENTATION.MIXUP,
            cutmix_alpha=cfg.AUGMENTATION.CUTMIX,
            cutmix_minmax=cfg.AUGMENTATION.CUTMIX_MINMAX,
            prob=cfg.AUGMENTATION.MIXUP_PROB,
            switch_prob=cfg.AUGMENTATION.MIXUP_SWITCH_PROB,
            mode=cfg.AUGMENTATION.MIXUP_MODE,
            label_smoothing=cfg.AUGMENTATION.SMOOTHING,
            num_classes=cfg.DATA.NUM_CLASSES
        )
    else:
        mixup_fn = None
    
    return mixup_fn


def create_scheduler_from_config(optimizer, num_training_steps_per_epoch):
    """Create learning rate scheduler from global configuration."""
    cfg = get_cfg()
    
    # Create scheduler
    # lr_scheduler, _ = create_scheduler(cfg.OPTIMIZATION, optimizer)
    
    # Create learning rate schedule values
    if cfg.OPTIMIZATION.SCHED == 'cosine':
        lr_schedule_values = utils.cosine_scheduler(
            cfg.OPTIMIZATION.LR,
            cfg.OPTIMIZATION.MIN_LR,
            cfg.TRAINING.EPOCHS,
            num_training_steps_per_epoch,
            warmup_epochs=cfg.OPTIMIZATION.WARMUP_EPOCHS,
            warmup_steps=cfg.OPTIMIZATION.WARMUP_STEPS
        )
    else:
        lr_schedule_values = None
    
    # Create weight decay schedule values
    if cfg.OPTIMIZATION.WEIGHT_DECAY_END is not None:
        wd_schedule_values = utils.cosine_scheduler(
            cfg.OPTIMIZATION.WEIGHT_DECAY,
            cfg.OPTIMIZATION.WEIGHT_DECAY_END,
            cfg.TRAINING.EPOCHS,
            num_training_steps_per_epoch
        )
    else:
        wd_schedule_values = None
    
    return lr_schedule_values, wd_schedule_values


def main(args):
    """Main training function."""
    # Setup distributed training
    utils.init_distributed_mode()
    
    # Load and freeze configuration (recommended workflow)
    if args.config:
        print(f"Loading configuration from: {args.config}")
        cfg = get_cfg()
        merge_config_file(cfg, args.config)
    else:
        print("Using default configuration")
        cfg = get_cfg()
    
    # Set output directory
    cfg.TRAINING.OUTPUT_DIR = args.output_dir
    
    # Freeze configuration to make it immutable
    print("Freezing configuration...")
    freeze_cfg()
    
    # Print final configuration
    print("Final configuration:")
    print(cfg)
    
    # Set device
    device = torch.device(cfg.SYSTEM.DEVICE if torch.cuda.is_available() else 'cpu')
    
    # Fix seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Enable cudnn benchmark
    cudnn.benchmark = True
    
    # Create data loaders
    data_loader_train, data_loader_val = create_data_loaders()
    print(f"Data loader train size: {len(data_loader_train)}")
    print(f"Data loader val size: {len(data_loader_val)}")
    
    # Create model
    model = create_model_from_config()
    model.to(device)
    
    # Wrap model for distributed training
    model_without_ddp = model
    if cfg.SYSTEM.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[cfg.SYSTEM.GPU], find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    # Setup training components
    
    if cfg.OPTIMIZATION.LAYER_DECAY is not None:
        get_num_layer = model_without_ddp.get_num_layer if hasattr(model_without_ddp, 'get_num_layer') else None
        get_layer_scale = model_without_ddp.get_layer_scale if hasattr(model_without_ddp, 'get_layer_scale') else None
    
    optimizer = create_optimizer_from_config(model_without_ddp, get_num_layer, get_layer_scale)
    criterion = create_criterion_from_config()
    mixup_fn = create_mixup_from_config()
    
    # Create model EM
    model_ema = None
    if cfg.TRAINING.MODELEMA_:
        model_ema = ModelEma(
            model_without_ddp,
            decay=cfg.TRAINING.MODEL_EMA_DECAY,
            device='cpu' if cfg.TRAINING.MODEL_EMA_FORCE_CPU else None
        )
    
    # Create loss scaler
    loss_scaler = NativeScaler()
    
    # Create scheduler
    num_training_steps_per_epoch = len(data_loader_train) // cfg.TRAINING.UPDATE_FREQ
    lr_schedule_values, wd_schedule_values = create_scheduler_from_config(
        optimizer, num_training_steps_per_epoch
    )
    
    # Create log writer
    log_writer = TensorboardLogger(
        log_dir=cfg.TRAINING.OUTPUT_DIR,
    )
    
    # Create training engine
    training_engine = TrainingEngine(
        model, 
        optimizer = optimizer,
        loss_scaler = loss_scaler,
        criterion = criterion,
        model_ema = model_ema,
        mixup_fn = mixup_fn,
        log_writer = log_writer,
        device = device,
        )

    # Create validation engine
    validation_engine = ValidationEngine(model, device)
    
    # Resume from checkpoint
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        else:
            start_epoch = 0
    else:
        start_epoch = 0
    
    # Evaluation mode
    if args.eval:
        test_stats = validation_engine.validate(data_loader_val)
        print(f"Validation results: {test_stats}")
        return
    
    # Training loop
    print(f"Start training for {cfg.TRAINING.EPOCHS} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_epoch = 0
    
    for epoch in range(start_epoch, cfg.TRAINING.EPOCHS):
        if cfg.SYSTEM.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)
        
        # Training
        train_stats = training_engine.train_one_epoch(
            data_loader_train,
            epoch,
            lr_schedule_values,
            wd_schedule_values,
            num_training_steps_per_epoch,
            epoch * num_training_steps_per_epoch
        )
        
        # Validation
        test_stats = validation_engine.validate(data_loader_val)
        
        # Update learning rate
        # lr_scheduler.step(epoch)
        
        # Save checkpoint
        if cfg.TRAINING.OUTPUT_DIR and utils.is_main_process():
            checkpoint_path = os.path.join(cfg.TRAINING.OUTPUT_DIR, f'checkpoint-{epoch}.pth')
            utils.save_checkpoint(
                {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'scaler': loss_scaler.state_dict(),
                    'config': cfg,
                    'model_ema': model_ema.state_dict() if model_ema else None,
                },
                checkpoint_path
            )
        
        # Log stats
        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,
            'n_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        if cfg.TRAINING.OUTPUT_DIR and utils.is_main_process():
            with open(os.path.join(cfg.TRAINING.OUTPUT_DIR, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Track best accuracy
        if cfg.DATA.DATASET_NAME == 'Gaze360':
            current_metric = test_stats.get('mean_angle_error', 1e8)
            if current_metric < max_accuracy or epoch == start_epoch:
                max_accuracy = current_metric
                best_epoch = epoch
        else:
            current_metric = test_stats.get('acc1', 0)
            if current_metric > max_accuracy:
                max_accuracy = current_metric
                best_epoch = epoch
        
        print(f'Max accuracy: {max_accuracy:.4f} at epoch {best_epoch}')
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f'Training time {total_time_str}')


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
