#!/usr/bin/env python3
"""
VideoMAE pretraining script using YACS configuration system.
Refactored version with global configuration management for pretraining.
"""
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from timm.models import create_model
from src.optim.optim_factory import create_optimizer
from src.dataset.datasets import build_pretraining_dataset
from src.engine.pretrain_engine import PretrainEngine
from src.utils.utils import NativeScalerWithGradNormCount as NativeScaler
from src.utils.config import get_cfg, merge_config_file, freeze_cfg, load_and_freeze_config
from src.utils.logger import TensorboardLogger
from src import utils
from src.models import ViT_pretrain, ViT, layers


def get_args_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser('VideoMAE pretraining script with YACS configuration', add_help=False)
    
    # Basic arguments
    parser.add_argument('--config', default='', type=str, help='Path to YAML config file', required=True)
    parser.add_argument('--output_dir', default='./output', type=str, help='Output directory')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    
    return parser


def create_data_loader():
    """Create data loader from global configuration."""
    cfg = get_cfg()
    
    # Create dataset
    dataset_train = build_pretraining_dataset()
    
    # Create sampler
    num_tasks = utils.utils.get_world_size()
    global_rank = utils.utils.get_rank()
    sampler_rank = global_rank
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
    )
    print("Sampler_train = %s" % str(sampler_train))
    
    # Create data loader
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        batch_size=cfg.DATA.BATCH_SIZE,
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=cfg.DATA.PIN_MEMORY,
        drop_last=True,
        worker_init_fn=utils.utils.seed_worker
    )
    
    return data_loader_train, len(dataset_train)


def create_model_from_config():
    """Create model from global configuration."""
    cfg = get_cfg()
    
    print(f"Creating model: {cfg.MODEL.NAME}")
    
    # Handle model name variations
    model_name = cfg.MODEL.NAME
    if 'no_depth' in model_name and cfg.MODEL.ENCODER_DEPTH is not None:
        model = create_model(
            model_name,
            pretrained=False,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_block_rate=None,
            decoder_depth=cfg.MODEL.DECODER_DEPTH,
            encoder_depth=cfg.MODEL.ENCODER_DEPTH,
            attn_type=cfg.MODEL.ATTN_TYPE,
            lg_region_size=cfg.MODEL.LG_REGION_SIZE,
            lg_first_attn_type=cfg.MODEL.LG_FIRST_ATTN_TYPE,
            lg_third_attn_type=cfg.MODEL.LG_THIRD_ATTN_TYPE,
            lg_attn_param_sharing_first_third=cfg.MODEL.LG_ATTN_PARAM_SHARING_FIRST_THIRD,
            lg_attn_param_sharing_all=cfg.MODEL.LG_ATTN_PARAM_SHARING_ALL,
            lg_no_second=cfg.MODEL.LG_NO_SECOND,
            lg_no_third=cfg.MODEL.LG_NO_THIRD,
        )
    else:
        model = create_model(
            model_name,
            pretrained=False,
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_block_rate=None,
            decoder_depth=cfg.MODEL.DECODER_DEPTH,
            attn_type=cfg.MODEL.ATTN_TYPE,
            lg_region_size=cfg.MODEL.LG_REGION_SIZE,
            lg_first_attn_type=cfg.MODEL.LG_FIRST_ATTN_TYPE,
            lg_third_attn_type=cfg.MODEL.LG_THIRD_ATTN_TYPE,
            lg_attn_param_sharing_first_third=cfg.MODEL.LG_ATTN_PARAM_SHARING_FIRST_THIRD,
            lg_attn_param_sharing_all=cfg.MODEL.LG_ATTN_PARAM_SHARING_ALL,
            lg_no_second=cfg.MODEL.LG_NO_SECOND,
            lg_no_third=cfg.MODEL.LG_NO_THIRD,
        )
    
    model = model.float()
    
    # Get patch size and set window size
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    
    # Update config with computed values
    cfg.MODEL.WINDOW_SIZE = (cfg.DATA.NUM_FRAMES // cfg.MODEL.TUBELET_SIZE,
                            cfg.MODEL.INPUT_SIZE // patch_size[0],
                            cfg.MODEL.INPUT_SIZE // patch_size[1])
    cfg.MODEL.PATCH_SIZE = patch_size
    
    return model


def create_optimizer_from_config(model):
    """Create optimizer from global configuration."""
    cfg = get_cfg()
    
    model_without_ddp = model
    optimizer = create_optimizer(model_without_ddp)
    
    return optimizer


def create_scheduler_from_config(num_training_steps_per_epoch):
    """Create learning rate scheduler from global configuration."""
    cfg = get_cfg()
    
    # Scale learning rates by batch size
    total_batch_size = cfg.DATA.BATCH_SIZE * utils.utils.get_world_size()
    cfg.OPTIMIZATION.LR = cfg.OPTIMIZATION.LR * total_batch_size / 256
    cfg.OPTIMIZATION.MIN_LR = cfg.OPTIMIZATION.MIN_LR * total_batch_size / 256
    cfg.OPTIMIZATION.WARMUP_LR = cfg.OPTIMIZATION.WARMUP_LR * total_batch_size / 256
    
    print("LR = %.8f" % cfg.OPTIMIZATION.LR)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))
    
    # Create learning rate schedule values
    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.utils.cosine_scheduler(
        cfg.OPTIMIZATION.LR, cfg.OPTIMIZATION.MIN_LR, cfg.TRAINING.EPOCHS, num_training_steps_per_epoch,
        warmup_epochs=cfg.OPTIMIZATION.WARMUP_EPOCHS, warmup_steps=cfg.OPTIMIZATION.WARMUP_STEPS,
    )
    
    # Create weight decay schedule values
    if cfg.OPTIMIZATION.WEIGHT_DECAY_END is None:
        cfg.OPTIMIZATION.WEIGHT_DECAY_END = cfg.OPTIMIZATION.WEIGHT_DECAY
    wd_schedule_values = utils.utils.cosine_scheduler(
        cfg.OPTIMIZATION.WEIGHT_DECAY, cfg.OPTIMIZATION.WEIGHT_DECAY_END, 
        cfg.TRAINING.EPOCHS, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))
    
    return lr_schedule_values, wd_schedule_values


def main(args):
    """Main pretraining function."""
    # Load configuration FIRST
    if args.config:
        # cfg = load_and_freeze_config(args.config)
        cfg = get_cfg()
        cfg = merge_config_file(cfg, args.config)
    else:
        raise ValueError("Config file is required for pretraining")
    
    # Set output directory
    cfg.TRAINING.OUTPUT_DIR = args.output_dir
    
    # Setup distributed training AFTER loading config
    if cfg.SYSTEM.DISTRIBUTED:
        print(f"Setting up distributed training with backend: {cfg.SYSTEM.DIST_BACKEND}")
        utils.utils.init_distributed_mode()
        
        # Update system config from utils after distributed init
        cfg.SYSTEM.DISTRIBUTED = utils.utils.is_dist_avail_and_initialized()
        cfg.SYSTEM.WORLD_SIZE = utils.utils.get_world_size()
        cfg.SYSTEM.LOCAL_RANK = utils.utils.get_rank()
    
    print(args)
    
    # Set device based on local rank
    if torch.cuda.is_available() and cfg.SYSTEM.DIST_BACKEND.lower() != 'gloo':
        device = torch.device(f'cuda:{utils.utils.get_rank()}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    
    print(f"Rank {utils.utils.get_rank()}: Using device {device} with backend {cfg.SYSTEM.DIST_BACKEND}")
    
    # Fix seed
    seed = args.seed + utils.utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Enable cudnn benchmark
    cudnn.benchmark = True
    
    # Create model
    model = create_model_from_config()
    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("Model = %s" % str(model_without_ddp))
    print('number of params: {} M'.format(n_parameters / 1e6))
    
    # Create data loader
    data_loader_train, dataset_length = create_data_loader()
    num_training_steps_per_epoch = dataset_length // cfg.DATA.BATCH_SIZE // utils.utils.get_world_size()
    
    # Setup logging
    if utils.utils.get_rank() == 0 and cfg.SYSTEM.LOG_DIR is not None:
        os.makedirs(cfg.SYSTEM.LOG_DIR, exist_ok=True)
        log_writer = TensorboardLogger(log_dir=cfg.SYSTEM.LOG_DIR)
    else:
        log_writer = None
    
    # Setup distributed training
    if cfg.SYSTEM.DISTRIBUTED:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[utils.utils.get_rank()], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # Create optimizer and scheduler
    optimizer = create_optimizer_from_config(model_without_ddp)
    loss_scaler = NativeScaler()
    lr_schedule_values, wd_schedule_values = create_scheduler_from_config(num_training_steps_per_epoch)
    
    # Auto load model if needed
    utils.utils.auto_load_model(
        model=model, model_without_ddp=model_without_ddp, 
        optimizer=optimizer, loss_scaler=loss_scaler)
    
    # Create training engine
    engine = PretrainEngine(
        model=model, 
        optimizer=optimizer, 
        loss_scaler=loss_scaler,
        log_writer=log_writer,
        device=device
    )
    
    # Clear cache and start training
    torch.cuda.empty_cache()
    print(f"Start training for {cfg.TRAINING.EPOCHS} epochs")
    start_time = time.time()
    
    for epoch in range(cfg.TRAINING.START_EPOCH, cfg.TRAINING.EPOCHS):
        if cfg.SYSTEM.DISTRIBUTED:
            data_loader_train.sampler.set_epoch(epoch)
        
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        
        # Train one epoch
        train_stats = engine.train_one_epoch(
            data_loader=data_loader_train,
            epoch=epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            start_steps=epoch * num_training_steps_per_epoch,
            patch_size=cfg.MODEL.PATCH_SIZE[0],
        )
        
        # Save checkpoint
        if cfg.TRAINING.OUTPUT_DIR:
            if (epoch + 1) % cfg.TRAINING.SAVE_CKPT_FREQ == 0 or epoch + 1 == cfg.TRAINING.EPOCHS:
                utils.utils.save_model(
                    epoch=epoch, model=model, model_without_ddp=model_without_ddp, 
                    optimizer=optimizer, loss_scaler=loss_scaler)
        
        # Log stats
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}
        
        if cfg.TRAINING.OUTPUT_DIR and utils.utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(cfg.TRAINING.OUTPUT_DIR, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)
