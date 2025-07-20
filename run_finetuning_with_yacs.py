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
from collections import OrderedDict
from functools import partial

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
        
    model = model.float()

    # 针对Gaze360任务修改输出层
    if cfg.DATA.DATASET_NAME == 'Gaze360':
        in_features = model.head.in_features
        if cfg.GAZE.USE_L2CS:
            # L2CS: 两个分类头，每个有 num_bins 个输出
            # [DEBUG] change into ModuleList further, now like this for debug
            model.head = torch.nn.ModuleDict({
                'pitch': torch.nn.Linear(in_features, cfg.GAZE.NUM_BINS),
                'yaw': torch.nn.Linear(in_features, cfg.GAZE.NUM_BINS)
            })
        else:
            # 原始方式：直接回归2个角度
            model.head = torch.nn.Linear(in_features, cfg.DATA.NUM_CLASSES)
    else:
        model.head = torch.nn.Linear(model.head.in_features, cfg.DATA.NUM_CLASSES)
    
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    # args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    # args.patch_size = patch_size
    # cfg.MODEL.PATCH_SIZE = patch_size
    cfg.MODEL.WINDOW_SIZE = (cfg.DATA.NUM_FRAMES // cfg.MODEL.TUBELET_SIZE,
                             cfg.MODEL.INPUT_SIZE // patch_size[0],
                                cfg.MODEL.INPUT_SIZE // patch_size[1])
    cfg.MODEL.PATCH_SIZE = patch_size

    if cfg.TRAINING.FINETUNE:
        if cfg.TRAINING.FINETUNE.startswith('https://'):
            checkpoint = torch.hub.load_state_dict_from_url(
                cfg.TRAINING.FINETUNE, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(cfg.TRAINING.FINETUNE, map_location='cpu', weights_only=False)

        print("Load ckpt from %s" % cfg.TRAINING.FINETUNE)
        checkpoint_model = None
        for model_key in cfg.MODEL.MODEL_KEY.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches #
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(cfg.DATA.NUM_FRAMES // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (cfg.DATA.NUM_FRAMES // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, cfg.DATA.NUM_FRAMES // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, cfg.DATA.NUM_FRAMES // model.patch_embed.tubelet_size, new_size, new_size, embedding_size)
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=cfg.MODEL.PREFIX)

    # print("Model = %s" % str(model_without_ddp))
    # print('number of params:', n_parameters)
    
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
    skip_weight_decay_list = model.no_weight_decay()
    
    # 正确获取函数
    get_num_layer_fn = assigner.get_layer_id if assigner is not None else None
    get_layer_scale_fn = assigner.get_scale if assigner is not None else None
    
    optimizer = create_optimizer(
        model_without_ddp, skip_list=skip_weight_decay_list,
        get_num_layer=get_num_layer_fn, 
        get_layer_scale=get_layer_scale_fn)
    
    return optimizer


def create_criterion_from_config():
    """Create loss criterion from global configuration."""
    cfg = get_cfg()
    
    if cfg.DATA.DATASET_NAME == 'Gaze360':
        if cfg.GAZE.USE_L2CS:
            from src.utils.gaze import l2cs_criterion
            criterion = l2cs_criterion
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
    # Load configuration FIRST
    if args.config:
        print(f"Loading configuration from: {args.config}")
        cfg = get_cfg()
        merge_config_file(cfg, args.config)
    else:
        print("Using default configuration")
        cfg = get_cfg()
    
    # Set output directory
    cfg.TRAINING.OUTPUT_DIR = args.output_dir
    
    # Setup distributed training AFTER loading config
    print(f"Setting up distributed training with backend: {cfg.SYSTEM.DIST_BACKEND}")
    if cfg.SYSTEM.DISTRIBUTED:
        utils.init_distributed_mode()
    
    # Update system config from utils after distributed init
        cfg.SYSTEM.DISTRIBUTED = utils.is_dist_avail_and_initialized()
        cfg.SYSTEM.WORLD_SIZE = utils.get_world_size()
        cfg.SYSTEM.LOCAL_RANK = utils.get_rank()
    
    # Set device based on local rank
    if torch.cuda.is_available():
        torch.cuda.set_device(utils.get_rank())
        device = torch.device(f'cuda:{utils.get_rank()}')
        cfg.SYSTEM.DEVICE = f'cuda:{utils.get_rank()}'
    else:
        device = torch.device('cpu')
        cfg.SYSTEM.DEVICE = 'cpu'
    
    print(f"Rank {utils.get_rank()}: Using device {device} with backend {cfg.SYSTEM.DIST_BACKEND}")
    
    # Fix seed
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    # Enable cudnn benchmark
    cudnn.benchmark = True
    
    # Create data loaders
    data_loader_train, data_loader_val = create_data_loaders()
    print(f"Data loader train size: {len(data_loader_train)}")
    print(f"Data loader val size: {len(data_loader_val)}")
    
    # Create model
    model = create_model_from_config()
    model.to(device)
    
    # Wrap model for distributed training BEFORE creating optimizer
    model_without_ddp = model
    if cfg.SYSTEM.DISTRIBUTED:
        print(f"Rank {utils.get_rank()}: Wrapping model with DDP (backend: {cfg.SYSTEM.DIST_BACKEND})")
        # Synchronize before DDP creation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # DDP 参数根据后端类型调整
        if cfg.SYSTEM.DIST_BACKEND.lower() == 'gloo':
            # Gloo 后端不需要指定 device_ids
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=False,
                broadcast_buffers=True
            )
        else:
            # NCCL 后端需要指定 device_ids
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[utils.get_rank()], 
                output_device=utils.get_rank(),
                find_unused_parameters=False,
                broadcast_buffers=True
            )
        model_without_ddp = model.module
        print(f"Rank {utils.get_rank()}: DDP wrapper created successfully")
    
    # Freeze configuration to make it immutable
    print("Freezing configuration...")
    freeze_cfg()
    
    # Print final configuration if main process
    if utils.get_rank() == 0:
        print("Final configuration:")
        print(cfg)
    
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
        
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)
    
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
        
        if cfg.TRAINING.OUTPUT_DIR and utils.is_main_process() and epoch % cfg.TRAINING.SAVE_CKPT_FREQ == 0:
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
