"""
Configuration management system for facial foundation model training.
Uses YACS (Yet Another Configuration System) for flexible configuration management.
"""

import yaml
import argparse
from typing import Optional, List, Union, Dict, Any
from pathlib import Path
from yacs.config import CfgNode as CN

# Global configuration object
_C = CN()

# Model configuration
_C.MODEL = CN()
_C.MODEL.NAME = 'vit_base_patch16_224'
_C.MODEL.TUBELET_SIZE = 2
_C.MODEL.INPUT_SIZE = 224
_C.MODEL.DROP = 0.0
_C.MODEL.ATTN_DROP_RATE = 0.0
_C.MODEL.DROP_PATH = 0.1
_C.MODEL.DEPTH = None
_C.MODEL.USE_CHECKPOINT = False
_C.MODEL.USE_MEAN_POOLING = False
_C.MODEL.INIT_SCALE = 0.001
_C.MODEL.WITH_CP = False
_C.MODEL.COS_ATTN = False

# Masking configuration
_C.MODEL.MASK_TYPE = 'tube'
_C.MODEL.WINDOW_SIZE = [8, 14, 14]
_C.MODEL.MASK_RATIO = 0.75
_C.MODEL.PART_WIN_SIZE = [8, 7, 7]
_C.MODEL.PART_APPLY_SYMMETRY = True

# Attention configuration
_C.MODEL.ATTN_TYPE = 'local_global'
_C.MODEL.LG_REGION_SIZE = [2, 2, 10]
_C.MODEL.LG_FIRST_ATTN_TYPE = 'self'
_C.MODEL.LG_THIRD_ATTN_TYPE = 'cross'
_C.MODEL.LG_ATTN_PARAM_SHARING_FIRST_THIRD = False
_C.MODEL.LG_ATTN_PARAM_SHARING_ALL = False
_C.MODEL.LG_CLASSIFY_TOKEN_TYPE = 'org'
_C.MODEL.LG_NO_SECOND = False
_C.MODEL.LG_NO_THIRD = False

# Data configuration
_C.DATA = CN()
_C.DATA.DATASET_NAME = 'Kinetics-400'
_C.DATA.DATA_PATH = '/path/to/data'
_C.DATA.EVAL_DATA_PATH = None
_C.DATA.NUM_CLASSES = 400
_C.DATA.NUM_SEGMENTS = 1
_C.DATA.NUM_FRAMES = 16
_C.DATA.SAMPLING_RATE = 4
_C.DATA.IMAGENET_DEFAULT_MEAN_AND_STD = True
_C.DATA.BATCH_SIZE = 64
_C.DATA.NUM_WORKERS = 4
_C.DATA.PIN_MEMORY = True
_C.DATA.TEST_NUM_SEGMENT = 5
_C.DATA.TEST_NUM_CROP = 3
_C.DATA.SHORT_SIDE_SIZE = 320

# Augmentation configuration
_C.AUGMENTATION = CN()
_C.AUGMENTATION.COLOR_JITTER = 0.4
_C.AUGMENTATION.TRAIN_INTERPOLATION = 'bicubic'
_C.AUGMENTATION.NUM_SAMPLE = 2
_C.AUGMENTATION.AUTO_AUGMENT = 'rand-m7-n4-mstd0.5-inc1'
_C.AUGMENTATION.MIXUP = 0.8
_C.AUGMENTATION.CUTMIX = 1.0
_C.AUGMENTATION.CUTMIX_MINMAX = None
_C.AUGMENTATION.MIXUP_PROB = 1.0
_C.AUGMENTATION.MIXUP_SWITCH_PROB = 0.5
_C.AUGMENTATION.MIXUP_MODE = 'batch'
_C.AUGMENTATION.RANDOM_ERASE_PROB = 0.25
_C.AUGMENTATION.RANDOM_ERASE_MODE = 'pixel'
_C.AUGMENTATION.RANDOM_ERASE_COUNT = 1
_C.AUGMENTATION.RANDOM_ERASE_SPLIT = False
_C.AUGMENTATION.LABEL_SMOOTHING = 0.1
_C.AUGMENTATION.NO_AUGMENTATION = False

# mixup_fn = Mixup(
#             mixup_alpha=cfg.AUGMENTATION.MIXUP_ALPHA,
#             cutmix_alpha=cfg.AUGMENTATION.CUTMIX_ALPHA,
#             cutmix_minmax=cfg.AUGMENTATION.CUTMIX_MINMAX,
#             prob=cfg.AUGMENTATION.MIXUP_PROB,
#             switch_prob=cfg.AUGMENTATION.MIXUP_SWITCH_PROB,
#             mode=cfg.AUGMENTATION.MIXUP_MODE,
#             label_smoothing=cfg.AUGMENTATION.SMOOTHING,
#             num_classes=cfg.MODEL.NUM_CLASSES
#         )

# Optimization configuration
_C.OPTIMIZATION = CN()
_C.OPTIMIZATION.OPTIMIZER = 'adamw'
_C.OPTIMIZATION.SCHED = 'cosine'
_C.OPTIMIZATION.LR = 1e-3
_C.OPTIMIZATION.MIN_LR = 1e-6
_C.OPTIMIZATION.WARMUP_LR = 1e-6
_C.OPTIMIZATION.WEIGHT_DECAY = 0.05
_C.OPTIMIZATION.WEIGHT_DECAY_END = None
_C.OPTIMIZATION.WARMUP_EPOCHS = 5
_C.OPTIMIZATION.WARMUP_STEPS = -1
_C.OPTIMIZATION.OPT_EPS = 1e-8
_C.OPTIMIZATION.OPT_BETAS = [0.9, 0.999]
_C.OPTIMIZATION.MOMENTUM = 0.9
_C.OPTIMIZATION.CLIP_GRAD = None
_C.OPTIMIZATION.LAYER_DECAY = 0.75

# Training configuration
_C.TRAINING = CN()
_C.TRAINING.EPOCHS = 30
_C.TRAINING.START_EPOCH = 0
_C.TRAINING.UPDATE_FREQ = 1
_C.TRAINING.SAVE_CKPT_FREQ = 100
_C.TRAINING.MODELEMA_ = False
_C.TRAINING.MODEL_EMA_DECAY = 0.9999
_C.TRAINING.MODEL_EMA_FORCE_CPU = False
_C.TRAINING.DISABLE_EVAL_DURING_FINETUNING = False
_C.TRAINING.EVAL_ONLY = False
_C.TRAINING.DIST_EVAL = False
_C.TRAINING.SAVE_CKPT = True
_C.TRAINING.AUTO_RESUME = True
_C.TRAINING.RESUME = ''
_C.TRAINING.FINETUNE = ''
_C.TRAINING.VAL_METRIC = 'acc1'

# Gaze estimation configuration
_C.GAZE = CN()
_C.GAZE.USE_L2CS = False
_C.GAZE.NUM_BINS = 90
_C.GAZE.ALPHA_REG = 1.0
_C.GAZE.BIN_WIDTH = 2.0

# System configuration, and distributed training settings
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE = 'cuda'
_C.SYSTEM.SEED = 0
_C.SYSTEM.OUTPUT_DIR = ''
_C.SYSTEM.LOG_DIR = None
_C.SYSTEM.WORLD_SIZE = 1
_C.SYSTEM.LOCAL_RANK = -1
_C.SYSTEM.DIST_ON_ITP = False
_C.SYSTEM.DIST_URL = 'env://'
_C.SYSTEM.ENABLE_DEEPSPEED = False
_C.SYSTEM.SAVE_FEATURE = False
_C.SYSTEM.GPU = -1  # Default GPU, will be set in init_distributed_mode
# _C.SYSTEM.RANK = 0  # Default rank, will be set in init_distributed_mode
_C.SYSTEM.DISTRIBUTED = False  # Default distributed mode, will be set in init_distributed_mode
_C.SYSTEM.DIST_BACKEND = 'nccl'  # Default backend for distributed training


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()



def merge_config_file(cfg, yaml_file):
    """Merge configuration from YAML file."""
    with open(yaml_file, 'r') as f:
        yaml_cfg = yaml.safe_load(f)
    
    # Recursively merge YAML config
    def merge_dict(cfg_node, yaml_dict):
        for key, value in yaml_dict.items():
            if key.upper() in cfg_node:
                if isinstance(value, dict):
                    merge_dict(cfg_node[key.upper()], value)
                else:
                    cfg_node[key.upper()] = value
    
    merge_dict(cfg, yaml_cfg)
    return cfg


def setup_config():
    """Setup configuration from command line arguments."""
    parser = argparse.ArgumentParser(description='Training configuration')
    parser.add_argument('--config-file', type=str, default='',
                        help='path to config file')
    parser.add_argument('--opts', nargs=argparse.REMAINDER, default=[],
                        help='Modify config options using the command-line')
    
    args = parser.parse_args()
    
    # Create config
    cfg = get_cfg_defaults()
    
    # Merge from file
    if args.config_file:
        cfg = merge_config_file(cfg, args.config_file)
    
    # Merge from command line
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Make config immutable
    cfg.freeze()
    
    return cfg


# Global config instance
cfg = None


def get_cfg():
    """Get the global config instance."""
    global cfg
    if cfg is None:
        cfg = get_cfg_defaults()
    return cfg


def set_cfg(new_cfg):
    """Set the global config instance."""
    global cfg
    cfg = new_cfg


def freeze_cfg():
    """
    Freeze the global configuration to make it immutable.
    Call this after loading and merging all configuration files.
    """
    global cfg
    if cfg is not None:
        cfg.freeze()
        print("Configuration has been frozen and is now immutable.")
    else:
        print("Warning: No configuration to freeze.")


def is_cfg_frozen():
    """Check if the global configuration is frozen."""
    global cfg
    if cfg is not None:
        return cfg.is_frozen()
    return False


def reset_cfg():
    """Reset the global configuration to default values."""
    global cfg
    cfg = get_cfg_defaults()
    print("Configuration has been reset to default values.")


def load_and_freeze_config(yaml_file=None):
    """
    Convenience function to load configuration from YAML file and freeze it.
    This is the recommended way to initialize configuration for training.
    
    Args:
        yaml_file (str, optional): Path to YAML configuration file
    """
    cfg = get_cfg()
    
    if yaml_file:
        print(f"Loading configuration from: {yaml_file}")
        merge_config_file(cfg, yaml_file)
    
    freeze_cfg()
    return cfg


# Legacy compatibility functions
def create_config_from_args(args: argparse.Namespace):
    """Create configuration from argparse arguments (for backward compatibility)."""
    cfg = get_cfg_defaults()
    cfg = update_config(cfg, args)
    return cfg


def update_config_from_args(cfg, args):
    """Update config from command line arguments for backward compatibility."""
    if hasattr(args, 'model') and args.model:
        cfg.MODEL.NAME = args.model
    if hasattr(args, 'tubelet_size') and args.tubelet_size:
        cfg.MODEL.TUBELET_SIZE = args.tubelet_size
    if hasattr(args, 'input_size') and args.input_size:
        cfg.MODEL.INPUT_SIZE = args.input_size
    if hasattr(args, 'drop') and args.drop is not None:
        cfg.MODEL.DROP = args.drop
    if hasattr(args, 'attn_drop_rate') and args.attn_drop_rate is not None:
        cfg.MODEL.ATTN_DROP_RATE = args.attn_drop_rate
    if hasattr(args, 'drop_path') and args.drop_path is not None:
        cfg.MODEL.DROP_PATH = args.drop_path
    if hasattr(args, 'depth') and args.depth is not None:
        cfg.MODEL.DEPTH = args.depth
    
    # Data config
    if hasattr(args, 'data_set') and args.data_set:
        cfg.DATA.DATASET_NAME = args.data_set
    if hasattr(args, 'data_path') and args.data_path:
        cfg.DATA.DATA_PATH = args.data_path
    if hasattr(args, 'eval_data_path') and args.eval_data_path:
        cfg.DATA.EVAL_DATA_PATH = args.eval_data_path
    if hasattr(args, 'nb_classes') and args.nb_classes:
        cfg.DATA.NUM_CLASSES = args.nb_classes
    if hasattr(args, 'num_segments') and args.num_segments:
        cfg.DATA.NUM_SEGMENTS = args.num_segments
    if hasattr(args, 'num_frames') and args.num_frames:
        cfg.DATA.NUM_FRAMES = args.num_frames
    if hasattr(args, 'sampling_rate') and args.sampling_rate:
        cfg.DATA.SAMPLING_RATE = args.sampling_rate
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.DATA.BATCH_SIZE = args.batch_size
    if hasattr(args, 'num_workers') and args.num_workers:
        cfg.DATA.NUM_WORKERS = args.num_workers
    if hasattr(args, 'pin_mem') and args.pin_mem is not None:
        cfg.DATA.PIN_MEMORY = args.pin_mem
    
    # Augmentation config
    if hasattr(args, 'color_jitter') and args.color_jitter is not None:
        cfg.AUGMENTATION.COLOR_JITTER = args.color_jitter
    if hasattr(args, 'train_interpolation') and args.train_interpolation:
        cfg.AUGMENTATION.TRAIN_INTERPOLATION = args.train_interpolation
    if hasattr(args, 'num_sample') and args.num_sample:
        cfg.AUGMENTATION.NUM_SAMPLE = args.num_sample
    if hasattr(args, 'aa') and args.aa:
        cfg.AUGMENTATION.AUTO_AUGMENT = args.aa
    if hasattr(args, 'mixup') and args.mixup is not None:
        cfg.AUGMENTATION.MIXUP = args.mixup
    if hasattr(args, 'cutmix') and args.cutmix is not None:
        cfg.AUGMENTATION.CUTMIX = args.cutmix
    if hasattr(args, 'cutmix_minmax') and args.cutmix_minmax:
        cfg.AUGMENTATION.CUTMIX_MINMAX = args.cutmix_minmax
    if hasattr(args, 'mixup_prob') and args.mixup_prob is not None:
        cfg.AUGMENTATION.MIXUP_PROB = args.mixup_prob
    if hasattr(args, 'mixup_switch_prob') and args.mixup_switch_prob is not None:
        cfg.AUGMENTATION.MIXUP_SWITCH_PROB = args.mixup_switch_prob
    if hasattr(args, 'mixup_mode') and args.mixup_mode:
        cfg.AUGMENTATION.MIXUP_MODE = args.mixup_mode
    if hasattr(args, 'reprob') and args.reprob is not None:
        cfg.AUGMENTATION.RANDOM_ERASE_PROB = args.reprob
    if hasattr(args, 'remode') and args.remode:
        cfg.AUGMENTATION.RANDOM_ERASE_MODE = args.remode
    if hasattr(args, 'recount') and args.recount:
        cfg.AUGMENTATION.RANDOM_ERASE_COUNT = args.recount
    if hasattr(args, 'resplit') and args.resplit is not None:
        cfg.AUGMENTATION.RANDOM_ERASE_SPLIT = args.resplit
    if hasattr(args, 'smoothing') and args.smoothing is not None:
        cfg.AUGMENTATION.LABEL_SMOOTHING = args.smoothing
    
    # Optimization config
    if hasattr(args, 'opt') and args.opt:
        cfg.OPTIMIZATION.OPTIMIZER = args.opt
    if hasattr(args, 'lr') and args.lr is not None:
        cfg.OPTIMIZATION.LR = args.lr
    if hasattr(args, 'min_lr') and args.min_lr is not None:
        cfg.OPTIMIZATION.MIN_LR = args.min_lr
    if hasattr(args, 'warmup_lr') and args.warmup_lr is not None:
        cfg.OPTIMIZATION.WARMUP_LR = args.warmup_lr
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        cfg.OPTIMIZATION.WEIGHT_DECAY = args.weight_decay
    if hasattr(args, 'weight_decay_end') and args.weight_decay_end is not None:
        cfg.OPTIMIZATION.WEIGHT_DECAY_END = args.weight_decay_end
    if hasattr(args, 'warmup_epochs') and args.warmup_epochs is not None:
        cfg.OPTIMIZATION.WARMUP_EPOCHS = args.warmup_epochs
    if hasattr(args, 'warmup_steps') and args.warmup_steps is not None:
        cfg.OPTIMIZATION.WARMUP_STEPS = args.warmup_steps
    if hasattr(args, 'opt_eps') and args.opt_eps is not None:
        cfg.OPTIMIZATION.OPT_EPS = args.opt_eps
    if hasattr(args, 'opt_betas') and args.opt_betas:
        cfg.OPTIMIZATION.OPT_BETAS = list(args.opt_betas)
    if hasattr(args, 'momentum') and args.momentum is not None:
        cfg.OPTIMIZATION.MOMENTUM = args.momentum
    if hasattr(args, 'clip_grad') and args.clip_grad is not None:
        cfg.OPTIMIZATION.CLIP_GRAD = args.clip_grad
    if hasattr(args, 'layer_decay') and args.layer_decay is not None:
        cfg.OPTIMIZATION.LAYER_DECAY = args.layer_decay
    
    # Training config
    if hasattr(args, 'epochs') and args.epochs:
        cfg.TRAINING.EPOCHS = args.epochs
    if hasattr(args, 'start_epoch') and args.start_epoch:
        cfg.TRAINING.START_EPOCH = args.start_epoch
    if hasattr(args, 'update_freq') and args.update_freq:
        cfg.TRAINING.UPDATE_FREQ = args.update_freq
    if hasattr(args, 'save_ckpt_freq') and args.save_ckpt_freq:
        cfg.TRAINING.SAVE_CKPT_FREQ = args.save_ckpt_freq
    if hasattr(args, 'model_ema') and args.model_ema is not None:
        cfg.TRAINING.MODEL_EMA = args.model_ema
    if hasattr(args, 'model_ema_decay') and args.model_ema_decay is not None:
        cfg.TRAINING.MODEL_EMA_DECAY = args.model_ema_decay
    if hasattr(args, 'model_ema_force_cpu') and args.model_ema_force_cpu is not None:
        cfg.TRAINING.MODEL_EMA_FORCE_CPU = args.model_ema_force_cpu
    if hasattr(args, 'disable_eval_during_finetuning') and args.disable_eval_during_finetuning is not None:
        cfg.TRAINING.DISABLE_EVAL_DURING_FINETUNING = args.disable_eval_during_finetuning
    if hasattr(args, 'eval') and args.eval is not None:
        cfg.TRAINING.EVAL_ONLY = args.eval
    if hasattr(args, 'dist_eval') and args.dist_eval is not None:
        cfg.TRAINING.DIST_EVAL = args.dist_eval
    if hasattr(args, 'save_ckpt') and args.save_ckpt is not None:
        cfg.TRAINING.SAVE_CKPT = args.save_ckpt
    if hasattr(args, 'auto_resume') and args.auto_resume is not None:
        cfg.TRAINING.AUTO_RESUME = args.auto_resume
    if hasattr(args, 'resume') and args.resume:
        cfg.TRAINING.RESUME = args.resume
    if hasattr(args, 'finetune') and args.finetune:
        cfg.TRAINING.FINETUNE = args.finetune
    if hasattr(args, 'val_metric') and args.val_metric:
        cfg.TRAINING.VAL_METRIC = args.val_metric
    
    # Gaze config
    if hasattr(args, 'use_l2cs') and args.use_l2cs is not None:
        cfg.GAZE.USE_L2CS = args.use_l2cs
    if hasattr(args, 'num_bins') and args.num_bins is not None:
        cfg.GAZE.NUM_BINS = args.num_bins
    if hasattr(args, 'alpha_reg') and args.alpha_reg is not None:
        cfg.GAZE.ALPHA_REG = args.alpha_reg
    if hasattr(args, 'bin_width') and args.bin_width is not None:
        cfg.GAZE.BIN_WIDTH = args.bin_width
    
    # System config
    if hasattr(args, 'device') and args.device:
        cfg.SYSTEM.DEVICE = args.device
    if hasattr(args, 'seed') and args.seed is not None:
        cfg.SYSTEM.SEED = args.seed
    if hasattr(args, 'output_dir') and args.output_dir:
        cfg.SYSTEM.OUTPUT_DIR = args.output_dir
    if hasattr(args, 'log_dir') and args.log_dir:
        cfg.SYSTEM.LOG_DIR = args.log_dir
    if hasattr(args, 'world_size') and args.world_size:
        cfg.SYSTEM.WORLD_SIZE = args.world_size
    if hasattr(args, 'local_rank') and args.local_rank is not None:
        cfg.SYSTEM.LOCAL_RANK = args.local_rank
    if hasattr(args, 'dist_on_itp') and args.dist_on_itp is not None:
        cfg.SYSTEM.DIST_ON_ITP = args.dist_on_itp
    if hasattr(args, 'dist_url') and args.dist_url:
        cfg.SYSTEM.DIST_URL = args.dist_url
    if hasattr(args, 'enable_deepspeed') and args.enable_deepspeed is not None:
        cfg.SYSTEM.ENABLE_DEEPSPEED = args.enable_deepspeed
    if hasattr(args, 'save_feature') and args.save_feature is not None:
        cfg.SYSTEM.SAVE_FEATURE = args.save_feature
    
    return cfg
