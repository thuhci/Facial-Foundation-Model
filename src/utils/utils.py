import io
import os
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
# from torch._six import inf # version not match, change a inf
# from torch._C import inf  # noqa: F401
import random

from tensorboardX import SummaryWriter

from src.utils.config import get_cfg



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode():
    cfg = get_cfg()
    
    # 如果配置还没有加载，先检查环境变量设置基本参数
    if not hasattr(cfg, '_content') or not cfg._content:
        # 配置尚未加载，使用默认值或环境变量
        if cfg.SYSTEM.DIST_BACKEND == 'nccl':  # 默认值
            cfg.SYSTEM.DIST_BACKEND = 'gloo'  # 临时设为 gloo 避免问题
    
    if cfg.SYSTEM.DIST_ON_ITP:
        cfg.SYSTEM.LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.SYSTEM.WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.SYSTEM.GPU = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        cfg.SYSTEM.DIST_URL = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(cfg.SYSTEM.GPU)
        os.environ['RANK'] = str(cfg.SYSTEM.LOCAL_RANK)
        os.environ['WORLD_SIZE'] = str(cfg.SYSTEM.WORLD_SIZE)
    elif 'SLURM_PROCID' in os.environ:
        cfg.SYSTEM.LOCAL_RANK = int(os.environ['SLURM_PROCID'])
        cfg.SYSTEM.GPU = int(os.environ['SLURM_LOCALID']) 
        cfg.SYSTEM.WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(cfg.SYSTEM.LOCAL_RANK)
        os.environ['LOCAL_RANK'] = str(cfg.SYSTEM.GPU)
        os.environ['WORLD_SIZE'] = str(cfg.SYSTEM.WORLD_SIZE)
        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        cfg.SYSTEM.LOCAL_RANK = int(os.environ["RANK"])
        cfg.SYSTEM.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
        cfg.SYSTEM.GPU = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        cfg.SYSTEM.DISTRIBUTED = False
        return

    cfg.SYSTEM.DISTRIBUTED = True

    # 设置 GPU 设备（无论什么后端都需要设置正确的设备）
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.SYSTEM.GPU)
        # Gloo 后端可以使用 GPU 模型进行计算，只是通信用 CPU
        # NCCL 后端全部用 GPU
    
    print('| distributed init (rank {}): {}, gpu {}, backend {}'.format(
        cfg.SYSTEM.LOCAL_RANK, cfg.SYSTEM.DIST_URL, cfg.SYSTEM.GPU, cfg.SYSTEM.DIST_BACKEND), flush=True)
    
    # 初始化进程组
    torch.distributed.init_process_group(
        backend=cfg.SYSTEM.DIST_BACKEND, 
        init_method=cfg.SYSTEM.DIST_URL,
        world_size=cfg.SYSTEM.WORLD_SIZE, 
        rank=cfg.SYSTEM.LOCAL_RANK
    )
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(cfg.SYSTEM.LOCAL_RANK == 0)

def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    cfg = get_cfg()
    output_dir = Path(cfg.SYSTEM.OUTPUT_DIR)
    epoch_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'cfg': cfg.dump(),
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)
        model.save_checkpoint(save_dir=output_dir,
                              tag="checkpoint-%s" % epoch_name, client_state=client_state)


def auto_load_model(model, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    cfg = get_cfg()
    output_dir = Path(cfg.SYSTEM.OUTPUT_DIR)
    if loss_scaler is not None:
        # torch.amp
        if cfg.TRAINING.AUTO_RESUME and len(cfg.TRAINING.RESUME) == 0:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                cfg.TRAINING.RESUME = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % cfg.TRAINING.RESUME)

        if cfg.TRAINING.RESUME:
            if cfg.TRAINING.RESUME.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    cfg.TRAINING.RESUME, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(cfg.TRAINING.RESUME, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'])
            print("Resume checkpoint %s" % cfg.TRAINING.RESUME)
            if 'optimizer' in checkpoint and 'epoch' in checkpoint and not cfg.TRAINING.EVAL_ONLY:
                optimizer.load_state_dict(checkpoint['optimizer'])
                # me: handling to eval best checkpoint
                if checkpoint['epoch'] != 'best':
                    cfg.TRAINING.START_EPOCH = checkpoint['epoch'] + 1
                else:
                    assert cfg.TRAINING.EVAL_ONLY, f"Error: can not resume from 'best' checkpoint because checkpoint['epoch'] is not available. "
                if cfg.TRAINING.MODELEMA_:
                    _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                if 'scaler' in checkpoint:
                    loss_scaler.load_state_dict(checkpoint['scaler'])
                print("With optim & sched!")
    else:
        # deepspeed, only support '--auto_resume'.
        if cfg.TRAINING.AUTO_RESUME:
            import glob
            all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
            latest_ckpt = -1
            for ckpt in all_checkpoints:
                t = ckpt.split('-')[-1].split('.')[0]
                if t.isdigit():
                    latest_ckpt = max(int(t), latest_ckpt)
            if latest_ckpt >= 0:
                cfg.TRAINING.RESUME = os.path.join(output_dir, 'checkpoint-%d' % latest_ckpt)
                print("Auto resume checkpoint: %d" % latest_ckpt)
                _, client_states = model.load_checkpoint(cfg.SYSTEM.OUTPUT_DIR, tag='checkpoint-%d' % latest_ckpt)
                cfg.TRAINING.START_EPOCH = client_states['epoch'] + 1
                if model_ema is not None:
                    if cfg.TRAINING.MODELEMA_:
                        _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


# [TO BE MODIFIED]
def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))

def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data





