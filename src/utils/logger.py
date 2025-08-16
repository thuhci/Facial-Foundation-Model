import os
import pickle
import numpy as np
import random
import io
from collections import defaultdict, deque
import torch
import torch.distributed as dist
from tensorboardX import SummaryWriter
from typing import Dict, List, Tuple, Any, Union, Optional
import datetime
from src.utils.utils import is_dist_avail_and_initialized
import time

# Import wandb with error handling
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Install with 'pip install wandb' to enable wandb logging.")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        """Log metrics to tensorboard with head as prefix."""
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    continue
            assert isinstance(v, (float, int))
            # Use provided step or current step
            current_step = step if step is not None else self.step
            self.writer.add_scalar(head + "/" + k, v, current_step)

    def flush(self):
        self.writer.flush()


class WandbLogger(object):
    """Wandb logger that integrates with the existing logging system."""
    
    def __init__(self):
        self.run = None
        self.step = 0
        self.last_step = 0  # Track last logged step to avoid going backwards
        self.initialized = False
        
    def init_wandb(self, config_dict: Optional[Dict] = None, resume_id: Optional[str] = None):
        """Initialize wandb run."""
        if not WANDB_AVAILABLE:
            print("Wandb not available, skipping wandb initialization")
            return
            
        # Import config here to avoid circular imports
        from src.utils.config import get_cfg
        from src.utils import utils
        
        cfg = get_cfg()
        
        # Only initialize on main process
        if utils.get_rank() != 0:
            return
            
        try:
            # Prepare config for wandb
            wandb_config = self._prepare_config(config_dict)
            
            # Generate run name if not provided
            run_name = cfg.WANDB.RUN_NAME
            if run_name is None:
                run_name = self._generate_run_name()
            
            # Initialize wandb
            self.run = wandb.init(
                project=cfg.WANDB.PROJECT,
                entity=cfg.WANDB.ENTITY,
                name=run_name,
                tags=list(cfg.WANDB.TAGS) if cfg.WANDB.TAGS else None,
                notes=cfg.WANDB.NOTES,
                config=wandb_config,
                mode=cfg.WANDB.MODE,
                save_code=cfg.WANDB.SAVE_CODE,
                id=resume_id,
                resume="allow" if resume_id else None
            )
            
            self.initialized = True
            print(f"Wandb initialized: {self.run.url}")
            
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
    
    def watch_model(self, model: torch.nn.Module):
        """Watch model gradients and parameters."""
        if not self._should_log():
            return
            
        try:
            from src.utils.config import get_cfg
            cfg = get_cfg()
            
            if cfg.WANDB.WATCH_MODEL and self.run is not None:
                wandb.watch(
                    model, 
                    log=cfg.WANDB.WATCH_LOG,
                    log_freq=cfg.WANDB.LOG_FREQ
                )
        except Exception as e:
            print(f"Failed to watch model: {e}")
    
    def set_step(self, step=None):
        """Set current step number."""
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def update(self, head='scalar', step=None, **kwargs):
        """Log metrics to wandb with head as prefix."""
        if not self._should_log():
            return
            
        try:
            # Convert metrics to loggable format
            processed_metrics = {}
            for key, value in kwargs.items():
                if value is None:
                    continue
                    
                metric_name = f"{head}/{key}" if head != 'scalar' else key
                
                if isinstance(value, (torch.Tensor, np.ndarray)):
                    if hasattr(value, 'numel') and value.numel() == 1:
                        processed_metrics[metric_name] = value.item()
                    elif hasattr(value, 'size') and value.size == 1:
                        processed_metrics[metric_name] = value.item()
                    else:
                        # For multi-dimensional tensors, log as histogram
                        if isinstance(value, torch.Tensor):
                            processed_metrics[f"{metric_name}_hist"] = wandb.Histogram(value.detach().cpu().numpy())
                        else:
                            processed_metrics[f"{metric_name}_hist"] = wandb.Histogram(value)
                elif isinstance(value, (int, float)):
                    processed_metrics[metric_name] = value
                else:
                    try:
                        processed_metrics[metric_name] = float(value)
                    except (ValueError, TypeError):
                        print(f"Warning: Cannot log metric {key} with value {value}")
            
            # Log to wandb - ensure we don't go backwards in steps
            if processed_metrics:
                if step is not None:
                    # Use the provided step, but ensure it's not less than last_step
                    current_step = max(step, self.last_step)
                else:
                    # Auto increment from last step
                    current_step = self.last_step + 1
                
                # Update last_step
                self.last_step = current_step
                
                # Log to wandb
                self.run.log(processed_metrics, step=current_step)
                
        except Exception as e:
            print(f"Failed to log metrics to wandb: {e}")
    
    def log_confusion_matrix(self, y_true: List, y_pred: List, class_names: Optional[List[str]] = None,
                           step: Optional[int] = None, prefix: str = "val"):
        """Log confusion matrix to wandb."""
        if not self._should_log():
            return
            
        try:
            self.run.log({
                f"{prefix}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true,
                    preds=y_pred,
                    class_names=class_names
                )
            }, step=step or self.step)
        except Exception as e:
            print(f"Failed to log confusion matrix: {e}")
    
    def log_model_architecture(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture and parameters."""
        if not self._should_log():
            return
            
        try:
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.run.log({
                "model/total_parameters": total_params,
                "model/trainable_parameters": trainable_params,
                "model/non_trainable_parameters": total_params - trainable_params
            })
            
            # Log model summary as text
            model_summary = str(model)
            self.run.log({"model/architecture": wandb.Html(f"<pre>{model_summary}</pre>")})
            
        except Exception as e:
            print(f"Failed to log model architecture: {e}")
    
    def log_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """Log model checkpoint as wandb artifact."""
        if not self._should_log():
            return
            
        try:
            artifact_name = f"model-checkpoint-{self.run.id}"
            if is_best:
                artifact_name = f"best-{artifact_name}"
                
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                metadata={"epoch": self.step, "is_best": is_best}
            )
            
            artifact.add_file(checkpoint_path)
            self.run.log_artifact(artifact)
            
        except Exception as e:
            print(f"Failed to log checkpoint: {e}")
    
    def flush(self):
        """Flush wandb (no-op for wandb)."""
        pass
    
    def finish(self):
        """Finish wandb run."""
        if self.run is not None:
            try:
                self.run.finish()
            except Exception as e:
                print(f"Failed to finish wandb run: {e}")
    
    def _should_log(self) -> bool:
        """Check if should log to wandb."""
        if not WANDB_AVAILABLE or not self.initialized or self.run is None:
            return False
            
        try:
            from src.utils.config import get_cfg
            from src.utils import utils
            cfg = get_cfg()
            return cfg.WANDB.USE_WANDB and utils.get_rank() == 0
        except:
            return False
    
    def _prepare_config(self, additional_config: Optional[Dict] = None) -> Dict:
        """Prepare configuration dictionary for wandb."""
        from src.utils.config import get_cfg
        cfg = get_cfg()
        
        # Convert CfgNode to dict
        config_dict = {}
        
        # Add main config sections
        config_dict.update({
            "model": dict(cfg.MODEL),
            "data": dict(cfg.DATA),
            "training": dict(cfg.TRAINING),
            "optimization": dict(cfg.OPTIMIZATION),
            "augmentation": dict(cfg.AUGMENTATION),
            "system": dict(cfg.SYSTEM),
            "gaze": dict(cfg.GAZE) if hasattr(cfg, 'GAZE') else {},
            "pretraining": dict(cfg.PRETRAINING) if hasattr(cfg, 'PRETRAINING') else {}
        })
        
        # Add additional config if provided
        if additional_config:
            config_dict.update(additional_config)
            
        return config_dict
    
    def _generate_run_name(self) -> str:
        """Generate a run name based on configuration."""
        from src.utils.config import get_cfg
        cfg = get_cfg()
        
        components = []
        
        # Add model info
        if cfg.MODEL.NAME:
            components.append(cfg.MODEL.NAME.split('_')[0])  # e.g., 'vit' from 'vit_base_patch16_224'
            
        # Add task info
        if cfg.DATA.TASK:
            components.append(cfg.DATA.TASK)
            
        # Add dataset info
        if cfg.DATA.DATASET_NAME:
            dataset_name = cfg.DATA.DATASET_NAME.lower().replace('-', '').replace('_', '')
            components.append(dataset_name[:10])  # Truncate long names
            
        # Add key hyperparameters
        if cfg.OPTIMIZATION.LR:
            components.append(f"lr{cfg.OPTIMIZATION.LR}")
            
        if cfg.DATA.BATCH_SIZE:
            components.append(f"bs{cfg.DATA.BATCH_SIZE}")
            
        return "-".join(components)


class CombinedLogger(object):
    """
    Combined logger that can log to TensorBoard, Wandb, or both based on configuration.
    Provides the same interface as TensorboardLogger for easy replacement.
    """
    
    def __init__(self, log_dir: Optional[str] = None, config_dict: Optional[Dict] = None, resume_id: Optional[str] = None):
        # Import config here to avoid circular imports
        from src.utils.config import get_cfg
        from src.utils import utils
        
        self.cfg = get_cfg()
        self.step = 0
        self.global_step = 0  # Global step counter
        
        # Initialize TensorBoard logger if log_dir is provided
        self.tb_logger = None
        if log_dir is not None:
            try:
                self.tb_logger = TensorboardLogger(log_dir)
                print(f"TensorBoard logger initialized with log_dir: {log_dir}")
            except Exception as e:
                print(f"Failed to initialize TensorBoard logger: {e}")
        
        # Initialize Wandb logger if enabled
        self.wandb_logger = None
        if self.cfg.WANDB.USE_WANDB:
            try:
                self.wandb_logger = WandbLogger()
                self.wandb_logger.init_wandb(config_dict, resume_id)
            except Exception as e:
                print(f"Failed to initialize Wandb logger: {e}")
    
    def watch_model(self, model: torch.nn.Module):
        """Watch model with wandb."""
        if self.wandb_logger is not None:
            self.wandb_logger.watch_model(model)
    
    def log_model_info(self, model: torch.nn.Module, input_shape: tuple):
        """Log model architecture information."""
        if self.wandb_logger is not None:
            self.wandb_logger.log_model_architecture(model, input_shape)
    
    def log_confusion_matrix(self, y_true: List, y_pred: List, class_names: Optional[List[str]] = None,
                           step: Optional[int] = None, prefix: str = "val"):
        """Log confusion matrix to wandb."""
        if self.wandb_logger is not None:
            self.wandb_logger.log_confusion_matrix(y_true, y_pred, class_names, step, prefix)
    
    def log_checkpoint(self, checkpoint_path: str, is_best: bool = False):
        """Log checkpoint to wandb."""
        if self.wandb_logger is not None:
            self.wandb_logger.log_checkpoint(checkpoint_path, is_best)
    
    def set_step(self, step: Optional[int] = None):
        """Set step for both loggers."""
        if step is not None:
            self.step = step
            self.global_step = step
        else:
            self.step += 1
            self.global_step += 1
            
        if self.tb_logger is not None:
            self.tb_logger.set_step(self.step)
        if self.wandb_logger is not None:
            self.wandb_logger.set_step(self.step)
    
    def update(self, head: str = 'scalar', step: Optional[int] = None, **kwargs):
        """Update both loggers with metrics."""
        # Determine the step to use for TensorBoard (can use any step)
        if step is not None:
            tb_step = step
            # For wandb, ensure monotonic increasing steps
            wandb_step = max(step, self.wandb_logger.last_step + 1) if self.wandb_logger is not None else step
        else:
            tb_step = self.global_step
            wandb_step = self.global_step
            
        # Log to TensorBoard (can handle any step)
        if self.tb_logger is not None:
            self.tb_logger.update(head=head, step=tb_step, **kwargs)
        
        # Log to Wandb (with monotonic step handling)
        if self.wandb_logger is not None:
            self.wandb_logger.update(head=head, step=wandb_step, **kwargs)
    
    def flush(self):
        """Flush both loggers."""
        if self.tb_logger is not None:
            self.tb_logger.flush()
        if self.wandb_logger is not None:
            self.wandb_logger.flush()
    
    def finish(self):
        """Finish logging."""
        if self.wandb_logger is not None:
            self.wandb_logger.finish()


def create_logger(log_dir: Optional[str] = None, config_dict: Optional[Dict] = None, 
                 resume_id: Optional[str] = None) -> CombinedLogger:
    """
    Create a combined logger that supports both TensorBoard and Wandb.
    
    Args:
        log_dir: Directory for TensorBoard logs (if None, TensorBoard logging is disabled)
        config_dict: Additional configuration to log to wandb
        resume_id: Resume run ID for wandb
    
    Returns:
        CombinedLogger instance
    """
    return CombinedLogger(log_dir=log_dir, config_dict=config_dict, resume_id=resume_id)


def log_system_info(logger: Optional[CombinedLogger] = None):
    """Log system information to wandb."""
    try:
        from src.utils.config import get_cfg
        from src.utils import utils
        
        cfg = get_cfg()
        if not cfg.WANDB.USE_WANDB or utils.get_rank() != 0:
            return
            
        if logger is None or logger.wandb_logger is None:
            return
            
        import psutil
        
        system_info = {
            "system/cpu_count": psutil.cpu_count(),
            "system/memory_gb": psutil.virtual_memory().total / (1024**3),
        }
        
        # GPU info
        if torch.cuda.is_available():
            system_info.update({
                "system/gpu_count": torch.cuda.device_count(),
                "system/gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "system/cuda_version": torch.version.cuda
            })
        
        if WANDB_AVAILABLE and logger.wandb_logger.run is not None:
            logger.wandb_logger.run.log(system_info)
        
    except Exception as e:
        print(f"Failed to log system info: {e}")
        
        

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






