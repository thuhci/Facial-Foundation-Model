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


# def init_distributed_mode(args):
#     if args.dist_on_itp:
#         args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
#         args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
#         args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
#         args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['RANK'] = str(args.rank)
#         os.environ['WORLD_SIZE'] = str(args.world_size)
#     elif 'SLURM_PROCID' in os.environ:
#         args.rank = int(os.environ['SLURM_PROCID'])
#         args.gpu = int(os.environ['SLURM_LOCALID'])
#         args.world_size = int(os.environ['SLURM_NTASKS'])
#         os.environ['RANK'] = str(args.rank)
#         os.environ['LOCAL_RANK'] = str(args.gpu)
#         os.environ['WORLD_SIZE'] = str(args.world_size)

#         node_list = os.environ['SLURM_NODELIST']
#         addr = subprocess.getoutput(
#             f'scontrol show hostname {node_list} | head -n1')
#         if 'MASTER_ADDR' not in os.environ:
#             os.environ['MASTER_ADDR'] = addr
#     elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ['WORLD_SIZE'])
#         args.gpu = int(os.environ['LOCAL_RANK'])
#     else:
#         print('Not using distributed mode')
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = 'nccl'
#     print('| distributed init (rank {}): {}, gpu {}'.format(
#         args.rank, args.dist_url, args.gpu), flush=True)
#     torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
#                                          world_size=args.world_size, rank=args.rank)
#     torch.distributed.barrier()
#     # assert torch.distributed.is_initialized()
#     setup_for_distributed(args.rank == 0)