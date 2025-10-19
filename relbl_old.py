
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
# sys.path.insert(0, str(Path(__file__).parent))

# sys.path.append("../../")

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

from run_finetuning_with_yacs import create_data_loaders, create_model_from_config


from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class relblDataset(Dataset):
    def __init__(self, all_data):
        self.all_data = all_data

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        item = self.all_data[idx]
        images = np.array(item["images"])
        label = item["label"]
        folder = item["folder"]
        # if self.transform:
        #     images = [self.transform(image=image)["image"] for image in images]
        img_tensor = torch.tensor(images)/255.0
        img_tensor = img_tensor.unsqueeze(0)
        # data_transform = video_transforms.Compose([
        #     # video_transforms.Resize(size=(160, 160), interpolation='bilinear'),
        #     # volume_transforms.ClipToTensor(),
        #     video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                 std=[0.229, 0.224, 0.225])
        # ])
        # img_tensor = data_transform(img_tensor)
        # print("shape of img_tensor", img_tensor.shape) # ([1, 16, 224, 224, 3])
        # interpolate to 160*160, but use nn.interpolate
        img_tensor = F.interpolate(img_tensor, size = (160, 160, 3))
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 1, 3)
        img_tensor = (img_tensor - mean) / std
        
        img_tensor = img_tensor.permute(0, 4, 1, 2, 3)  # Change to [B, C, T, H, W]
        img_tensor = img_tensor.squeeze(0)
        return {"images": img_tensor, "label": label, "folder": folder}

for split in ["train", "val"]:

    all_data = []

    csv_path = f"/home/qzk/Facial-Foundation-Model/saved/data/dfew_224/org/split01/{split}.csv"
    all_folders = []

    import pandas as pd
    import cv2
    df = pd.read_csv(csv_path)
    folder_lbls = list(df.values[:,0])

    # folder_lbls = folder_lbls[:10] # to debug

    all_folders = [folder_lbl.split(" ")[0] for folder_lbl in folder_lbls]
    all_lbls = [folder_lbl.split(" ")[1:] for folder_lbl in folder_lbls]
    print("we have ", len(all_folders), " folders in total, like", all_folders[0])
    print("we have ", len(all_lbls), " lbls in total, like", all_lbls[0])


    for i, folder in enumerate(all_folders):
        if not os.path.isdir(folder):
            continue
        imgs = []
        all_imgs_path = os.listdir(folder)
        # sort by decimal
        all_imgs_path.sort(key=lambda x: int(os.path.splitext(x)[0]))
        # print(all_imgs_path)
        for img_path in all_imgs_path:
            img = cv2.imread(os.path.join(folder, img_path))
            if img is not None:
                imgs.append(img)
        all_data.append({"folder": str(all_folders[i]), "images": imgs, "label": all_lbls[i]})

    print("len of all_data", len(all_data))

    data_loader = DataLoader(relblDataset(all_data), batch_size=32, shuffle=False, num_workers=20)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config_path = 'configs/gaze360T.yaml'
    cfg = get_cfg()
    merge_config_file(cfg, config_path)
    model = create_model_from_config()
    model_path = "/home/qzk/Facial-Foundation-Model/output/gaze360T_16dataset/checkpoint-best.pth"
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = state_dict["model"]
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    out_csv_path = csv_path.replace(f"{split}", f"{split}_withGaze360")

    print("relabling data...")

    for j, data in enumerate(data_loader):
        if j % 100 ==0:
            print(f"relabeling {j} out of {len(data_loader)} data")
        images = data["images"].to(device)
        labels = data["label"][0]
        folders = data["folder"]
        with torch.no_grad():
            output = model(images)
            # print(output.shape)
            output=output.reshape(images.shape[0], images.shape[2], 2)
            data["gaze"] = output.detach().cpu().numpy()

        # print(images.shape)
        # print(folders)
        # print(labels)
        for i in range(images.shape[0]):
            for t in range(images.shape[2]):
                gaze = data["gaze"][i][t]
                folder = folders[i]
                label = labels[i]
                with open(out_csv_path, "a") as f:
                    f.write(f"{os.path.join(folder, f"{t}.jpg")} {label} {gaze[0]} {gaze[1]}\n")

