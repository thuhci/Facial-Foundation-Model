
import os
# import sys
# import argparse
# import datetime
import numpy as np
# import time
import torch

import cv2

from src.utils.config import get_cfg, merge_config_file, freeze_cfg, load_and_freeze_config


from run_finetuning_with_yacs import create_data_loaders, create_model_from_config


all_data = []

# your split-by-video folder
csv_folder = "/home/qzk/CelebV-Text/downloaded_celebvtext/txt"  

# config & path for your relabel model
config_path = '/home/qzk/Facial-Foundation-Model/output/dfew_pure_10_11_b/config.yaml'
model_path = "/home/qzk/Facial-Foundation-Model/output/dfew_pure_10_11_b/checkpoint-best.pth"

# output path for the relabeled data
out_csv_path = "output/relbl/dfew_CelebV_my.csv"

# read all csv  N frames
# split by 16 frames, with name & lbl for each

import pandas as pd
# import cv2
import imageio


all_csvs = os.listdir(csv_folder)
processed_count = 0
for csv in all_csvs:
    if not ( csv.endswith(".csv") or csv.endswith(".txt")):
        continue
    
    # processed_count += 1
    # print(f"Processing CSV {processed_count}: {csv}")

    with open(os.path.join(csv_folder, csv), "r") as f:
        lines = f.readlines()
        folders = []
        labels = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" ")
            if len(parts) < 2:
                continue
            folder = parts[0]
            lbls = parts[1:]
            folders.append(folder)
            labels.append(lbls)
        for i in range(0, len(folders), 16):
            if i + 16 > len(folders):
                break
            all_data.append({"folder": folders[i:i+16], "labels": labels[i:i+16]})
    

print(f"we have {len(all_data)} * 16 items in total")

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# 16 paths & lbls per item
class relblDataset(Dataset):
    def __init__(self, all_data):
        self.all_data = all_data
        # Initialize OpenCV
        cv2.setNumThreads(1)  # Force single-threaded OpenCV

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        
        paths = all_data[idx]["folder"]
        labels = all_data[idx]["labels"]


        images = []
        for i, path in enumerate(paths):
            try:
                # Check if file exists
                if not os.path.exists(path):
                    print(f"Warning: File does not exist: {path}")
                    # Use a black placeholder image
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                    images.append(img)
                    continue
                
                # Check file permissions and size
                file_stat = os.stat(path)
                if file_stat.st_size == 0:
                    print(f"Warning: Empty file: {path}")
                    img = np.zeros((224, 224, 3), dtype=np.uint8)
                    images.append(img)
                    continue
                
                # Try to read the image with more specific error handling
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"Warning: cv2.imread returned None for: {path}")
                    img = np.zeros((224, 224, 3), dtype=np.uint8)

                # Check if image has correct shape
                if len(img.shape) != 3 or img.shape[2] != 3:
                    print(f"Warning: Invalid image shape {img.shape} for: {path}")
                    # Resize to expected format
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Resize to standard size to avoid memory issues
                img = cv2.resize(img, (224, 224))
                images.append(img)
                # if i < 3:  # Only print for first few images to avoid spam
                #     print(f"Successfully read img {i}: {img.shape}")
                
            except Exception as e:
                print(f"Error reading image {path}: {str(e)}")
                print(f"Exception type: {type(e).__name__}")
                # Use a black placeholder image
                img = np.zeros((224, 224, 3), dtype=np.uint8)
                images.append(img)
        
        # Ensure we have exactly 16 images
        while len(images) < 16:
            images.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # images = images[:16]  # Truncate if we have more than 16
        images = np.array(images)        
        # print("Final image array shape:", images.shape)
        
        img_tensor = torch.tensor(images, dtype=torch.float32)/255.0
        # print("shape of img_tensor before processing:", img_tensor.shape) # ([16, 224, 224, 3])
        
        # Convert to [T, C, H, W] format first
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # [T, C, H, W]
        # print("shape after permute:", img_tensor.shape) # ([16, 3, 224, 224])
        
        # Resize each frame to 160x160
        img_tensor = F.interpolate(img_tensor, size=(160, 160), mode='bilinear', align_corners=False)
        # print("shape after interpolate:", img_tensor.shape) # ([16, 3, 160, 160])
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        # Final shape should be [C, T, H, W]
        img_tensor = img_tensor.permute(1, 0, 2, 3)  # [C, T, H, W]
        # print("Final img_tensor shape:", img_tensor.shape) # ([3, 16, 160, 160])
        
        return {"images": img_tensor, "labels": labels, "folder": paths}

data_loader = DataLoader(relblDataset(all_data), batch_size=256, shuffle=False, num_workers=20)

print("len of data_loader is ", len(data_loader))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



cfg = get_cfg()
merge_config_file(cfg, config_path)


model = create_model_from_config()


state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
state_dict = state_dict["model"]
state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

# out_csv_path = csv_path.replace("val", "val_withGaze360")

print("relabling data...")

relabeled_data = {}

for j, data in enumerate(data_loader):
    if j % 100 ==0:
        print(f"relabeling {j} out of {len(data_loader)} data")
    images = data["images"].to(device)
    labels = data["labels"]
    folders = data["folder"]
    with torch.no_grad():
        output = model(images)
        # print(output.shape)
        output_cls=output.unsqueeze(1)
        output_cls=output_cls.repeat(1,images.shape[2],1)
        output_cls = output_cls.reshape(images.shape[0], images.shape[2], -1).argmax(dim=-1)
        data["emo"] = output_cls.detach().cpu().numpy()

    # print("[DEBUG]", images.shape)
    # print("[DEBUG len folder]", len(folders))
    # print("[DEBUG len labels]", len(labels), len(labels[0]))
    # print("[DEBUG]", folders[1])
    # print("[DEBUG]", labels)
    
    
    for i in range(images.shape[0]):
        for t in range(images.shape[2]):
            emo = data["emo"][i][t]
            relabeled_data[folders[t][i]] = [labels[t][0][i], emo]
            # folder = folders[i]
            # label = labels[i]
            # with open(out_csv_path, "a") as f:
            #     f.write(f"{os.path.join(folder, f"{t}.jpg")} {label} {gaze[0]} {gaze[1]}\n")
            
    # break

# print("relbled_data", relabeled_data)


with open(out_csv_path, "w") as f:
    for folder, lbl in relabeled_data.items():
        f.write(f"{folder} {lbl[0]} {lbl[1]}\n")
