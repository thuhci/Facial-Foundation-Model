import os
import pandas as pd
import sys
import h5py


data_root = "/root/shared/GazeDatasets/eve_dataset"

for split in ["train", "val"]:
# for split in ["val"]:
    txt_lines = []
    # split_dirs = os.find(os.path.join(data_root, split+"*"))
    split_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if d.startswith(split)]
    for split_dir in split_dirs:
        step_dirs = os.listdir(split_dir)
        for step_dir in step_dirs:
            video_dir = os.path.join(split_dir, step_dir, "basler_face.mp4")
            if not os.path.exists(video_dir):
                continue

            # txt_lines.append((video_dir, 0))
            # read the video and get the number of frames
            with h5py.File(os.path.join(split_dir, step_dir, "basler.h5"), 'r') as f:
                num_frames = len(f['face_g_tobii']["data"])
            # txt_lines.append((video_dir, num_frames - 1))  # Use last frame
                step_size = min(30, (num_frames - 1)//20)
                for i in range(90, num_frames-1, 30):
                    lbl = f['face_g_tobii']["data"][i]
                    if not lbl.any():
                        continue
                    txt_lines.append((video_dir, i))
            
    csv_path = f"saved/data/eve/{split}.csv"
    if not os.path.exists(os.path.dirname(csv_path)):
        os.makedirs(os.path.dirname(csv_path))
    df = pd.DataFrame(txt_lines, columns=['video_path', 'frame_idx'])
    df.to_csv(csv_path, header=None, index=False, sep =' ')
            