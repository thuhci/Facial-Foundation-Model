import os
import pandas as pd

data_root = '../GazeCapture/Gaze360'

for split in ["train", "val", "test"]:
# split = "test"
    txt_path = f"../saved/data/gaze360/{split}.txt"

    txt_lines = []
    # read from txt file
    with open(txt_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            # delete the '.jpg' suffix
            # line = line.replace('.jpg', '')
            # add "data_root" to the beginning
            rec_name = line.split('/')[0]
            vid_name = line.split('/')[2]
            frame_name = line.split('/')[3]
            
            line = os.path.join(data_root, line)
            
            
            
            txt_lines.append((line, rec_name, vid_name, frame_name))

    # sort by rec_name, vid_name, frame_name
    txt_lines.sort(key=lambda x: (x[1], x[2], x[3]))
    out_lines = [f"{line[0]}" for line in txt_lines]
    # write to csv file
    csv_path = f"../saved/data/gaze360/{split}.csv"
    df = pd.DataFrame(out_lines, columns=['file_path'])
    # no need the headline
    # df = pd.DataFrame(txt_lines)
    df.to_csv(csv_path, header=None, index=False)
