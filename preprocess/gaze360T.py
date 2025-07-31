import os
import pandas as pd

data_root = '/root/shared/Gaze360/normalized_imgs'

gap_threshold = 5
len_threshold = 100


for split in ["train", "val", "test"]:
# for split in ["test"]:
    txt_path = f"/root/shared/Gaze360/lbls/{split}.txt"
    out_path = f"saved/data/gaze360T/{split}"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

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
            frame_name = line.split('/')[3].split('.')[0]
            lbl = line.split('/')[3]

            # print("rec_name", rec_name, "vid_name", vid_name, "frame_name", frame_name)

            line = os.path.join(data_root, line)
            
            
            
            txt_lines.append((line, rec_name, vid_name, frame_name, lbl))

    # sort by rec_name, vid_name, frame_name
    txt_lines.sort(key=lambda x: (x[1], x[2], x[3]))
    lst = 0
    files = []

    curr_file = [txt_lines[0]]
    for i in range(1, len(txt_lines)):
        txt_line = txt_lines[i]
        lst_line = txt_lines[i-1]
        if txt_line[1] == lst_line[1] and txt_line[2] == lst_line[2] and int(txt_line[3]) <= int(lst_line[3])+gap_threshold:
            # curr_file.append(txt_line)
            for j in range(int(txt_line[3]) - int(lst_line[3])):
                curr_file.append(txt_line)
            continue
        lst = i
        files.append(curr_file)
        curr_file = [txt_line]
        
    files.append(curr_file)
    
    files = [file for file in files if len(file) >= len_threshold]
    
    sum_lines = []
    
    for i, file in enumerate(files):
        out_lines = [f"{line[0]}" for line in file]
        # if len(file) < len_threshold:
        #     continue
        # write to csv file
        csv_path = f"saved/data/gaze360T/{split}/{split}_{file[0][1]}_{file[0][2]}_{i}.csv"
        df = pd.DataFrame(out_lines, columns=['file_path'])
        # no need the headline
        # df = pd.DataFrame(txt_lines)
        df.to_csv(csv_path, header=None, index=False)
        
        
    
    
    
    # out_lines = [f"{line[0]}" for line in txt_lines]
    # # write to csv file
    # csv_path = f"saved/data/gaze360/{split}.csv"
    # df = pd.DataFrame(out_lines, columns=['file_path'])
    # # no need the headline
    # # df = pd.DataFrame(txt_lines)
    # df.to_csv(csv_path, header=None, index=False)
