import os
import argparse
import pandas as pd
import numpy as np

#################################################################
#      modify this output_root                                  #
#################################################################

output_root = '/home/qzk/Facial-Foundation-Model/saved/data/gaze360T_combine'


def GazeTo2d(gaze):
    yaw = np.arctan2(gaze[0], -gaze[2])
    pitch = np.arcsin(gaze[1])
    return np.array([pitch, yaw])

def GazeTo3d(gaze):
    x = np.cos(gaze[0]) * np.sin(gaze[1])
    y = np.sin(gaze[0])
    z = -np.cos(gaze[0]) * np.cos(gaze[1])
    return np.array([x, y, z])
    
def preprocess(gap_threshold=5, len_threshold=16, save_discarded=True):
    # remove all files in the output directory
    if os.path.exists(output_root):
        pass
        for root, dirs, files in os.walk(output_root, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
    else:
        os.makedirs(output_root)
    for split in ["train", "test", "val"]:
        print("processing split", split)
    # for split in ["test"]:
    
        #################################################################
        #      modify this txt_path to your original csv / txt          #
        #################################################################
        txt_path = f"/home/qzk/Facial-Foundation-Model/saved/data/gaze360/{split}.csv"
        out_path = f"{output_root}/{split}"
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        txt_lines = []
        # read from txt file
        with open(txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                # print(line)
                # delete the '.jpg' suffix
                # line = line.replace('.jpg', '')
                # add "data_root" to the beginning
                
                #################################################################
                #      modify here to get proper rec_name & frame_name          #
                #################################################################
                
                rec_name = int(line.split('/')[5].split("_")[-1])
                per_name =  int(line.split('/')[7])
                frame_name = int(line.split('/')[8].split('.')[0])
                # frame_name = line.split('/')[9].split('.')[0]
                lbl = line.split('/')[8].split(' ')[1:]
                # print("rec_name", rec_name, "per_name", per_name, "frame_name", frame_name, "lbl", lbl)
                lbl = [float(x) for x in lbl]
                lbl = GazeTo2d(lbl)

                line = line.split(' ')[0] + " 0 " + str(lbl[0]) + ' ' + str(lbl[1])

                # print("rec_name", rec_name, "per_name", per_name, "frame_name", frame_name, "lbl", lbl)
                
                # print("line: ", line)
                
                txt_lines.append((line, rec_name, per_name, frame_name, lbl))

        # sort by rec_name, vid_name, frame_name
        txt_lines.sort(key=lambda x: (x[1], x[2], x[3]))
        lst = 0
        files = []

        curr_file = [txt_lines[0]]
        for i in range(1, len(txt_lines)):
            txt_line = txt_lines[i]
            lst_line = txt_lines[i-1]
            # if txt_line[1] == lst_line[1] and txt_line[2] <= lst_line[2] + gap_threshold:
            if txt_line[1] == lst_line[1] and txt_line[2] == lst_line[2] and txt_line[3] <= lst_line[3] + gap_threshold:
                # curr_file.append(txt_line)
                for j in range(int(txt_line[3]) - int(lst_line[3])):
                    curr_file.append(txt_line)
                continue
            lst = i
            files.append(curr_file)
            curr_file = [txt_line]
            
        files.append(curr_file)
        

        filtered_files = [file for file in files if len(file) >= len_threshold]
        discarded_files = [file for file in files if len(file) < len_threshold]
        files = filtered_files
        
        sum_lines = []
        
        for i, file in enumerate(files):
            out_lines = [f"{line[0]}" for line in file]
            csv_path = f"{output_root}/{split}/{split}_{file[0][1]}_{i}.csv"
            if os.path.exists(csv_path):
                os.remove(csv_path)
            df = pd.DataFrame(out_lines, columns=['file_path'])
            df.to_csv(csv_path, header=None, index=False)
            
        if not save_discarded:
            continue
            
        for i, file in enumerate(discarded_files):
            out_lines = [line[0] for line in file]
            csv_path = f"{output_root}/{split}_discarded/{split}_{file[0][1]}_{i}.csv"
            if not os.path.exists(f"{output_root}/{split}_discarded"):
                os.makedirs(f"{output_root}/{split}_discarded")
            if os.path.exists(csv_path):
                os.remove(csv_path)
            df = pd.DataFrame(out_lines, columns=['file_path'])
            df.to_csv(csv_path, header=None, index=False)
        
        
    
if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('-g', type=int, default=5, help='The gap threshold for grouping frames in a video')
    args.add_argument('-l', type=int, default=16, help='The minimum length of a video to be considered valid')
    args.add_argument('-d', action='store_true', help='Whether to save discarded videos')
    args = args.parse_args()

    preprocess(gap_threshold=args.g, len_threshold=args.l, save_discarded=args.d)


    # out_lines = [f"{line[0]}" for line in txt_lines]
    # # write to csv file
    # csv_path = f"saved/data/gaze360/{split}.csv"
    # df = pd.DataFrame(out_lines, columns=['file_path'])
    # # no need the headline
    # # df = pd.DataFrame(txt_lines)
    # df.to_csv(csv_path, header=None, index=False)
