import os
import sys
import shutil
import re
sys.path.append("../core/")
import data_processing_core as dpc


root = "/home/cyh/GazeDataset20200519/Original/Rt-Gene"
out_root = "/home/cyh/GazeDataset20200519/FaceBased/RT-Gene"

valid = ["s014", "s015", "s016"]

train_fold1 = ["s001", "s002", "s008", "s010"]
train_fold2 = ["s003", "s004", "s007", "s009"]
train_fold3 = ["s005", "s006", "s011", "s012", "s013"]

def ImageProcessing_RT():
    # Acquire the list of files
    file_names = os.listdir(root)
    file_names = [name for name in file_names if "_glasses" in name]
    total = len(file_names)

    if not os.path.exists(os.path.join(out_root, "Label/test")):
        os.makedirs(os.path.join(out_root, "Label/test"))

    # Prepare outfiles' path of label
    label_outpath1 = os.path.join(out_root, "Label/test", "train1.label")
    label_outpath2 = os.path.join(out_root, "Label/test", "train2.label")
    label_outpath3 = os.path.join(out_root, "Label/test", "train3.label")
    label_outpath4 = os.path.join(out_root, "Label/test", "valid.label")

    # Prepare the head of label files
    # note that the 3DHead is the vector of z axis while the rotated vector.
    with open(label_outpath1, 'w') as outfile:
        outfile.write("Face Left Right Origin 3DGaze 3DHead 2DGaze 2DHead\n")
    with open(label_outpath2, 'w') as outfile:
        outfile.write("Face Left Right Origin 3DGaze 3DHead 2DGaze 2DHead\n")
    with open(label_outpath3, 'w') as outfile:
        outfile.write("Face Left Right Origin 3DGaze 3DHead 2DGaze 2DHead\n")
    with open(label_outpath4, 'w') as outfile:
        outfile.write("Face Left Right Origin 3DGaze 3DHead 2DGaze 2DHead\n")

    # Process each person's data
    for count, person_path in enumerate(file_names):
        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(count/total * 20))
        progressbar = "\r" + progressbar + f" {count}|{total}"
        print(progressbar, end="", flush=True)

        person_name = person_path[0:4]

        if person_name in train_fold1: 
            ImageProcessing_Person(root, person_path, label_outpath1)
        elif person_name in train_fold2:
            ImageProcessing_Person(root, person_path, label_outpath2)
        elif person_name in train_fold3:
            ImageProcessing_Person(root, person_path, label_outpath3)
        elif person_name in valid:
            ImageProcessing_Person(root, person_path, label_outpath4)
        else:
            pass
         

def ImageProcessing_Person(im_root, person_path, label_outpath):

    outfile = open(label_outpath, 'a')

    label_path = os.path.join(im_root, person_path, "label_combined.txt")

    imgs_face = os.listdir(os.path.join(im_root, person_path, "original", "face"))
    imgs_left = os.listdir(os.path.join(im_root, person_path, "original", "left"))
    imgs_right = os.listdir(os.path.join(im_root, person_path, "original", "right"))

    imgs_face.sort(key = lambda x:int(x.split("_")[1]))
    imgs_left.sort(key = lambda x:int(x.split("_")[1]))
    imgs_right.sort(key = lambda x:int(x.split("_")[1]))

    pimgs_face = os.listdir(os.path.join(im_root, person_path, "inpainted", "face"))
    pimgs_left = os.listdir(os.path.join(im_root, person_path, "inpainted", "left"))
    pimgs_right = os.listdir(os.path.join(im_root, person_path, "inpainted", "right"))

    pimgs_face.sort(key = lambda x:int(x.split("_")[1]))
    pimgs_left.sort(key = lambda x:int(x.split("_")[1]))
    pimgs_right.sort(key = lambda x:int(x.split("_")[1]))


    with open(label_path) as infile:
        label_info = infile.readlines()

    for num, label in enumerate(label_info):
        label = re.split('\[| |,|\]', label.strip())

        save_name_face = os.path.join(f"{person_path}/original/face/{imgs_face[num]}")
        save_name_left = os.path.join(f"{person_path}/original/left/{imgs_left[num]}")
        save_name_right = os.path.join(f"{person_path}/original/right/{imgs_right[num]}")

        save_person_name = person_path[0:4]

        save_head = dpc.GazeTo3d(list(map(eval, [label[3], label[5]])))
        save_gaze = dpc.GazeTo3d(list(map(eval, [label[9], label[11]])))

        save_head = ",".join(save_head.astype("str"))
        save_gaze = ",".join(save_gaze.astype("str"))
        save_head2d = f"{label[3]},{label[5]}"
        save_gaze2d = f"{label[9]},{label[11]}"

        save_str = " ".join([save_name_face, save_name_left, save_name_right, save_person_name, save_gaze, save_head, save_gaze2d, save_head2d])

        outfile.write(save_str + "\n")

if __name__ == "__main__":
    ImageProcessing_RT()
