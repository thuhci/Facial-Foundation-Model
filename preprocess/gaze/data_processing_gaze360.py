import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
sys.path.append("../core/")
# import data_processing_core as dpc

root = "../Gaze360/"
out_root = "../Gaze360/phi_normalize"

def ImageProcessing_Gaze360():
    msg = sio.loadmat(os.path.join(root, "metadata.mat"))
    
    recordings = msg["recordings"]
    gazes = msg["gaze_dir"]
    head_bbox = msg["person_head_bbox"]
    face_bbox = msg["person_face_bbox"]
    lefteye_bbox = msg["person_eye_left_bbox"]
    righteye_bbox = msg["person_eye_right_bbox"]
    splits = msg["splits"]

    split_index = msg["split"]
    recording_index = msg["recording"]
    person_index = msg["person_identity"]
    frame_index = msg["frame"]
  
    total_num = recording_index.shape[1]
    outfiles = []
    
    out_lines = []
    
    
    if not os.path.exists(out_root):
        os.makedirs(out_root)
    

    # Build folders for saving image and label.
    if not os.path.exists(os.path.join(out_root, "Label")):
        os.makedirs(os.path.join(out_root, "Label"))
        
    for i in range(4):
        if not os.path.exists(os.path.join(out_root, "Image", splits[0, i][0])):
            # os.makedirs(os.path.join(out_root, "Image", splits[0, i][0], "Left"))
            # os.makedirs(os.path.join(out_root, "Image", splits[0, i][0], "Right"))
            os.makedirs(os.path.join(out_root, "Image", splits[0, i][0], "Face"))

        outfiles.append(open(os.path.join(out_root, "Label", f"{splits[0, i][0]}.label"), 'w')) 
        outfiles[i].write("Face 2DGaze\n")

    # process each image
    for i in range(total_num):
        im_path = os.path.join(root, "imgs",
            recordings[0, recording_index[0, i]][0],
            "head", '%06d' % person_index[0, i],
            '%06d.jpg' % frame_index[0, i]
            )

        progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(i/total_num * 20))
        progressbar = "\r" + progressbar + f" {i}|{total_num}"
        print(progressbar, end = "", flush=True)
        if (face_bbox[i] == np.array([-1, -1, -1, -1])).all():
            continue

        category = splits[0, split_index[0, i]][0]
        gaze = gazes[i]

        img = cv2.imread(im_path)
        face = CropFaceImg(img, head_bbox[i], face_bbox[i])
        # lefteye = CropEyeImg(img, head_bbox[i], lefteye_bbox[i])
        # righteye = CropEyeImg(img, head_bbox[i], righteye_bbox[i]) 

        out_img_path = os.path.join(out_root, "Image", category, "Face", recordings[0, recording_index[0, i]][0],
            "head", '%06d' % person_index[0, i],
            '%06d.jpg' % frame_index[0, i])
        
        if not os.path.exists(os.path.dirname(out_img_path)):
            os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

        cv2.imwrite(out_img_path, face)
        # cv2.imwrite(os.path.join(out_root, "Image", category, "Left", f"{i+1}.jpg"), lefteye)
        # cv2.imwrite(os.path.join(out_root, "Image", category, "Right", f"{i+1}.jpg"), righteye)

        gaze2d = GazeTo2d(gaze) 

        # save_name_face = os.path.join(category, "Face", f"{i+1}.jpg")
        # save_name_left = os.path.join(category, "Left", f"{i+1}.jpg")
        # save_name_right = os.path.join(category, "Right", f"{i+1}.jpg")

        # save_origin = os.path.join(recordings[0, recording_index[0, i]][0],
        #     "head", "%06d" % person_index[0, i], "%06d.jpg"% frame_index[0, i])

        # save_gaze = ",".join(gaze.astype("str"))
        save_gaze2d = ",".join(gaze2d.astype("str"))

        save_str = " ".join([out_img_path, save_gaze2d])
        outfiles[split_index[0, i]].write(save_str + "\n")
        
        out_lines.append((out_img_path, save_gaze2d))

    for i in outfiles:
        i.close()
        
    
        # for split in ["train", "val", ]
        #     out_txt_file = os.path.join(out_root, f"{split}.txt")
        #     os.makedirs(os.path.dirname(out_txt_file), exist_ok=True)
        #     with open(out_txt_file, 'w') as f:
        #         for output_name, lbls in outfiles:
        #             lbl_str = " ".join(lbls)
        #         f.write(f"{output_name} {lbl_str}\n")
    #

def GazeTo2d(gaze):
  yaw = np.arctan2(gaze[0], -gaze[2])
  pitch = np.arcsin(gaze[1])
  return np.array([yaw, pitch])

def CropFaceImg(img, head_bbox, cropped_bbox):
    bbox =np.array([ (cropped_bbox[0] - head_bbox[0])/head_bbox[2],
              (cropped_bbox[1] - head_bbox[1])/head_bbox[3],
              cropped_bbox[2] / head_bbox[2],
              cropped_bbox[3] / head_bbox[3]])

    size = np.array([img.shape[1], img.shape[0]])

    bbox_pixel = np.concatenate([bbox[:2] * size, bbox[2:] * size]).astype("int")

    # Find the image center and crop head images with length = max(weight, height)
    center = np.array([bbox_pixel[0]+bbox_pixel[2]//2, bbox_pixel[1]+bbox_pixel[3]//2])

    length = int(max(bbox_pixel[2], bbox_pixel[3])/2) 

    center[0] = max(center[0], length)
    center[1] = max(center[1], length)

    result = img[(center[1] - length) : (center[1] + length),
                (center[0] - length) : (center[0] + length)] 

    result = cv2.resize(result, (224, 224))
    return result

def CropEyeImg(img, head_bbox, cropped_bbox):
    bbox =np.array([ (cropped_bbox[0] - head_bbox[0])/head_bbox[2],
              (cropped_bbox[1] - head_bbox[1])/head_bbox[3],
              cropped_bbox[2] / head_bbox[2],
              cropped_bbox[3] / head_bbox[3]])

    size = np.array([img.shape[1], img.shape[0]])

    bbox_pixel = np.concatenate([bbox[:2] * size, bbox[2:] * size]).astype("int")

    center = np.array([bbox_pixel[0]+bbox_pixel[2]//2, bbox_pixel[1]+bbox_pixel[3]//2])
    height = bbox_pixel[3]/36
    weight = bbox_pixel[2]/60
    ratio = max(height, weight) 

    size = np.array([ratio*30, ratio*18]).astype("int")

    center[0] = max(center[0], size[0])
    center[1] = max(center[1], size[1])


    result = img[(center[1] - size[1]): (center[1] + size[1]),
                (center[0] - size[0]): (center[0] + size[0])]

    result = cv2.resize(result, (60, 36)) 
    return result

if __name__ == "__main__":
    ImageProcessing_Gaze360()
