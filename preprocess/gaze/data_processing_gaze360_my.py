import numpy as np
import scipy.io as sio
import cv2 
import os
import sys
# sys.path.append("../core/")
# import data_processing_core as dpc
from facenet_pytorch import MTCNN
# import List

import time



root = "../Gaze360/imgs"
out_root = "../Gaze360/normalized_imgs"
txt_root = "../Gaze360/gaze360/code"
out_txt_root = "../Gaze360/lbls"

detector = MTCNN(keep_all=False, post_process=False, min_face_size=80, device='cpu')


def recognize_faces(frame: np.ndarray, device: str):
    """
    Detects faces in the given image and returns the facial images cropped from the original.

    This function reads an image from the specified path, detects faces using the MTCNN
    face detection model, and returns a list of cropped face images.

    Args:
        frame (numpy.ndarray): The image frame in which faces need to be detected.
        device (str): The device to run the MTCNN face detection model on, e.g., 'cpu' or 'cuda'.

    Returns:
        list: A list of numpy arrays, representing a cropped face image from the original image.

    Example:
        faces = recognize_faces('image.jpg', 'cuda')
        # faces contains the cropped face images detected in 'image.jpg'.
    """

    def detect_face(frame: np.ndarray):
        # print("size of frame:", frame.shape)
        bounding_boxes, probs = detector.detect(frame, landmarks=False)
        if bounding_boxes is None or len(bounding_boxes) == 0:
            return []
        if probs[0] is None:
            return []
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    # facial_images = []
    # for bbox in bounding_boxes:
    #     box = bbox.astype(int)
    #     x1, y1, x2, y2 = box[0:4]
    #     facial_images.append(frame[y1:y2, x1:x2, :])
        
    return bounding_boxes

def ImageProcessing_Gaze360(split):
    # msg = sio.loadmat(os.path.join(root, "metadata.mat"))
    
    # recordings = msg["recordings"]
    # gazes = msg["gaze_dir"]
    # head_bbox = msg["person_head_bbox"]
    # face_bbox = msg["person_face_bbox"]
    # lefteye_bbox = msg["person_eye_left_bbox"]
    # righteye_bbox = msg["person_eye_right_bbox"]
    # splits = msg["splits"]

    # split_index = msg["split"]
    # recording_index = msg["recording"]
    # person_index = msg["person_identity"]
    # frame_index = msg["frame"]
  
    # total_num = recording_index.shape[1]
    
    start_time = time.time()
    outfiles = []
    
    txt_file = os.path.join(txt_root, f"{split}.txt")
    lines = open(txt_file, 'r').readlines()
    num_lines = len(lines)
    print(f"Processing {num_lines} images in {split} split...")
    for i, line in enumerate(lines):
        # print(line.strip())
        if i % 100 == 0:
            time_elapsed = time.time() - start_time
            # print(f"Time elapsed: {time_elapsed:.2f} seconds", end="\r")
            time_to_go = (num_lines - i) * (time_elapsed / (i + 1))
            # print(f"Estimated time to go: {time_to_go:.2f} seconds", end="\r")
            progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(i/num_lines * 20))
            progressbar = "\r" + progressbar + f" {i}|{num_lines}"
            print(progressbar, end="", flush=True)
            
        
        file_name = line.strip().split()[0]
        lbls = line.strip().split()[1:]
        img = cv2.imread(os.path.join(root, file_name))
        if img is None:
            print(f"Image {file_name} not found, skipping...")
            continue
        bboxs = recognize_faces(img, device='cpu')
        if len(bboxs) > 0:
            bbox = bboxs[0].astype(int)
        else:
            continue
        
        # pick ful img as head bbox
        # # [DEBUG] xyxy or xywh
        head_bbox = np.array([0, 0, img.shape[1], img.shape[0]])
        bbox = np.array([bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])  # xywh format
        cropped_face = CropFaceImg(img, head_bbox, bbox)
        # cropped_face = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # cropped_face = cv2.resize(cropped_face, (224, 224))
        
        output_name = os.path.join(out_root, file_name)
        os.makedirs(os.path.dirname(output_name), exist_ok=True)
        cv2.imwrite(output_name, cropped_face)
        outfiles.append((file_name, lbls))
        
    out_txt_file = os.path.join(out_txt_root, f"{split}.txt")
    os.makedirs(os.path.dirname(out_txt_file), exist_ok=True)
    with open(out_txt_file, 'w') as f:
        for output_name, lbls in outfiles:
            lbl_str = " ".join(lbls)
            f.write(f"{output_name} {lbl_str}\n")
        

    # process each image
    # for i in range(total_num):
    #     im_path = os.path.join(root, "imgs",
    #         recordings[0, recording_index[0, i]][0],
    #         "head", '%06d' % person_index[0, i],
    #         '%06d.jpg' % frame_index[0, i]
    #         )

    #    	progressbar = "".join(["\033[41m%s\033[0m" % '   '] * int(i/total_num * 20))
    #     progressbar = "\r" + progressbar + f" {i}|{total_num}"
    #     print(progressbar, end = "", flush=True)
    #     if (face_bbox[i] == np.array([-1, -1, -1, -1])).all():
    #         continue

    #     category = splits[0, split_index[0, i]][0]
    #     gaze = gazes[i]

    #     img = cv2.imread(im_path)
    #     face = CropFaceImg(img, head_bbox[i], face_bbox[i])
    #     lefteye = CropEyeImg(img, head_bbox[i], lefteye_bbox[i])
    #     righteye = CropEyeImg(img, head_bbox[i], righteye_bbox[i]) 
        
    #     cv2.imwrite(os.path.join(out_root, "Image", category, "Face", f"{i+1}.jpg"), face)
    #     cv2.imwrite(os.path.join(out_root, "Image", category, "Left", f"{i+1}.jpg"), lefteye)
    #     cv2.imwrite(os.path.join(out_root, "Image", category, "Right", f"{i+1}.jpg"), righteye)

    #     gaze2d = GazeTo2d(gaze) 

    #     save_name_face = os.path.join(category, "Face", f"{i+1}.jpg")
    #     save_name_left = os.path.join(category, "Left", f"{i+1}.jpg")
    #     save_name_right = os.path.join(category, "Right", f"{i+1}.jpg")

    #     save_origin = os.path.join(recordings[0, recording_index[0, i]][0],
    #         "head", "%06d" % person_index[0, i], "%06d.jpg"% frame_index[0, i])

    #     save_gaze = ",".join(gaze.astype("str"))
    #     save_gaze2d = ",".join(gaze2d.astype("str"))

    #     save_str = " ".join([save_name_face, save_name_left, save_name_right, save_origin, save_gaze, save_gaze2d])
    #     outfiles[split_index[0, i]].write(save_str + "\n")

    # for i in outfiles:
    #     i.close()
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
    for split in ["val", "train", "test"]:
        print(f"Processing {split} split...")
        ImageProcessing_Gaze360(split)
