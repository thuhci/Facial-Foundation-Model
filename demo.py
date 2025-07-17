import torch
import numpy as np
import cv2
from PIL import Image
from timm.models import create_model
from src.utils import load_state_dict
import torchvision  as transforms
from src.models import modeling_finetune
from src.dataset.kinetics import VideoClsDataset, VideoMAE, VideoClsDatasetFrame
import video_transforms
import matplotlib.pyplot as plt
from src.dataset.augment import volume_transforms
# from pathlib import Path
from facenet_pytorch import MTCNN
from datetime import datetime
from typing import List
import numpy as np
import math

def recognize_faces(frame: np.ndarray, device: str) -> List[np.array]:
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
        mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device='cpu')
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs[0] is None:
            return []
        bounding_boxes = bounding_boxes[probs > 0.9]
        return bounding_boxes

    bounding_boxes = detect_face(frame)
    facial_images = []
    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]
        facial_images.append(frame[y1:y2, x1:x2, :])
        
    return facial_images,bounding_boxes

LABEL_DFER = ["Happy","Sad","Neutral","Angry","Surprise","Disgust","Fear"]
LABEL_MAFW = ["Anger","Disgust","Fear","Happiness","Sadness","Surprise","Contempt","Anxiety","Helplessness","Disappointment","Neutral"]

LABEL = LABEL_DFER

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load("checkpoint/checkpoint-99-dfew.pth",  map_location='cpu')
print(checkpoint.keys())  

model = create_model(
            "vit_base_dim512_no_depth_patch16_160",
            pretrained=False,
            num_classes=len(LABEL),
            all_frames=16 * 1,
            tubelet_size=2,
            drop_rate=0.0,
            drop_path_rate=0.1,
            attn_drop_rate=0.0,
            drop_block_rate=None,
            use_mean_pooling=True, # change from AI's instructions
            init_scale=0.001,
            depth=16,
            attn_type="local_global",
            lg_region_size=[2,5,10], lg_first_attn_type="self",
            lg_third_attn_type="cross",
            lg_attn_param_sharing_first_third=False,
            lg_attn_param_sharing_all=False,
            lg_classify_token_type="region",
            lg_no_second=False, lg_no_third=False,
        )


state_dict = checkpoint["model"]
load_state_dict(model=model, state_dict=state_dict)
model.eval()

print("Model loaded successfully.")

# cap = cv2.VideoCapture(0)  
# read from  a video file
cap = cv2.VideoCapture('gaze360/code/test_video_3.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  160)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  160)

all_frames = []
with torch.no_grad(): 
    while True:
        ret, frame = cap.read() 
        # frame = np.zeros((160, 160, 3), dtype=np.uint8)  # Create a dummy frame for testing
        # print("shape of frame:", frame.shape)
        # ret = True  # Simulate successful frame capture
        if not ret:
            continue
        face, bbox = recognize_faces(frame, device='cpu')
        if face:
            print(f"Detected {len(face)} faces")
            if len(all_frames)>=16:
                all_frames.pop(0)
            all_frames.append(face[0])  
            for j,(x1, y1, x2, y2) in enumerate(bbox):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255, 0, 0), 2)
                
                # cv2.putText(frame, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        else:
            print("No face detected")
        if len(all_frames)==16:
            buffer = all_frames
            short_side_size = 160
            resize = video_transforms.Compose([video_transforms.Resize(size=(short_side_size, short_side_size), interpolation='bilinear')])
            data_transform = video_transforms.Compose([
                volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
            ])
            buffer = resize(buffer) # resize to 160*160
            if isinstance(buffer, list): # if buffer is a list, convert to numpy array
                buffer = np.stack(buffer, 0)
    
            buffer = data_transform(buffer) # convert to tensor and normalize
            print(buffer.shape)

            buffer = np.expand_dims(buffer, axis=0) # add batch dimension
            buffer = torch.from_numpy(buffer)
            output= model(buffer, save_feature=False)
            emotion = LABEL[torch.argmax(output)]
            cv2.putText(frame,  emotion, (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # for i, sampled_frame in enumerate(all_frames):
            #     cv2.putText(sampled_frame, emotion, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            #     cv2.imshow(f'Sampled Frame {i}', sampled_frame)
        
        # cv2.imshow('Emotion  Detection', frame)
        cv2.imwrite('output/frame.jpg', frame)  # Save the frame with detected faces
        if cv2.waitKey(1)  & 0xFF == ord('q'):  
            break 
 
cap.release() 
cv2.destroyAllWindows() 

fig, axes = plt.subplots(4, 4, figsize=(20, 20))  
for ax, frame in zip(axes.flatten(), all_frames):
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  
    ax.axis('off') 
print("Emotion Detected:", emotion)

fig.suptitle("Detected Emotion: " + emotion, fontsize=36, fontweight='bold', y=0.95)
emotion_dict = {key: np.round(value.numpy(), 2) for key, value in zip(LABEL, output[0])}
print("Emotion Status:", emotion_dict)


emotion_status = "\n".join([f"{key}: {str(np.round(value, 2))}" for key, value in emotion_dict.items()])  # 分行显示
print("Emotion Status:", emotion_status)
fig.text(0.99, 0.99, "Emotion Status:\n" + emotion_status,
         fontsize=20, ha='right', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# 使用 fig.text 添加文本
# fig.text(0.5, 0.85, emotion_text, fontsize=24, ha='center', va='top', wrap=True)  # wrap=True 可以换行
fig.subplots_adjust(wspace=0.1, hspace=0.1)  # 设置宽度和高度的间距
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
fig.savefig(f'output/image_{current_time}.png')
plt.close(fig)
