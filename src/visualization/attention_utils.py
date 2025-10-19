"""
注意力可视化的使用示例和工具函数
"""

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List
import os

def load_video_for_visualization(video_path: str, 
                                target_size: Tuple[int, int] = (160, 160),
                                num_frames: int = 16) -> torch.Tensor:
    """
    加载视频用于可视化
    
    Args:
        video_path: 视频文件路径
        target_size: 目标尺寸 (H, W)
        num_frames: 帧数
    
    Returns:
        视频张量 (1, 3, T, H, W)
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            # BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            frame = cv2.resize(frame, target_size)
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
    
    cap.release()
    
    # Convert to tensor (T, H, W, C) -> (1, C, T, H, W)
    video_array = np.stack(frames)  # (T, H, W, C)
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
    
    return video_tensor

def load_image_sequence_for_visualization(image_dir: str,
                                        target_size: Tuple[int, int] = (160, 160),
                                        num_frames: int = 16) -> torch.Tensor:
    """
    从图像序列加载视频用于可视化
    
    Args:
        image_dir: 图像目录路径
        target_size: 目标尺寸
        num_frames: 帧数
        
    Returns:
        视频张量 (1, 3, T, H, W)
    """
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    if len(image_files) < num_frames:
        # 如果图像不够，重复最后一张
        image_files.extend([image_files[-1]] * (num_frames - len(image_files)))
    elif len(image_files) > num_frames:
        # 如果图像太多，均匀采样
        indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
        image_files = [image_files[i] for i in indices]
    
    frames = []
    for img_file in image_files[:num_frames]:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        frames.append(img)
    
    video_array = np.stack(frames)
    video_tensor = torch.from_numpy(video_array).permute(3, 0, 1, 2).unsqueeze(0)
    
    return video_tensor

def visualize_gaze_prediction(attention_center: Tuple[float, float],
                            predicted_gaze: Tuple[float, float],
                            image_size: Tuple[int, int] = (160, 160)) -> np.ndarray:
    """
    可视化注意力中心和预测的注视方向
    
    Args:
        attention_center: 注意力中心 (x, y) in [-1, 1]
        predicted_gaze: 预测的注视角度 (yaw, pitch) in radians
        image_size: 图像尺寸
        
    Returns:
        可视化图像
    """
    h, w = image_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 绘制注意力中心
    center_x = int((attention_center[0] + 1) * w / 2)
    center_y = int((attention_center[1] + 1) * h / 2)
    cv2.circle(img, (center_x, center_y), 5, (255, 0, 0), -1)  # 红色圆点
    
    # 绘制注视方向向量
    yaw, pitch = predicted_gaze
    
    # 将角度转换为图像坐标系中的方向向量
    # 注意：这里需要根据你的坐标系定义来调整
    vec_x = np.sin(yaw) * 30  # 缩放因子
    vec_y = -np.sin(pitch) * 30  # 负号因为图像坐标系y轴向下
    
    end_x = int(center_x + vec_x)
    end_y = int(center_y + vec_y)
    
    # 确保终点在图像范围内
    end_x = np.clip(end_x, 0, w - 1)
    end_y = np.clip(end_y, 0, h - 1)
    
    # 绘制箭头
    cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), (0, 255, 0), 2)  # 绿色箭头
    
    return img

def create_attention_comparison_plot(attention_maps: List[np.ndarray],
                                   layer_names: List[str],
                                   original_frame: np.ndarray,
                                   save_path: str = None) -> plt.Figure:
    """
    创建多层注意力对比图
    
    Args:
        attention_maps: 注意力图列表
        layer_names: 层名称列表
        original_frame: 原始图像
        save_path: 保存路径
        
    Returns:
        matplotlib图像对象
    """
    num_layers = len(attention_maps)
    fig, axes = plt.subplots(2, num_layers + 1, figsize=(4 * (num_layers + 1), 8))
    
    if num_layers == 1:
        axes = axes.reshape(2, -1)
    
    # 原始图像
    axes[0, 0].imshow(original_frame)
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(original_frame)
    axes[1, 0].set_title('Original Frame')
    axes[1, 0].axis('off')
    
    # 各层注意力图
    for i, (attn_map, layer_name) in enumerate(zip(attention_maps, layer_names)):
        # 纯注意力图
        im = axes[0, i + 1].imshow(attn_map, cmap='hot', interpolation='bilinear')
        axes[0, i + 1].set_title(f'{layer_name}\nAttention Map')
        axes[0, i + 1].axis('off')
        plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046, pad=0.04)
        
        # 叠加图
        axes[1, i + 1].imshow(original_frame)
        axes[1, i + 1].imshow(attn_map, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[1, i + 1].set_title(f'{layer_name}\nOverlay')
        axes[1, i + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

# 使用示例函数
def example_usage():
    """注意力可视化的完整使用示例"""
    
    # 1. 假设你已经有了训练好的模型
    # model = YourViTModel.load_from_checkpoint('path/to/checkpoint')
    # model.eval()
    
    # 2. 创建可视化器
    # visualizer = AttentionVisualizer(model, save_dir='./visualizations')
    
    # 3. 准备输入数据
    # 方式1: 从视频文件加载
    # video_tensor = load_video_for_visualization('path/to/video.mp4')
    
    # 方式2: 从图像序列加载
    # video_tensor = load_image_sequence_for_visualization('path/to/image/dir')
    
    # 方式3: 使用现有的预处理数据
    # video_tensor = your_preprocessed_data.unsqueeze(0)  # 添加batch维度
    
    # 4. 进行预测并获取注意力
    # with torch.no_grad():
    #     prediction = model(video_tensor)
    
    # 5. 生成综合报告
    # report_dir = visualizer.generate_comprehensive_report(
    #     video_tensor, prediction, sample_name='example_sample'
    # )
    
    # 6. 单独的可视化
    # spatial_attention = visualizer.visualize_spatial_attention(video_tensor, layer_idx=-1)
    # temporal_attention = visualizer.visualize_temporal_attention(video_tensor, layer_idx=-1)
    
    # 7. 清理
    # visualizer.remove_hooks()
    
    print("Example usage completed! Check the generated visualizations.")

if __name__ == "__main__":
    # 运行示例（需要实际的模型和数据）
    print("This is the attention visualization toolkit for Video ViT gaze estimation.")
    print("Please see the example_usage() function for how to use it.")
    print("Key features:")
    print("1. Spatial attention visualization - shows which image regions are important")
    print("2. Temporal attention visualization - shows which frames are important") 
    print("3. Attention-prediction correlation analysis")
    print("4. Comprehensive visualization reports")
