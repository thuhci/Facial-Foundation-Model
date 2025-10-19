"""
注意力可视化演示脚本
展示如何使用AttentionVisualizer进行眼部注视角度模型的注意力分析
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# 添加src路径以便导入模块
sys.path.append('/home/qzk/Facial-Foundation-Model/src')

from visualization.attention_visualizer import AttentionVisualizer
from visualization.attention_utils import (
    load_video_for_visualization, 
    load_image_sequence_for_visualization,
    create_attention_comparison_plot
)

def demo_attention_visualization():
    """
    演示注意力可视化的完整流程
    """
    print("=" * 60)
    print("眼部注视角度模型 - 注意力可视化演示")
    print("=" * 60)
    
    # 创建一个模拟的ViT模型用于演示
    print("\n1. 创建模拟模型（实际使用时请替换为你的训练好的模型）")
    model = create_dummy_vit_model()
    print(model)
    model.eval()
    
    # 创建可视化器
    print("\n2. 初始化注意力可视化器")
    visualizer = AttentionVisualizer(model, save_dir='./demo_visualizations')
    
    # 修改模型以捕获注意力权重
    print("\n3. 配置模型以捕获注意力权重")
    visualizer.modify_attention_for_visualization()
    
    # 创建模拟输入数据（实际使用时替换为真实数据）
    print("\n4. 准备输入数据")
    video_tensor = create_dummy_video_data()
    print(f"   输入视频形状: {video_tensor.shape}")
    
    # 进行预测
    print("\n5. 进行模型预测")
    with torch.no_grad():
        prediction = model(video_tensor)
    print(f"   预测的注视角度: yaw={prediction[0,0]:.4f}, pitch={prediction[0,1]:.4f}")
    
    # 生成综合可视化报告
    print("\n6. 生成注意力可视化报告")
    
    report_dir = visualizer.generate_comprehensive_report(
            video_tensor, prediction, sample_name='demo_sample'
        )
    print(f"   报告保存在: {report_dir}")
    try:
        report_dir = visualizer.generate_comprehensive_report(
            video_tensor, prediction, sample_name='demo_sample'
        )
        print(f"   报告保存在: {report_dir}")
    except Exception as e:
        print(f"   报告生成失败: {e}")
        print("   继续进行单独的可视化演示...")
    
    # 演示单独的可视化功能
    print("\n7. 演示各种可视化功能")
    
    # 空间注意力可视化
    try:
        print("   - 空间注意力可视化")
        spatial_results = visualizer.visualize_spatial_attention(
            video_tensor, layer_idx=-1, frame_idx=0
        )
        
        if spatial_results:
            batch_key = list(spatial_results.keys())[0]
            spatial_attn = spatial_results[batch_key]['spatial_attention']
            print(f"     空间注意力图形状: {spatial_attn.shape}")
            print(f"     注意力最大值位置: {np.unravel_index(np.argmax(spatial_attn), spatial_attn.shape)}")
        
    except Exception as e:
        print(f"     空间注意力可视化失败: {e}")
    
    # 时间注意力可视化
    try:
        print("   - 时间注意力可视化")
        temporal_results = visualizer.visualize_temporal_attention(video_tensor, layer_idx=-1)
        
        if temporal_results:
            batch_key = list(temporal_results.keys())[0]
            temporal_attn = temporal_results[batch_key]
            print(f"     时间注意力形状: {temporal_attn.shape}")
            print(f"     最重要的帧: {np.argmax(temporal_attn)}")
        
    except Exception as e:
        print(f"     时间注意力可视化失败: {e}")
    
    # 注意力模式分析
    try:
        print("   - 注意力模式分析")
        analysis_results = visualizer.analyze_attention_pattern(video_tensor, prediction)
        print(f"     分析了 {len(analysis_results)} 个层的注意力模式")
        
    except Exception as e:
        print(f"     注意力模式分析失败: {e}")
    
    # 清理
    visualizer.remove_hooks()
    
    print("\n8. 演示完成！")
    print("\n使用说明:")
    print("- 将 create_dummy_vit_model() 替换为你的实际模型")
    print("- 将 create_dummy_video_data() 替换为实际的视频数据加载")
    print("- 使用 load_video_for_visualization() 或 load_image_sequence_for_visualization() 加载真实数据")
    print("- 查看生成的可视化图像了解模型的注意力模式")

def create_dummy_vit_model():
    """
    创建一个简化的ViT模型用于演示
    实际使用时请替换为你的真实模型
    """
    class DummyAttention(nn.Module):
        def __init__(self, dim=768, num_heads=12):
            super().__init__()
            self.num_heads = num_heads
            self.scale = (dim // num_heads) ** -0.5
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.attn_drop = nn.Dropout(0.0)
            self.proj_drop = nn.Dropout(0.0)
            
        def forward(self, x):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            
            # 保存注意力权重用于可视化
            self.last_attention_weights = attn.clone()
            
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x
    
    class DummyBlock(nn.Module):
        def __init__(self, dim=768):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = DummyAttention(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
                nn.Dropout(0.0)
            )
            
        def forward(self, x):
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
            return x
    
    class DummyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch_embed = DummyPatchEmbed()
            self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
            self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8*10*10, 768))  # CLS + patches
            
            # 创建多个transformer blocks
            self.blocks = nn.ModuleList([DummyBlock() for _ in range(12)])
            
            self.norm = nn.LayerNorm(768)
            self.head = nn.Linear(768, 2)  # 输出yaw和pitch
            
        def forward(self, x):
            B = x.shape[0]
            x = self.patch_embed(x)
            
            # 添加CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # 添加位置编码
            x = x + self.pos_embed[:, :x.size(1)]
            
            # 通过transformer blocks
            for block in self.blocks:
                x = block(x)
            
            x = self.norm(x)
            
            # 使用CLS token进行预测
            return self.head(x[:, 0])
    
    class DummyPatchEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Conv3d(3, 768, kernel_size=(2, 16, 16), stride=(2, 16, 16))
            self.input_token_size = (8, 10, 10)  # T, H, W
            
        def forward(self, x):
            # x: (B, C, T, H, W)
            x = self.proj(x)  # (B, 768, T', H', W')
            x = x.flatten(2).transpose(1, 2)  # (B, T'*H'*W', 768)
            return x
    
    return DummyViT()

def create_dummy_video_data():
    """
    创建模拟的视频数据用于演示
    实际使用时请替换为真实数据
    """
    # 创建一个渐变图案，模拟眼部图像
    B, C, T, H, W = 1, 3, 16, 160, 160
    
    video = torch.zeros(B, C, T, H, W)
    
    for t in range(T):
        for c in range(C):
            # 创建一个中心亮，边缘暗的图案，模拟眼部
            y, x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
            
            # 创建眼球图案
            eye_pattern = torch.exp(-(x**2 + y**2) / 0.3)
            
            # 添加一些随机噪声
            noise = torch.randn(H, W) * 0.1
            
            # 随时间变化的模式
            time_factor = torch.sin(torch.tensor(t / T * 2 * np.pi))
            
            video[0, c, t] = eye_pattern + noise + time_factor * 0.2
    
    # 归一化到[0, 1]
    video = (video - video.min()) / (video.max() - video.min())
    
    return video

def demonstrate_real_usage():
    """
    演示如何在真实场景中使用注意力可视化
    """
    print("\n" + "="*60)
    print("真实使用场景演示")
    print("="*60)
    
    print("\n假设你有以下文件:")
    print("- 训练好的模型: 'model.pth'")
    print("- 测试视频: 'test_video.mp4' 或图像序列目录")
    
    code_example = """
# 1. 加载训练好的模型
model = YourViTModel.load_from_checkpoint('path/to/model.pth')
model.eval()

# 2. 创建可视化器
visualizer = AttentionVisualizer(model, save_dir='./attention_analysis')

# 3. 加载测试数据
# 方式1: 从视频文件
video_tensor = load_video_for_visualization('test_video.mp4', target_size=(160, 160), num_frames=16)

# 方式2: 从图像序列
video_tensor = load_image_sequence_for_visualization('path/to/image/sequence/', target_size=(160, 160), num_frames=16)

# 4. 进行预测
with torch.no_grad():
    prediction = model(video_tensor)

# 5. 生成可视化报告
report_dir = visualizer.generate_comprehensive_report(
    video_tensor, prediction, sample_name='test_sample_001'
)

# 6. 分析特定层的注意力
spatial_attention = visualizer.visualize_spatial_attention(video_tensor, layer_idx=-1)
temporal_attention = visualizer.visualize_temporal_attention(video_tensor, layer_idx=-1)

# 7. 清理资源
visualizer.remove_hooks()

print(f"可视化结果保存在: {report_dir}")
"""
    
    print("\n使用代码示例:")
    print(code_example)
    
    print("\n生成的可视化内容包括:")
    print("1. 空间注意力热力图 - 显示模型关注的图像区域")
    print("2. 时间注意力图表 - 显示模型关注的时间帧")
    print("3. 注意力中心与预测角度的相关性分析")
    print("4. 多层注意力对比")
    print("5. 综合分析报告")

if __name__ == "__main__":
    # 运行演示
    demo_attention_visualization()
    
    # 显示真实使用方法
    demonstrate_real_usage()
