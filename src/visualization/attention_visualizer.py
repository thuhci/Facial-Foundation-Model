import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import cv2
from einops import rearrange
import os

class AttentionVisualizer:
    """
    注意力可视化组件，专门针对视频ViT模型的眼部注视角度回归任务
    """
    
    def __init__(self, model: nn.Module, save_dir: str = "./attention_visualizations"):
        """
        初始化可视化器
        
        Args:
            model: 训练好的ViT模型
            save_dir: 保存可视化结果的目录
        """
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储注意力权重的钩子
        self.attention_weights = {}
        self.hooks = []
        
    def register_hooks(self):
        """注册钩子函数来捕获注意力权重"""
        def get_attention_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attn') and hasattr(module.attn, 'attn_drop'):
                    # 对于标准的Attention层
                    if hasattr(module, 'last_attention_weights'):
                        self.attention_weights[name] = module.last_attention_weights.detach()
                return hook
        
        # 为所有Block注册钩子
        for name, module in self.model.named_modules():
            if 'blocks' in name and isinstance(module, (Block, LGBlock)):
                hook = get_attention_hook(name)
                handle = module.register_forward_hook(hook)
                self.hooks.append(handle)
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_weights.clear()
    
    def modify_attention_for_visualization(self):
        """修改模型中的注意力层以保存权重"""
        def modify_attention_forward(attention_module, module_name):
            original_forward = attention_module.forward
            
            def new_forward(x, context=None, mask=None):
                B, N, C = x.shape
                qkv_bias = None
                if hasattr(attention_module, 'q_bias') and attention_module.q_bias is not None:
                    if hasattr(attention_module, 'v_bias'):
                        qkv_bias = torch.cat((attention_module.q_bias, 
                                            torch.zeros_like(attention_module.v_bias, requires_grad=False), 
                                            attention_module.v_bias))
                
                if hasattr(attention_module, 'qkv'):  # 标准self-attention
                    qkv = torch.nn.functional.linear(input=x, weight=attention_module.qkv.weight, bias=qkv_bias)
                    qkv = qkv.reshape(B, N, 3, attention_module.num_heads, -1).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                else:  # cross-attention
                    q_bias, kv_bias = attention_module.q_bias, None
                    if attention_module.q_bias is not None:
                        kv_bias = torch.cat((torch.zeros_like(attention_module.v_bias, requires_grad=False), 
                                           attention_module.v_bias))
                    q = torch.nn.functional.linear(input=x, weight=attention_module.q.weight, bias=q_bias)
                    q = q.reshape(B, N, attention_module.num_heads, -1).transpose(1,2)
                    
                    kv_input = x if context is None else context
                    kv = torch.nn.functional.linear(input=kv_input, weight=attention_module.kv.weight, bias=kv_bias)
                    _, T2, _ = kv.shape
                    kv = kv.reshape(B, T2, 2, attention_module.num_heads, -1).permute(2, 0, 3, 1, 4)
                    k, v = kv[0], kv[1]

                q = q * attention_module.scale
                attn = (q @ k.transpose(-2, -1))

                if mask is not None:
                    nW = mask.shape[0]
                    attn = attn.view(B // nW, nW, attention_module.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                    attn = attn.view(-1, attention_module.num_heads, N, N)

                attn = attn.softmax(dim=-1)
                
                # 保存注意力权重到可视化器的字典中
                self.attention_weights[module_name] = attn.clone().detach()
                
                attn = attention_module.attn_drop(attn)
                x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
                x = attention_module.proj(x)
                x = attention_module.proj_drop(x)
                return x
            
            attention_module.forward = new_forward
        
        # 修改所有注意力层并记录模块名称
        for name, module in self.model.named_modules():
            if hasattr(module, 'attn') and hasattr(module.attn, 'scale'):
                modify_attention_forward(module.attn, name + '.attn')
            if hasattr(module, 'first_attn') and hasattr(module.first_attn, 'scale'):
                modify_attention_forward(module.first_attn, name + '.first_attn')
            if hasattr(module, 'second_attn') and hasattr(module.second_attn, 'scale'):
                modify_attention_forward(module.second_attn, name + '.second_attn')
            if hasattr(module, 'third_attn') and hasattr(module.third_attn, 'scale'):
                modify_attention_forward(module.third_attn, name + '.third_attn')

    def visualize_spatial_attention(self, 
                                  video_tensor: torch.Tensor, 
                                  layer_idx: int = -1, 
                                  head_idx: Optional[int] = None,
                                  frame_idx: int = 0,
                                  attn_type: str = 'auto') -> Dict:
        """
        可视化空间注意力模式
        
        Args:
            video_tensor: 输入视频张量 (B, C, T, H, W)
            layer_idx: 要可视化的层索引，-1表示最后一层
            head_idx: 注意力头索引，None表示平均所有头
            frame_idx: 要可视化的帧索引
            attn_type: 注意力类型 ('auto', 'first', 'second', 'third')
            
        Returns:
            包含注意力图的字典
        """
        self.model.eval()
        
        # 清空之前的注意力权重
        self.attention_weights.clear()
        
        with torch.no_grad():
            # 前向传播
            _ = self.model(video_tensor)
            
            # 获取所有包含'blocks'的注意力层
            layer_names = [name for name in self.attention_weights.keys() if 'blocks' in name]
            print(f"可用的注意力层: {layer_names}")
            print(f"总共捕获到 {len(self.attention_weights)} 个注意力权重")
            
            if not layer_names:
                print("警告: 没有捕获到任何注意力权重！")
                return {}
            
            # 确定要分析的注意力类型
            if attn_type == 'auto':
                # 自动选择：对于LGBlock，优先选择third_attn（局部到全局的交叉注意力）
                if any('third_attn' in name for name in layer_names):
                    attn_suffix = 'third_attn'
                elif any('first_attn' in name for name in layer_names):
                    attn_suffix = 'first_attn'
                elif any('second_attn' in name for name in layer_names):
                    attn_suffix = 'second_attn'
                else:
                    attn_suffix = 'attn'  # 标准attention
            else:
                attn_suffix = f'{attn_type}_attn' if attn_type in ['first', 'second', 'third'] else 'attn'
            
            # 过滤出指定类型的注意力层
            filtered_layers = [name for name in layer_names if name.endswith(attn_suffix)]
            if not filtered_layers:
                print(f"警告: 没有找到类型为 '{attn_suffix}' 的注意力层")
                filtered_layers = layer_names  # 回退到所有层
            
            print(f"使用注意力类型: {attn_suffix}")
            print(f"过滤后的层: {filtered_layers}")
            
            # 选择目标层
            if layer_idx >= 0:
                if layer_idx < len(filtered_layers):
                    target_layer = filtered_layers[layer_idx]
                else:
                    print(f"警告: 层索引 {layer_idx} 超出范围，使用最后一层")
                    target_layer = filtered_layers[-1]
            else:
                # 负索引
                if abs(layer_idx) <= len(filtered_layers):
                    target_layer = filtered_layers[layer_idx]
                else:
                    print(f"警告: 层索引 {layer_idx} 超出范围，使用最后一层")
                    target_layer = filtered_layers[-1]
            
            print(f"使用的注意力层: {target_layer}")
            attn_weights = self.attention_weights[target_layer]
            print(f"注意力权重形状: {attn_weights.shape}")
            
            # 处理注意力权重
            if head_idx is not None:
                attn_weights = attn_weights[:, head_idx]
            else:
                attn_weights = attn_weights.mean(dim=1)  # 平均所有头
            
            print(f"处理后的注意力权重形状: {attn_weights.shape}")
            
            # 根据注意力类型进行不同的处理
            results = {}
            
            if 'third_attn' in target_layer:
                # third_attn: 局部tokens到全局messenger tokens的交叉注意力
                # 形状: (B, local_tokens, messenger_tokens)
                results = self._process_third_attention(attn_weights, video_tensor, frame_idx)
                
            elif 'second_attn' in target_layer:
                # second_attn: 全局messenger tokens之间的自注意力
                # 形状: (B, messenger_tokens, messenger_tokens)
                results = self._process_second_attention(attn_weights, video_tensor, frame_idx)
                
            elif 'first_attn' in target_layer:
                # first_attn: 局部区域内的注意力
                # 形状: (B*N, region_size+1, region_size+1)
                results = self._process_first_attention(attn_weights, video_tensor, frame_idx)
                
            else:
                # 标准注意力
                results = self._process_standard_attention(attn_weights, video_tensor, frame_idx)
            
            return results

    def _process_third_attention(self, attn_weights, video_tensor, frame_idx):
        """处理third_attn的可视化（局部到全局的交叉注意力）"""
        # attn_weights形状: (B, local_tokens, messenger_tokens)
        results = {}
        
        # 获取模型配置
        if hasattr(self.model, 'lg_region_size'):
            t_region, h_region, w_region = self.model.lg_region_size
            nt, nh, nw = self.model.lg_num_region_size
        else:
            # 从配置中推断
            patch_embed = self.model.patch_embed
            T, H, W = patch_embed.input_token_size
            # 假设region配置
            t_region, h_region, w_region = 2, 2, 10
            nt, nh, nw = T//t_region, H//h_region, W//w_region
        
        print(f"Region配置: t_region={t_region}, h_region={h_region}, w_region={w_region}")
        print(f"Region数量: nt={nt}, nh={nh}, nw={nw}")
        
        for batch_idx in range(attn_weights.shape[0]):
            batch_results = {}
            
            # 取平均注意力权重 (local_tokens, messenger_tokens)
            local_to_global_attn = attn_weights[batch_idx]  # (800, 20)
            
            # 将局部tokens重新组织为时空结构
            # local_tokens = nt*nh*nw * t_region*h_region*w_region = 800
            local_tokens_spatial = local_to_global_attn.mean(dim=1)  # 对所有messenger求平均
            
            # 重塑为时空形状
            try:
                # 重塑为 (nt*nh*nw, t_region*h_region*w_region)
                num_regions = nt * nh * nw
                region_size = t_region * h_region * w_region
                
                if len(local_tokens_spatial) == num_regions * region_size:
                    spatial_attn = local_tokens_spatial.reshape(num_regions, region_size)
                    # 对每个region内部求平均，得到region级别的注意力
                    region_attn = spatial_attn.mean(dim=1)  # (num_regions,)
                    
                    # 重塑为空间网格
                    region_attn_spatial = region_attn.reshape(nt, nh, nw)
                    
                    # 选择指定的时间帧
                    if frame_idx < nt:
                        frame_attn = region_attn_spatial[frame_idx]  # (nh, nw)
                    else:
                        frame_attn = region_attn_spatial[0]  # 使用第一帧
                    
                    # 上采样到原始图像尺寸
                    original_h, original_w = video_tensor.shape[-2:]
                    frame_attn_resized = torch.nn.functional.interpolate(
                        frame_attn.unsqueeze(0).unsqueeze(0), 
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                    
                    batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
                    batch_results['temporal_attention'] = region_attn_spatial.mean(dim=(-1, -2)).cpu().numpy()
                    batch_results['raw_attention'] = region_attn_spatial.cpu().numpy()
                    
            except Exception as e:
                print(f"处理third_attn时出错: {e}")
                # 简单的fallback处理
                side_len = int(np.sqrt(len(local_tokens_spatial)))
                if side_len * side_len == len(local_tokens_spatial):
                    spatial_attn = local_tokens_spatial.reshape(side_len, side_len)
                    original_h, original_w = video_tensor.shape[-2:]
                    frame_attn_resized = torch.nn.functional.interpolate(
                        spatial_attn.unsqueeze(0).unsqueeze(0), 
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                    batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
            
            results[f'batch_{batch_idx}'] = batch_results
        
        return results

    def _process_second_attention(self, attn_weights, video_tensor, frame_idx):
        """处理second_attn的可视化（全局messenger tokens之间的自注意力）"""
        # attn_weights形状: (B, messenger_tokens, messenger_tokens)
        results = {}
        
        for batch_idx in range(attn_weights.shape[0]):
            batch_results = {}
            
            # 取对角线注意力作为自注意力强度
            self_attn = torch.diag(attn_weights[batch_idx])  # (messenger_tokens,)
            
            # 获取region配置
            if hasattr(self.model, 'lg_num_region_size'):
                nt, nh, nw = self.model.lg_num_region_size
            else:
                # 推断region数量
                num_messengers = len(self_attn)
                nt = nh = nw = int(round(num_messengers ** (1/3)))
                if nt * nh * nw != num_messengers:
                    # 2D layout
                    nt = 1
                    nh = nw = int(np.sqrt(num_messengers))
            
            try:
                # 重塑为时空形状
                if nt * nh * nw == len(self_attn):
                    region_attn = self_attn.reshape(nt, nh, nw)
                    
                    # 选择时间帧
                    if frame_idx < nt:
                        frame_attn = region_attn[frame_idx]
                    else:
                        frame_attn = region_attn.mean(dim=0)  # 平均所有时间帧
                    
                    # 上采样
                    original_h, original_w = video_tensor.shape[-2:]
                    frame_attn_resized = torch.nn.functional.interpolate(
                        frame_attn.unsqueeze(0).unsqueeze(0), 
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                    
                    batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
                    batch_results['temporal_attention'] = region_attn.mean(dim=(-1, -2)).cpu().numpy()
                    batch_results['raw_attention'] = region_attn.cpu().numpy()
            
            except Exception as e:
                print(f"处理second_attn时出错: {e}")
            
            results[f'batch_{batch_idx}'] = batch_results
        
        return results

    def _process_first_attention(self, attn_weights, video_tensor, frame_idx):
        """处理first_attn的可视化（局部区域内的注意力）"""
        # attn_weights形状: (B*N, region_size+1, region_size+1)
        results = {}
        
        # 这里需要更复杂的处理，因为是局部注意力
        # 简化处理：取messenger token (index 0) 对其他tokens的注意力
        batch_size = video_tensor.shape[0]
        num_regions = attn_weights.shape[0] // batch_size
        
        for batch_idx in range(batch_size):
            batch_results = {}
            
            # 提取该batch的所有region的注意力
            start_idx = batch_idx * num_regions
            end_idx = (batch_idx + 1) * num_regions
            batch_attn = attn_weights[start_idx:end_idx]  # (num_regions, region_size+1, region_size+1)
            
            # 取messenger token对局部tokens的注意力
            messenger_to_local = batch_attn[:, 0, 1:]  # (num_regions, region_size)
            
            # 对所有region求平均，得到一个全局的注意力图
            avg_attn = messenger_to_local.mean(dim=0)  # (region_size,)
            
            # 尝试重塑为空间形状
            region_size = len(avg_attn)
            side_len = int(np.sqrt(region_size))
            
            if side_len * side_len == region_size:
                spatial_attn = avg_attn.reshape(side_len, side_len)
                
                # 上采样
                original_h, original_w = video_tensor.shape[-2:]
                frame_attn_resized = torch.nn.functional.interpolate(
                    spatial_attn.unsqueeze(0).unsqueeze(0), 
                    size=(original_h, original_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
            
            results[f'batch_{batch_idx}'] = batch_results
        
        return results

    def _process_standard_attention(self, attn_weights, video_tensor, frame_idx):
        """处理标准attention的可视化"""
        results = {}
        
        # 获取patch embedding的信息
        patch_embed = self.model.patch_embed
        if hasattr(patch_embed, 'input_token_size'):
            T, H, W = patch_embed.input_token_size
            print(f"从patch_embed获取的尺寸: T={T}, H={H}, W={W}")
        else:
            # 估计空间尺寸
            total_patches = attn_weights.shape[-1] - 1  # 减去CLS token
            print(f"总patch数（不含CLS）: {total_patches}")
            
            # 根据输入视频尺寸估计patch数量
            input_frames = video_tensor.shape[2]
            tubelet_size = 2  # 默认假设
            T = input_frames // tubelet_size
            spatial_patches = total_patches // T
            H = W = int(np.sqrt(spatial_patches))
            print(f"估计的尺寸: T={T}, H={H}, W={W}")
        
        for batch_idx in range(video_tensor.shape[0]):
            batch_results = {}
            
            # CLS token对所有patch的注意力
            cls_attention = attn_weights[batch_idx, 0, 1:]  # 除去CLS token自己
            print(f"CLS注意力形状: {cls_attention.shape}")
            
            # 重塑为时空注意力图
            expected_patches = T * H * W
            if len(cls_attention) == expected_patches:
                spatial_temporal_attn = cls_attention.reshape(T, H, W)
                print(f"重塑为时空注意力图: {spatial_temporal_attn.shape}")
                
                # 确保frame_idx在有效范围内
                if frame_idx >= T:
                    frame_idx = T - 1
                    print(f"帧索引超出范围，使用最后一帧: {frame_idx}")
                
                # 对指定帧的空间注意力
                frame_attn = spatial_temporal_attn[frame_idx]  # (H, W)
                
                # 上采样到原始图像尺寸
                original_h, original_w = video_tensor.shape[-2:]
                frame_attn_resized = torch.nn.functional.interpolate(
                    frame_attn.unsqueeze(0).unsqueeze(0), 
                    size=(original_h, original_w), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze()
                
                batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
                batch_results['temporal_attention'] = spatial_temporal_attn.mean(dim=(-1, -2)).cpu().numpy()
                batch_results['raw_attention'] = spatial_temporal_attn.cpu().numpy()
            else:
                print(f"警告: CLS注意力长度 {len(cls_attention)} 与期望的 {expected_patches} 不匹配")
                # 尝试直接使用注意力权重
                if len(cls_attention) > 0:
                    # 简单的空间注意力可视化
                    side_len = int(np.sqrt(len(cls_attention)))
                    if side_len * side_len == len(cls_attention):
                        spatial_attn = cls_attention.reshape(side_len, side_len)
                        original_h, original_w = video_tensor.shape[-2:]
                        frame_attn_resized = torch.nn.functional.interpolate(
                            spatial_attn.unsqueeze(0).unsqueeze(0), 
                            size=(original_h, original_w), 
                            mode='bilinear', 
                            align_corners=False
                        ).squeeze()
                        batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
            
            results[f'batch_{batch_idx}'] = batch_results
        
        return results

    def visualize_temporal_attention(self, video_tensor: torch.Tensor, layer_idx: int = -1) -> Dict:
        """
        可视化时间注意力模式
        
        Args:
            video_tensor: 输入视频张量 (B, C, T, H, W)
            layer_idx: 要可视化的层索引
            
        Returns:
            时间注意力结果
        """
        attention_results = self.visualize_spatial_attention(video_tensor, layer_idx)
        
        temporal_results = {}
        for batch_key, batch_data in attention_results.items():
            if 'temporal_attention' in batch_data:
                temporal_results[batch_key] = batch_data['temporal_attention']
        
        return temporal_results

    def plot_attention_heatmap(self, 
                             original_frame: np.ndarray, 
                             attention_map: np.ndarray, 
                             title: str = "Attention Heatmap",
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制注意力热力图覆盖在原始图像上
        
        Args:
            original_frame: 原始帧 (H, W, C) 或 (H, W)
            attention_map: 注意力图 (H, W)
            title: 图像标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 原始图像
        if len(original_frame.shape) == 3:
            axes[0].imshow(original_frame)
        else:
            axes[0].imshow(original_frame, cmap='gray')
        axes[0].set_title('Original Frame')
        axes[0].axis('off')
        
        # 注意力图
        im1 = axes[1].imshow(attention_map, cmap='hot', interpolation='bilinear')
        axes[1].set_title('Attention Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])
        
        # 叠加图
        if len(original_frame.shape) == 3:
            axes[2].imshow(original_frame)
        else:
            axes[2].imshow(original_frame, cmap='gray')
        
        axes[2].imshow(attention_map, cmap='hot', alpha=0.6, interpolation='bilinear')
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def plot_temporal_attention(self, 
                              temporal_weights: np.ndarray, 
                              title: str = "Temporal Attention",
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制时间注意力
        
        Args:
            temporal_weights: 时间注意力权重 (T,)
            title: 图像标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        frames = np.arange(len(temporal_weights))
        ax.bar(frames, temporal_weights, alpha=0.7, color='skyblue')
        ax.plot(frames, temporal_weights, marker='o', color='red', linewidth=2)
        
        ax.set_xlabel('Frame Index')
        ax.set_ylabel('Attention Weight')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # 标记最重要的帧
        max_frame = np.argmax(temporal_weights)
        ax.annotate(f'Peak: Frame {max_frame}', 
                   xy=(max_frame, temporal_weights[max_frame]),
                   xytext=(max_frame + 1, temporal_weights[max_frame] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig

    def analyze_attention_pattern(self, video_tensor: torch.Tensor, prediction: torch.Tensor) -> Dict:
        """
        分析注意力模式与预测结果的关系
        
        Args:
            video_tensor: 输入视频
            prediction: 模型预测的注视角度 [pitch, yaw]
            
        Returns:
            分析结果
        """
        results = {}
        
        # 对于LGBlock，分析不同类型的注意力
        attention_types = ['third', 'second', 'first']  # 按重要性排序
        
        # 获取多层注意力
        layer_indices = [-3, -2, -1]  # 最后三层
        for layer_idx in layer_indices:
            for attn_type in attention_types:
                try:
                    attention_data = self.visualize_spatial_attention(
                        video_tensor, layer_idx, attn_type=attn_type
                    )
                    if attention_data:  # 如果成功获取到数据
                        results[f'layer_{layer_idx}_{attn_type}'] = attention_data
                        break  # 成功获取一种类型就继续下一层
                except Exception as e:
                    print(f"分析 layer_{layer_idx}_{attn_type} 时出错: {e}")
                    continue
        
        # 计算注意力中心
        for layer_key, layer_data in results.items():
            for batch_key, batch_data in layer_data.items():
                if 'spatial_attention' in batch_data:
                    spatial_attn = batch_data['spatial_attention']
                    h, w = spatial_attn.shape
                    
                    # 计算注意力重心
                    y_coords, x_coords = np.mgrid[0:h, 0:w]
                    total_attention = spatial_attn.sum()
                    
                    if total_attention > 0:
                        center_y = (y_coords * spatial_attn).sum() / total_attention
                        center_x = (x_coords * spatial_attn).sum() / total_attention
                        
                        # 归一化到[-1, 1]
                        norm_center_y = (center_y / h) * 2 - 1
                        norm_center_x = (center_x / w) * 2 - 1
                        
                        batch_data['attention_center'] = (norm_center_x, norm_center_y)
                        
                        # 与预测角度的关联性分析
                        if prediction is not None:
                            pred_yaw, pred_pitch = prediction[0].cpu().numpy()
                            
                            # 计算注意力中心与预测角度的相关性
                            correlation_x = np.corrcoef([norm_center_x], [pred_yaw])[0, 1]
                            correlation_y = np.corrcoef([norm_center_y], [pred_pitch])[0, 1]
                            
                            batch_data['correlation'] = {
                                'x_yaw': correlation_x,
                                'y_pitch': correlation_y
                            }
        
        return results

    def generate_comprehensive_report(self, 
                                    video_tensor: torch.Tensor, 
                                    prediction: torch.Tensor,
                                    sample_name: str = "sample") -> str:
        """
        生成综合的注意力分析报告
        
        Args:
            video_tensor: 输入视频
            prediction: 预测结果
            sample_name: 样本名称
            
        Returns:
            报告保存路径
        """
        # 设置可视化
        self.modify_attention_for_visualization()
        
        # 分析注意力模式
        analysis_results = self.analyze_attention_pattern(video_tensor, prediction)
        
        # 创建保存目录
        sample_dir = os.path.join(self.save_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        # 保存各种可视化
        for layer_key, layer_data in analysis_results.items():
            for batch_key, batch_data in layer_data.items():
                if 'spatial_attention' in batch_data:
                    # 空间注意力热力图
                    original_frame = video_tensor[0, :, 0].permute(1, 2, 0).cpu().numpy()
                    original_frame = (original_frame - original_frame.min()) / (original_frame.max() - original_frame.min())
                    
                    spatial_attn = batch_data['spatial_attention']
                    save_path = os.path.join(sample_dir, f'{layer_key}_{batch_key}_spatial.png')
                    self.plot_attention_heatmap(original_frame, spatial_attn, 
                                               f'Spatial Attention - {layer_key}', save_path)
                    plt.close()
                    
                if 'temporal_attention' in batch_data:
                    # 时间注意力
                    temporal_attn = batch_data['temporal_attention']
                    save_path = os.path.join(sample_dir, f'{layer_key}_{batch_key}_temporal.png')
                    self.plot_temporal_attention(temporal_attn, 
                                               f'Temporal Attention - {layer_key}', save_path)
                    plt.close()
        
        # 生成文本报告
        report_path = os.path.join(sample_dir, 'attention_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Attention Analysis Report for {sample_name}\n")
            f.write("=" * 50 + "\n\n")
            
            if prediction is not None:
                pred_yaw, pred_pitch = prediction[0].cpu().numpy()
                f.write(f"Predicted Gaze Direction:\n")
                f.write(f"  Yaw: {pred_yaw:.4f} rad ({np.degrees(pred_yaw):.2f}°)\n")
                f.write(f"  Pitch: {pred_pitch:.4f} rad ({np.degrees(pred_pitch):.2f}°)\n\n")
            
            f.write("Attention Analysis:\n")
            for layer_key, layer_data in analysis_results.items():
                f.write(f"\n{layer_key.upper()}:\n")
                for batch_key, batch_data in layer_data.items():
                    if 'attention_center' in batch_data:
                        center_x, center_y = batch_data['attention_center']
                        f.write(f"  {batch_key} - Attention Center: ({center_x:.4f}, {center_y:.4f})\n")
                        
                        if 'correlation' in batch_data:
                            corr = batch_data['correlation']
                            f.write(f"    Correlation with prediction: X-Yaw={corr['x_yaw']:.4f}, Y-Pitch={corr['y_pitch']:.4f}\n")
        
        return sample_dir

# 从layers.py导入必要的类
try:
    from ..models.layers import Block, LGBlock
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    sys.path.append('/home/qzk/Facial-Foundation-Model/src/models')
    from layers import Block, LGBlock
