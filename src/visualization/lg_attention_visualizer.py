import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import cv2
from einops import rearrange
import os

class LGAttentionVisualizer:
    """
    专门针对LGBlock (Local-Global Block) 架构的注意力可视化组件
    支持first_attn, second_attn, third_attn的可视化
    """
    
    def __init__(self, model: nn.Module, save_dir: str = "./lg_attention_visualizations"):
        """
        初始化可视化器
        
        Args:
            model: 使用LGBlock的ViT模型
            save_dir: 保存可视化结果的目录
        """
        self.model = model
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 存储注意力权重
        self.attention_weights = {}
        self.hooks = []
        
        # 获取模型配置
        self.lg_region_size = getattr(model, 'lg_region_size', [2, 2, 10])  # [T, H, W]
        self.lg_num_region_size = getattr(model, 'lg_num_region_size', None)
        if self.lg_num_region_size is None and hasattr(model, 'patch_embed'):
            # 计算region数量
            if hasattr(model.patch_embed, 'input_token_size'):
                token_size = model.patch_embed.input_token_size
                self.lg_num_region_size = [token_size[i] // self.lg_region_size[i] for i in range(3)]
            else:
                # 使用默认估计
                self.lg_num_region_size = [4, 5, 1]  # 基于你的输出推算
        
        print(f"LG Region Size: {self.lg_region_size}")
        print(f"LG Num Region Size: {self.lg_num_region_size}")
        
        # 计算总region数量
        if self.lg_num_region_size:
            self.total_regions = self.lg_num_region_size[0] * self.lg_num_region_size[1] * self.lg_num_region_size[2]
            print(f"Total Regions: {self.total_regions}")
        else:
            self.total_regions = 20  # 从你的输出推算
        
    def modify_attention_for_visualization(self):
        """修改LGBlock中的注意力层以保存权重"""
        def modify_lg_attention_forward(attention_module, module_name):
            original_forward = attention_module.forward
            
            def new_forward(x, context=None, mask=None):
                # 先保存输入，用于调试
                input_shape = x.shape if hasattr(x, 'shape') else 'Unknown'
                context_shape = context.shape if context is not None and hasattr(context, 'shape') else 'None'
                
                # 重新实现注意力计算以获取权重
                B, N, C = x.shape
                qkv_bias = None
                
                if hasattr(attention_module, 'qkv'):  # 标准self-attention (first_attn 和 second_attn)
                    if hasattr(attention_module, 'q_bias') and attention_module.q_bias is not None:
                        if hasattr(attention_module, 'v_bias'):
                            qkv_bias = torch.cat((attention_module.q_bias, 
                                                torch.zeros_like(attention_module.v_bias, requires_grad=False), 
                                                attention_module.v_bias))
                    
                    qkv = torch.nn.functional.linear(input=x, weight=attention_module.qkv.weight, bias=qkv_bias)
                    qkv = qkv.reshape(B, N, 3, attention_module.num_heads, -1).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv[0], qkv[1], qkv[2]
                    
                    q = q * attention_module.scale
                    attn = (q @ k.transpose(-2, -1))
                    
                    if mask is not None:
                        nW = mask.shape[0]
                        attn = attn.view(B // nW, nW, attention_module.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                        attn = attn.view(-1, attention_module.num_heads, N, N)
                    
                    attn = attn.softmax(dim=-1)
                    
                    # 保存注意力权重
                    self.attention_weights[module_name] = attn.clone().detach()
                    
                    attn = attention_module.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
                    x = attention_module.proj(x)
                    x = attention_module.proj_drop(x)
                    
                elif hasattr(attention_module, 'q') and hasattr(attention_module, 'kv'):  # cross-attention (third_attn)
                    q_bias, kv_bias = None, None
                    if hasattr(attention_module, 'q_bias') and attention_module.q_bias is not None:
                        q_bias = attention_module.q_bias
                        if hasattr(attention_module, 'v_bias'):
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
                    attn = attn.softmax(dim=-1)
                    
                    # 保存注意力权重
                    self.attention_weights[module_name] = attn.clone().detach()
                    
                    attn = attention_module.attn_drop(attn)
                    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
                    x = attention_module.proj(x)
                    x = attention_module.proj_drop(x)
                else:
                    # 如果无法重新计算，调用原始方法并尝试捕获权重
                    x = original_forward(x, context, mask)
                    # 尝试从模块中获取保存的权重
                    if hasattr(attention_module, 'last_attention_weights'):
                        self.attention_weights[module_name] = attention_module.last_attention_weights.clone().detach()
                
                return x
            
            attention_module.forward = new_forward
        
        # 修改所有LGBlock中的注意力层
        for name, module in self.model.named_modules():
            if hasattr(module, 'first_attn'):
                modify_lg_attention_forward(module.first_attn, name + '.first_attn')
            if hasattr(module, 'second_attn'):
                modify_lg_attention_forward(module.second_attn, name + '.second_attn')
            if hasattr(module, 'third_attn'):
                modify_lg_attention_forward(module.third_attn, name + '.third_attn')

    def visualize_lg_attention(self, 
                              video_tensor: torch.Tensor, 
                              layer_idx: int = -1, 
                              head_idx: Optional[int] = None,
                              attn_type: str = 'third',
                              frame_idx: int = 0) -> Dict:
        """
        可视化LGBlock的注意力模式
        
        Args:
            video_tensor: 输入视频张量 (B, C, T, H, W)
            layer_idx: 要可视化的层索引，-1表示最后一层
            head_idx: 注意力头索引，None表示平均所有头
            attn_type: 注意力类型 ('first', 'second', 'third')
            frame_idx: 要可视化的帧索引
            
        Returns:
            包含注意力图的字典
        """
        self.model.eval()
        
        # 清空之前的注意力权重
        self.attention_weights.clear()
        
        with torch.no_grad():
            # 前向传播
            _ = self.model(video_tensor)
            
            # 获取指定类型的注意力层
            layer_names = [name for name in self.attention_weights.keys() 
                          if 'blocks' in name and f'.{attn_type}_attn' in name]
            
            print(f"可用的 {attn_type}_attn 层: {layer_names}")
            
            if not layer_names:
                print(f"警告: 没有找到 {attn_type}_attn 层！")
                return {}
            
            if layer_idx >= 0:
                if layer_idx < len(layer_names):
                    target_layer = layer_names[layer_idx]
                else:
                    print(f"警告: 层索引 {layer_idx} 超出范围，使用最后一层")
                    target_layer = layer_names[-1]
            else:
                # 负索引
                if abs(layer_idx) <= len(layer_names):
                    target_layer = layer_names[layer_idx]
                else:
                    print(f"警告: 层索引 {layer_idx} 超出范围，使用最后一层")
                    target_layer = layer_names[-1]
            
            print(f"使用的注意力层: {target_layer}")
            attn_weights = self.attention_weights[target_layer]
            print(f"注意力权重形状: {attn_weights.shape}")
            
            # 处理不同类型的注意力
            if attn_type == 'third':
                return self._visualize_third_attention(attn_weights, video_tensor, head_idx, frame_idx)
            elif attn_type == 'second':
                return self._visualize_second_attention(attn_weights, video_tensor, head_idx, frame_idx)
            elif attn_type == 'first':
                return self._visualize_first_attention(attn_weights, video_tensor, head_idx, frame_idx)
            else:
                print(f"不支持的注意力类型: {attn_type}")
                return {}

    def _visualize_third_attention(self, attn_weights, video_tensor, head_idx, frame_idx):
        """
        可视化third_attn: local tokens -> messenger tokens的交叉注意力
        形状通常是 (B, H, N_local, N_messenger)
        """
        results = {}
        B, H, N_local, N_messenger = attn_weights.shape
        
        # 处理多头注意力
        if head_idx is not None:
            attn = attn_weights[:, head_idx]  # (B, N_local, N_messenger)
        else:
            attn = attn_weights.mean(dim=1)  # (B, N_local, N_messenger)
        
        # 获取patch embedding信息
        patch_embed = self.model.patch_embed
        if hasattr(patch_embed, 'input_token_size'):
            T, H_patch, W_patch = patch_embed.input_token_size
        else:
            # 从region size估计
            T_region, H_region, W_region = self.lg_region_size
            total_regions = N_messenger
            T = video_tensor.shape[2] // 2  # 假设tubelet_size=2
            spatial_regions = total_regions // T
            H_patch = int(np.sqrt(spatial_regions))
            W_patch = H_patch
        
        print(f"Patch 维度: T={T}, H={H_patch}, W={W_patch}")
        print(f"Local tokens: {N_local}, Messenger tokens: {N_messenger}")
        
        for batch_idx in range(attn_weights.shape[0]):
            batch_results = {}
            
            # 获取每个messenger token的注意力分布
            # 对所有messenger tokens的注意力求和，得到每个local token的重要性
            local_importance = attn[batch_idx].sum(dim=1)  # (N_local,)
            
            # 重塑为时空形状
            if N_local == T * H_patch * W_patch:
                # 标准情况：每个时空位置一个token
                spatial_temporal_attn = local_importance.reshape(T, H_patch, W_patch)
                
                # 提取指定帧
                if frame_idx >= T:
                    frame_idx = T - 1
                frame_attn = spatial_temporal_attn[frame_idx]  # (H_patch, W_patch)
                
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
                print(f"警告: local tokens数量 {N_local} 与期望的 {T * H_patch * W_patch} 不匹配")
                # 尝试简单重塑
                side_len = int(np.sqrt(N_local))
                if side_len * side_len == N_local:
                    spatial_attn = local_importance.reshape(side_len, side_len)
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

    def _visualize_second_attention(self, attn_weights, video_tensor, head_idx, frame_idx):
        """
        可视化second_attn: messenger tokens之间的自注意力
        形状通常是 (B, H, N_messenger, N_messenger)
        """
        results = {}
        B, H, N_messenger, _ = attn_weights.shape
        
        print(f"Second attention详细信息: B={B}, H={H}, N_messenger={N_messenger}")
        print(f"预期的total_regions: {self.total_regions}")
        
        # 处理多头注意力
        if head_idx is not None:
            attn = attn_weights[:, head_idx]  # (B, N_messenger, N_messenger)
        else:
            attn = attn_weights.mean(dim=1)  # (B, N_messenger, N_messenger)
        
        # 获取每个messenger token的重要性（通过接收到的注意力）
        messenger_importance = attn.sum(dim=-1)  # (B, N_messenger) - 每个token接收到的总注意力
        
        print(f"Messenger importance shape: {messenger_importance.shape}")
        print(f"Messenger importance range: [{messenger_importance.min():.4f}, {messenger_importance.max():.4f}]")
        
        for batch_idx in range(attn_weights.shape[0]):
            batch_results = {}
            
            importance = messenger_importance[batch_idx]  # (N_messenger,)
            
            # 检查是否与预期的region数量匹配
            if N_messenger == self.total_regions and self.lg_num_region_size:
                nt, nh, nw = self.lg_num_region_size
                print(f"重塑messenger importance为: ({nt}, {nh}, {nw})")
                
                try:
                    # 重塑为时空形状
                    spatial_temporal_attn = importance.reshape(nt, nh, nw)
                    
                    # 选择时间帧
                    if frame_idx >= nt:
                        frame_idx = nt - 1
                    frame_attn = spatial_temporal_attn[frame_idx]  # (nh, nw)
                    
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
                    print(f"Second attention成功生成空间注意力，形状: {frame_attn_resized.shape}")
                    
                except Exception as e:
                    print(f"重塑messenger importance失败: {e}")
            
            # 如果上面失败，尝试简单的1D到2D映射
            if 'spatial_attention' not in batch_results:
                print("尝试简单的1D到2D映射...")
                side_len = int(np.sqrt(N_messenger))
                if side_len * side_len == N_messenger:
                    spatial_attn = importance.reshape(side_len, side_len)
                    original_h, original_w = video_tensor.shape[-2:]
                    frame_attn_resized = torch.nn.functional.interpolate(
                        spatial_attn.unsqueeze(0).unsqueeze(0), 
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                    batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
                    print(f"Simple mapping成功，形状: {frame_attn_resized.shape}")
                else:
                    print(f"无法进行简单映射：{N_messenger} 不是完全平方数")
            
            results[f'batch_{batch_idx}'] = batch_results
        
        return results

    def _visualize_first_attention(self, attn_weights, video_tensor, head_idx, frame_idx):
        """
        可视化first_attn: 局部区域内的注意力
        形状通常是 (B*N_regions, H, S, S) 其中 S = 1 + region_tokens
        """
        results = {}
        BN, H, S, _ = attn_weights.shape
        
        print(f"First attention详细信息: BN={BN}, H={H}, S={S}")
        print(f"预期的每个region的token数: {S-1} (不含messenger)")
        
        # 处理多头注意力
        if head_idx is not None:
            attn = attn_weights[:, head_idx]  # (B*N_regions, S, S)
        else:
            attn = attn_weights.mean(dim=1)  # (B*N_regions, S, S)
        
        # 估算batch size和region数量
        B = video_tensor.shape[0]
        N_regions = BN // B
        
        print(f"推算: B={B}, N_regions={N_regions}, 预期total_regions={self.total_regions}")
        
        # 检查是否匹配
        if N_regions != self.total_regions:
            print(f"警告: 推算的region数量 {N_regions} 与预期的 {self.total_regions} 不匹配")
        
        # 每个区域内，messenger token (index 0) 对其他tokens的注意力
        messenger_to_local = attn[:, 0, 1:]  # (B*N_regions, S-1)
        
        print(f"Messenger to local shape: {messenger_to_local.shape}")
        print(f"Attention range: [{messenger_to_local.min():.4f}, {messenger_to_local.max():.4f}]")
        
        # 重塑为 (B, N_regions, S-1)
        messenger_to_local = messenger_to_local.reshape(B, N_regions, S-1)
        
        for batch_idx in range(B):
            batch_results = {}
            
            batch_attn = messenger_to_local[batch_idx]  # (N_regions, S-1)
            
            # 计算每个区域的平均注意力作为区域重要性
            region_importance = batch_attn.mean(dim=1)  # (N_regions,)
            
            print(f"Region importance shape: {region_importance.shape}")
            print(f"Region importance range: [{region_importance.min():.4f}, {region_importance.max():.4f}]")
            
            # 尝试映射到空间位置
            if N_regions == self.total_regions and self.lg_num_region_size:
                nt, nh, nw = self.lg_num_region_size
                print(f"重塑region importance为: ({nt}, {nh}, {nw})")
                
                try:
                    spatial_temporal_attn = region_importance.reshape(nt, nh, nw)
                    
                    if frame_idx >= nt:
                        frame_idx = nt - 1
                    frame_attn = spatial_temporal_attn[frame_idx]  # (nh, nw)
                    
                    # 上采样
                    original_h, original_w = video_tensor.shape[-2:]
                    frame_attn_resized = torch.nn.functional.interpolate(
                        frame_attn.unsqueeze(0).unsqueeze(0), 
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                    
                    batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
                    batch_results['temporal_attention'] = spatial_temporal_attn.mean(dim=(-1, -2)).cpu().numpy()
                    print(f"First attention成功生成空间注意力，形状: {frame_attn_resized.shape}")
                    
                except Exception as e:
                    print(f"重塑region importance失败: {e}")
            
            # 如果上面失败，尝试简单映射
            if 'spatial_attention' not in batch_results:
                print("尝试简单的1D到2D映射...")
                side_len = int(np.sqrt(N_regions))
                if side_len * side_len == N_regions:
                    spatial_attn = region_importance.reshape(side_len, side_len)
                    original_h, original_w = video_tensor.shape[-2:]
                    frame_attn_resized = torch.nn.functional.interpolate(
                        spatial_attn.unsqueeze(0).unsqueeze(0), 
                        size=(original_h, original_w), 
                        mode='bilinear', 
                        align_corners=False
                    ).squeeze()
                    batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
                    print(f"Simple mapping成功，形状: {frame_attn_resized.shape}")
                else:
                    print(f"无法进行简单映射：{N_regions} 不是完全平方数")
                    
                    # 最后尝试：直接使用attention的平均值
                    if len(batch_attn.shape) == 2 and batch_attn.shape[1] > 1:
                        # 如果每个region内有多个token，可能可以重塑
                        region_tokens = batch_attn.shape[1]  # S-1
                        token_side = int(np.sqrt(region_tokens))
                        if token_side * token_side == region_tokens:
                            # 将每个region内的token重塑为空间，然后平均
                            spatial_per_region = batch_attn.reshape(N_regions, token_side, token_side)
                            # 平均所有region的空间模式
                            avg_spatial = spatial_per_region.mean(dim=0)  # (token_side, token_side)
                            
                            original_h, original_w = video_tensor.shape[-2:]
                            frame_attn_resized = torch.nn.functional.interpolate(
                                avg_spatial.unsqueeze(0).unsqueeze(0), 
                                size=(original_h, original_w), 
                                mode='bilinear', 
                                align_corners=False
                            ).squeeze()
                            batch_results['spatial_attention'] = frame_attn_resized.cpu().numpy()
                            print(f"Alternative mapping成功，形状: {frame_attn_resized.shape}")
            
            results[f'batch_{batch_idx}'] = batch_results
        
        return results

    def plot_attention_heatmap(self, 
                             original_frame: np.ndarray, 
                             attention_map: np.ndarray, 
                             title: str = "LG Attention Heatmap",
                             save_path: Optional[str] = None) -> plt.Figure:
        """绘制注意力热力图"""
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

    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_weights.clear()

# 兼容性：保持原有的类名
AttentionVisualizer = LGAttentionVisualizer

# 从layers.py导入必要的类
try:
    from ..models.layers import Block, LGBlock
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    sys.path.append('/home/qzk/Facial-Foundation-Model/src/models')
    try:
        from layers import Block, LGBlock
    except ImportError:
        print("警告: 无法导入 Block 和 LGBlock 类，可能需要手动导入")
