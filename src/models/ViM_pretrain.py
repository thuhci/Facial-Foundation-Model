# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
import numpy as np
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from src.models.layers import get_sinusoid_encoding_table

class BlockViM(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block_ViM(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = BlockViM(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbedViM(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(self, output_dim=768, embed_dim=768, 
                 norm_layer=nn.LayerNorm, clip_norm_type='l2'):
        super().__init__()
        self.clip_norm_type = clip_norm_type
        print(f'Normalization Type: {clip_norm_type}')

        self.head = nn.Linear(embed_dim, output_dim)
        self.norm = norm_layer(output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))

        if self.clip_norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.clip_norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        return x
    

class VisionMamba(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            drop_path_rate=0.,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1, 
            num_frames=8, 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
            # clip,
            clip_decoder_embed_dim=768,
            clip_output_dim=512,
            clip_norm_type='l2',
            clip_return_layer=1,
            clip_student_return_interval=1,
            # pixel decoder
            pixel_reconstruction=True,  # 是否启用像素重构
            clip_reconstruction=False,   # 是否启用CLIP重构
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        print(f'Student return index: {self.return_index}')
        self.depth = depth

        # pretrain parameters
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbedViM(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block_ViM(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        
        # output head
        self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)

        # CLIP decoder
        if clip_reconstruction:
            self.clip_decoder = nn.ModuleList([
                Linear_Decoder(
                    output_dim=clip_output_dim, 
                    embed_dim=clip_decoder_embed_dim, 
                    norm_layer=nn.LayerNorm, 
                    clip_norm_type=clip_norm_type
                ) for _ in range(clip_return_layer)
            ])

            self.clip_pos_embed = get_sinusoid_encoding_table(
                num_patches * num_frames // kernel_size + 1, 
                clip_decoder_embed_dim
            )
            print("Clip position embedding shape:", self.clip_pos_embed.shape)

        # 像素解码器
        if pixel_reconstruction:
            self.pixel_decoder = PixelDecoder(
                embed_dim=embed_dim, 
                decoder_embed_dim=clip_decoder_embed_dim, 
                decoder_depth=clip_return_layer, 
                decoder_num_heads=8, # 这里可以适当减少head数
                patch_size=patch_size, 
                tubelet_size=kernel_size
            )
            
            self.decoder_pos_embed = get_sinusoid_encoding_table(
                num_patches * num_frames // kernel_size + 1, 
                clip_decoder_embed_dim
            )
            print("Decoder position embedding shape:", self.decoder_pos_embed.shape)

        # original init
        self.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, mask=None):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        # print("shape after patch embedding:", x.shape)

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # print("shape after temporal pos embedding:", x.shape)

        # mask
        if mask.shape[1] == x.shape[1]-1:
            mask = torch.cat((torch.ones(B, 1).to(torch.bool).to(mask.device), mask), dim=1)  # add cls token mask
        x_vis = x[~mask].reshape(B, -1, C) # ~mask means visible
        x_clip_vis = []

        # mamba impl
        residual = None
        hidden_states = x_vis
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=None
                )
            if (idx - 1) in self.return_index:
                x_clip_vis.append(self.norm(residual.to(dtype=self.norm.weight.dtype))) # share norm for mask

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                eps=self.norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if (self.depth - 1) in self.return_index:
            x_clip_vis.append(residual)
        x_clip_vis = torch.stack(x_clip_vis)
        # print("shape after mamba layers:", x_clip_vis.shape)

        return x_clip_vis

    # def forward(self, x, mask=None):
    #     x_clip_vis = self.forward_features(x, mask)
        
    #     # align CLIP
    #     K, B, _, C_CLIP = x_clip_vis.shape
    #     expand_clip_pos_embed = self.clip_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
    #     if mask.shape[1] == expand_clip_pos_embed.shape[1]-1:
    #         mask = torch.cat((torch.ones(B, 1).to(torch.bool).to(mask.device), mask), dim=1)
    #     # print(f'Clip position embedding shape: {expand_clip_pos_embed.shape}')
    #     # print(f'Clip visible shape: {x_clip_vis.shape}')
    #     # print(f'Clip visible mask shape: {mask.shape}')
    #     clip_pos_emd_vis = expand_clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat(K, 1, 1, 1)
    #     x_clip_full = x_clip_vis + clip_pos_emd_vis # [K, B, N, C_d_clip]

    #     x_clip = []
    #     for idx, clip_decoder in enumerate(self.clip_decoder):
    #         x_clip.append(clip_decoder(x_clip_full[idx]))
    #     x_clip = torch.stack(x_clip) # align and normalize

    #     return x_clip
    
    def forward(self, x, mask=None):
        x_clip_vis = self.forward_features(x, mask)
        
        # 2. CLIP解码器处理
        K, B, N_vis, C_CLIP = x_clip_vis.shape
        
        last_features = x_clip_vis[-1]  # [B, N_vis, D]
        expand_decoder_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
        # print(f'Expand decoder position embedding shape: {expand_decoder_pos_embed.shape}')
        if mask.shape[1] == expand_decoder_pos_embed.shape[1]-1:
            mask = torch.cat((torch.ones(B, 1).to(torch.bool).to(mask.device), mask), dim=1)
        # decoder_pos_emd_vis = expand_decoder_pos_embed[~mask]
        # print(f'Decoder position embedding visible shape: {decoder_pos_emd_vis.shape}')
        # decoder_pos_emd_vis = decoder_pos_emd_vis.view(B, -1, expand_decoder_pos_embed.shape[-1])
        # print(f'Decoder position embedding reshaped: {decoder_pos_emd_vis.shape}')
        # decoder_pos_emd_vis = decoder_pos_emd_vis.unsqueeze(0).repeat(K, 1, 1, 1)
        # print(f'Decoder position embedding shape: {decoder_pos_emd_vis.shape}')

        pixel_reconstruction = self.pixel_decoder(last_features, mask, expand_decoder_pos_embed)
        
        return pixel_reconstruction[:,1:,:] # discard cls token reconstruction
    
    def forward_with_pixel_reconstruction(self, x, mask):
        """同时进行像素重构和CLIP特征提取"""
        # 1. 获取CLIP特征（复用现有的forward_features）
        x_clip_vis = self.forward_features(x, mask)
        
        # 2. CLIP解码器处理
        K, B, N_vis, C_CLIP = x_clip_vis.shape
        
        clip_output = []
        if hasattr(self, 'clip_decoder') and hasattr(self, 'clip_pos_embed'):
            # 处理位置编码
            expand_clip_pos_embed = self.clip_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
            if mask.shape[1] == expand_clip_pos_embed.shape[1]-1:
                mask = torch.cat((torch.ones(B, 1).to(torch.bool).to(mask.device), mask), dim=1)
            
            clip_pos_emd_vis = expand_clip_pos_embed[~mask].view(B, -1, expand_clip_pos_embed.shape[-1])
            
            clip_pos_emd_vis = clip_pos_emd_vis.unsqueeze(0).repeat(K, 1, 1, 1)
            print(f'Clip position embedding shape: {clip_pos_emd_vis.shape}')
            x_clip_full = x_clip_vis + clip_pos_emd_vis

            for idx, clip_decoder in enumerate(self.clip_decoder):
                clip_output.append(clip_decoder(x_clip_full[idx]))
            clip_output = torch.stack(clip_output)
        else:
            clip_output = None
        
        # 3. 像素重构
        if hasattr(self, 'pixel_decoder'):
            # 使用最后一层的特征进行像素重构
            # print("X_clip_vis shape:", x_clip_vis.shape)
            # print(f'Last features shape: {x_clip_vis[-1].shape}')
            last_features = x_clip_vis[-1]  # [B, N_vis, D]
            expand_decoder_pos_embed = self.decoder_pos_embed.repeat(B, 1, 1).type_as(x).to(x.device).clone().detach()
            # print(f'Expand decoder position embedding shape: {expand_decoder_pos_embed.shape}')
            if mask.shape[1] == expand_decoder_pos_embed.shape[1]-1:
                mask = torch.cat((torch.ones(B, 1).to(torch.bool).to(mask.device), mask), dim=1)
            # decoder_pos_emd_vis = expand_decoder_pos_embed[~mask]
            # print(f'Decoder position embedding visible shape: {decoder_pos_emd_vis.shape}')
            # decoder_pos_emd_vis = decoder_pos_emd_vis.view(B, -1, expand_decoder_pos_embed.shape[-1])
            # print(f'Decoder position embedding reshaped: {decoder_pos_emd_vis.shape}')
            # decoder_pos_emd_vis = decoder_pos_emd_vis.unsqueeze(0).repeat(K, 1, 1, 1)
            # print(f'Decoder position embedding shape: {decoder_pos_emd_vis.shape}')

            pixel_reconstruction = self.pixel_decoder(last_features, mask, expand_decoder_pos_embed)
        else:
            pixel_reconstruction = None
        
        return pixel_reconstruction, clip_output




class PixelDecoder(nn.Module):
    """像素级解码器，用于重构被mask的patches"""
    def __init__(self, embed_dim=768, decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, patch_size=16, tubelet_size=2):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            create_block_ViM(
                decoder_embed_dim,
                ssm_cfg=None,
                norm_epsilon=1e-5,
                drop_path=0.,
                rms_norm=True,
                residual_in_fp32=True,
                fused_add_norm=True,
                layer_idx=i,
                bimamba=True,
                device=None,
                dtype=None,
            ) for i in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 
                                     patch_size * patch_size * tubelet_size * 3, bias=True)
        
    def forward(self, x, mask, pos_embed):
        # x: visible tokens [B, N_vis, D], mask: bool mask [B, N], pos_embed: position embedding [1, N, D]
        B, N_vis, D = x.shape
        N = mask.shape[1]  # 总patch数量
        
        # 解码器embedding
        x = self.decoder_embed(x)  # [B, N_vis, decoder_dim]
        decoder_dim = x.shape[-1]
        
        # 创建mask tokens
        N_mask = N - N_vis  # mask的token数量
        mask_tokens = self.mask_token.repeat(B, N_mask, 1)  # [B, N_mask, decoder_dim]
        mask_tokens = mask_tokens.to(x.device, dtype=x.dtype)  # 确保与x同设备和dtype
        
        # 重建完整序列
        x_full = torch.zeros(B, N, decoder_dim, device=x.device, dtype=x.dtype)
        x_full[~mask] = x.reshape(-1, decoder_dim)  # 将visible tokens放到对应位置
        x_full[mask] = mask_tokens.reshape(-1, decoder_dim)  # 将mask tokens放到对应位置
        
        # 添加位置编码 (需要先投影到decoder维度)
        pos_embed_decoder = pos_embed
        # print(f'Position embedding shape: {pos_embed_decoder.shape}')
        # print(f'Input x_full shape: {x_full.shape}')
        x_full = x_full + pos_embed_decoder
        
        # 解码器处理
        residual = None
        for blk in self.decoder_blocks:
            x_full, residual = blk(x_full, residual)
        
        # 最终norm
        if not blk.fused_add_norm:
            if residual is None:
                residual = x_full
            else:
                residual = residual + x_full
            x_full = self.decoder_norm(residual.to(dtype=self.decoder_norm.weight.dtype))
        else:
            from mamba_ssm.ops.triton.layernorm import layer_norm_fn
            x_full = layer_norm_fn(
                x_full,
                self.decoder_norm.weight,
                self.decoder_norm.bias,
                eps=self.decoder_norm.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=False,
            )
        # print("Shape after decoder blocks:", x_full.shape)
        
        # 预测像素
        x_rec = self.decoder_pred(x_full)  # [B, N, patch_size^2 * tubelet_size * 3]
        # print("Shape after decoder prediction:", x_rec.shape)
        
        # 只返回mask部分的重构
        return x_rec[mask].reshape(B, -1, x_rec.shape[-1])



@register_model
def videomamba_middle_pretrain(pretrained=False, **kwargs):
    model = VisionMamba(
        patch_size=16, 
        embed_dim=768, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def pretrain_videomamba_base_dim512_patch16_160(pretrained=False, **kwargs):
    model = VisionMamba(
        img_size=160,
        patch_size=16,
        embed_dim=512,
        # num_heads=8,
        # decoder_embed_dim=768,
        num_frames = 16, 
        depth=32, 
        kernel_size=2,  
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        clip_reconstruction=False,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


if __name__ == '__main__':
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224
    
    model = videomamba_middle_pretrain(num_frames=num_frames).cuda()
    mask = torch.cat([
        torch.ones(1, 8 * int(14 * 14 * 0.75)),
        torch.zeros(1, 8 * int(14 * 14 * 0.25)),
    ], dim=-1).to(torch.bool)
    dummy_input = torch.rand(1, 3, num_frames, img_size, img_size).cuda()
    mask_input = mask.cuda()
    print(model(dummy_input, mask_input).shape)
    
    if hasattr(model, 'forward_with_pixel_reconstruction'):
        print("Testing pixel reconstruction...")
        pixel_rec, clip_feat = model.forward_with_pixel_reconstruction(dummy_input, mask_input)
        if pixel_rec is not None:
            print(f"Pixel reconstruction shape: {pixel_rec.shape}")
        if clip_feat is not None:
            print(f"CLIP features shape: {clip_feat.shape}")
            
        # calculate dummy loss 
        
        print("✓ Pixel reconstruction test successful!")