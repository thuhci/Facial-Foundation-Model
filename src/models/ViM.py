# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Vision Mamba (ViM) for finetuning tasks.
Adapted from ViM_pretrain.py, removing decoder and adding classification head.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import Tensor
from typing import Optional

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _load_weights

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from src.models.ViM_pretrain import BlockViM, create_block_ViM, PatchEmbedViM, segm_init_weights
from src.models.layers import get_sinusoid_encoding_table



class VisionMambaFinetune(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_path_rate=0.,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            kernel_size=1, 
            num_frames=8, 
            tubelet_size=2,
            device=None,
            dtype=None,
            use_checkpoint=False,
            checkpoint_num=0,
            use_learnable_pos_emb=False,
            use_mean_pooling=True,
            keep_temporal_dim=False,  # For frame-level prediction
        ):
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        super().__init__()
        
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_frames = num_frames    
        self.img_size = img_size    
        # print("[DEBUG ] keep_temporal_dim:", keep_temporal_dim)
        self.keep_temporal_dim = keep_temporal_dim
        self.use_mean_pooling = use_mean_pooling
        
        self.return_index = []
        for i in range(1):
            self.return_index.append(depth - int(i * 1) - 1)
        
        
        # Patch embedding
        self.patch_embed = PatchEmbedViM(
            img_size=img_size, patch_size=patch_size, in_chans=channels, 
            embed_dim=embed_dim, kernel_size=tubelet_size
        )
        num_patches = self.patch_embed.num_patches
        
        # Position embedding
        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
        else:
            # Sinusoidal position embedding
            self.pos_embed = get_sinusoid_encoding_table(num_patches + 1, embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // kernel_size, embed_dim))
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        
        # Mamba blocks
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
        
        self.depth = depth
        
        # Final layer norm
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        
        # Classification head
        self.norm = nn.Identity() if use_mean_pooling else self.norm_f
        # self.norm = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.fc_norm = self.norm_f if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix="", ignore_missing="relative_position_index"):
        _load_weights(self, checkpoint_path, prefix, ignore_missing)

    def forward_features(self, x, mask=None):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        self.pos_embed = self.pos_embed.to(x.device, dtype=x.dtype)  # ensure pos_embed is on the same device and dtype
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

        x_vis = x.reshape(B, -1, C) # ~mask means visible
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
                x_clip_vis.append(self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))) # share norm for mask

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        
        if (self.depth - 1) in self.return_index:
            x_clip_vis.append(residual)
        x_clip_vis = torch.stack(x_clip_vis)
        # print("shape after mamba layers:", x_clip_vis.shape)

        return x_clip_vis

    def forward(self, x):
        # x: (B, C, T, H, W)
        x = self.forward_features(x)  # (B, num_patches, embed_dim)
        # print("[DEBUG] x shape after forward_features:", x.shape)
        x = x.squeeze(0)  # remove depth dim
        x = x[:, 1:, ...] # remove cls token
        # Global pooling
        if self.fc_norm is not None:
            if self.keep_temporal_dim:
                # Keep temporal dimension: (1, B, T, embed_dim)

                B, N, C = x.shape
                T = self.num_frames // self.patch_embed.tubelet_size   # temporal patches
                H = self.patch_embed.img_size[0] // self.patch_embed.patch_size[0]  # height patches
                W = self.patch_embed.img_size[1] // self.patch_embed.patch_size[1]  # width patches

                # Reshape to (B, T, H*W, C) and pool over spatial dimensions
                x = x.view(B, T, H*W, C)
                # print("[DEBUG] x shape before spatial pooling:", x.shape)
                x = x.mean(dim=2)  # (B, T, C)
                
                x = rearrange(x, 'b t c -> b c t')
                x = torch.nn.functional.interpolate(
                    x, scale_factor=self.patch_embed.tubelet_size,
                    mode='linear'
                )
                x = rearrange(x, 'b c t -> b t c')
                
                # print("[DEBUG] x shape after spatial pooling:", x.shape)
                x = self.fc_norm(x)
            else:
                # print("[DEBUG] FC_NORM shape of x before global pooling:", x.shape)
                # Standard global pooling: (1, B, T, embed_dim)
                x = x.mean(dim=1)  # (B, embed_dim)
                x = self.fc_norm(x)
        else:
            # print("[DEBUG] NO FC_NORM shape of x before global pooling:", x.shape)
            x = self.norm(x)
            if not self.keep_temporal_dim:
                x = x.mean(dim=1)  # (B, embed_dim)
        
        # print("[DEBUG] x shape before classification head:", x.shape)
        # Classification
        x = self.head(x)
        return x


@register_model
def videomamba_tiny(pretrained=False, **kwargs):
    model = VisionMambaFinetune(
        embed_dim=192, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        # TODO: add pretrained weights loading
        pass
    return model


@register_model  
def videomamba_small(pretrained=False, **kwargs):
    model = VisionMambaFinetune(
        embed_dim=384, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        # TODO: add pretrained weights loading
        pass
    return model


@register_model
def videomamba_base(pretrained=False, **kwargs):
    model = VisionMambaFinetune(
        embed_dim=768, depth=24, rms_norm=True, residual_in_fp32=True, fused_add_norm=True, **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        # TODO: add pretrained weights loading  
        pass
    return model


@register_model
def videomamba_middle_finetune(pretrained=False, **kwargs):
    """VideoMamba middle size for finetuning (same as pretrain but with classification head)"""
    model = VisionMambaFinetune(
        img_size=160,
        patch_size=16,
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        # TODO: add pretrained weights loading
        pass
    return model


@register_model
def videomamba_base_dim512_patch16_160(pretrained=False, **kwargs):
    model = VisionMambaFinetune(
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
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }
