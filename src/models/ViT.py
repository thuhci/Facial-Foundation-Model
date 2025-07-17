from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 400, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }




class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, 
                 img_size=224, 
                 patch_size=16, 
                 in_chans=3, 
                 num_classes=1000, 
                 embed_dim=768, 
                 depth=12,
                 num_heads=12, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False, 
                 init_scale=0.,
                 all_frames=16,
                 tubelet_size=2,
                 use_mean_pooling=True,
                 keep_temporal_dim=False, # do not perform temporal pooling, has higher priority than 'use_mean_pooling'
                 head_activation_func=None, # activation function after head fc, mainly for the regression task
                 attn_type='joint',
                 lg_region_size=(2, 2, 10), lg_first_attn_type='self', lg_third_attn_type='cross',  # for local_global
                 lg_attn_param_sharing_first_third=False, lg_attn_param_sharing_all=False,
                 lg_classify_token_type='org', lg_no_second=False, lg_no_third=False,
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.tubelet_size = tubelet_size
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, num_frames=all_frames, tubelet_size=self.tubelet_size)
        num_patches = self.patch_embed.num_patches

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # me: support more attention types
        self.attn_type = attn_type
        if attn_type == 'joint':
            self.blocks = nn.ModuleList([
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values)
                for i in range(depth)])
        elif attn_type == 'local_global':
            print(f"==> Note: Use 'local_global' for compute reduction (lg_region_size={lg_region_size},"
                  f"lg_first_attn_type={lg_first_attn_type}, lg_third_attn_type={lg_third_attn_type},"
                  f"lg_attn_param_sharing_first_third={lg_attn_param_sharing_first_third},"
                  f"lg_attn_param_sharing_all={lg_attn_param_sharing_all},"
                  f"lg_classify_token_type={lg_classify_token_type},"
                  f"lg_no_second={lg_no_second}, lg_no_third={lg_no_third})")
            self.blocks = nn.ModuleList([
                LGBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    init_values=init_values,
                    first_attn_type=lg_first_attn_type, third_attn_type=lg_third_attn_type,
                    attn_param_sharing_first_third=lg_attn_param_sharing_first_third,
                    attn_param_sharing_all=lg_attn_param_sharing_all,
                    no_second=lg_no_second, no_third=lg_no_third,
                )
                for i in range(depth)])
            # region tokens
            self.lg_region_size = lg_region_size # (t, h, w)
            self.lg_num_region_size = list(i//j for i,j in zip(self.patch_embed.input_token_size, lg_region_size)) # (nt, nh, nw)
            num_regions = self.lg_num_region_size[0] * self.lg_num_region_size[1] * self.lg_num_region_size[2] # nt * nh * nw
            print(f"==> Number of local regions: {num_regions} (size={self.lg_num_region_size})")
            self.lg_region_tokens = nn.Parameter(torch.zeros(num_regions, embed_dim))
            trunc_normal_(self.lg_region_tokens, std=.02)

            # The token type used for final classification
            self.lg_classify_token_type = lg_classify_token_type
            assert lg_classify_token_type in ['org', 'region', 'all'], \
                f"Error: wrong 'lg_classify_token_type' in local_global attention ('{lg_classify_token_type}'), expected 'org'/'region'/'all'!"

        else:
            raise NotImplementedError

        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        # 原始的head保持不变，用于非L2CS模式
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # me: add frame-level prediction support
        self.keep_temporal_dim = keep_temporal_dim

        # me: add head activation function support for regression task
        if head_activation_func is not None:
            if head_activation_func == 'sigmoid':
                self.head_activation_func = nn.Sigmoid()
            elif head_activation_func == 'relu':
                self.head_activation_func = nn.ReLU()
            elif head_activation_func == 'tanh':
                self.head_activation_func = nn.Tanh()
            else:
                raise NotImplementedError
        else: # default
            self.head_activation_func = nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)

        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)

        self.head.weight.data.mul_(init_scale)
        self.head.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'part_tokens', 'lg_region_tokens'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, _, _ = x.size()

        if self.pos_embed is not None:
            # print("shape of pos_embed:", self.pos_embed.shape)
            # print("shape of x before pos_embed:", x.shape)
            x = x + self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        x = self.pos_drop(x)

        if self.attn_type == 'local_global':
            # input: region partition
            nt, t = self.lg_num_region_size[0], self.lg_region_size[0]
            nh, h = self.lg_num_region_size[1], self.lg_region_size[1]
            nw, w = self.lg_num_region_size[2], self.lg_region_size[2]
            b = x.size(0)
            x = rearrange(x, 'b (nt t nh h nw w) c -> b (nt nh nw) (t h w) c', nt=nt,nh=nh,nw=nw,t=t,h=h,w=w)
            # add region (i.e., representative) tokens
            region_tokens = repeat(self.lg_region_tokens, 'n c -> b n 1 c', b=b)
            x = torch.cat([region_tokens, x], dim=2) # (b, nt*nh*nw, 1+thw, c)
            x = rearrange(x, 'b n s c -> (b n) s c') # s = 1 + thw
            # run through each block
            for blk in self.blocks:
                x = blk(x, b) # (b*n, s, c)

            x = rearrange(x, '(b n) s c -> b n s c', b=b) # s = 1 + thw
            # token for final classification
            if self.lg_classify_token_type == 'region': # only use region tokens for classification
                x = x[:,:,0] # (b, n, c)
            elif self.lg_classify_token_type == 'org': # only use original tokens for classification
                x = rearrange(x[:,:,1:], 'b n s c -> b (n s) c') # s = thw
            else: # use all tokens for classification
                x = rearrange(x, 'b n s c -> b (n s) c') # s = 1 + thw

        else:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        if self.fc_norm is not None:
            # me: add frame-level prediction support
            if self.keep_temporal_dim:
                x = rearrange(x, 'b (t hw) c -> b c t hw',
                              t=self.patch_embed.temporal_seq_len,
                              hw=self.patch_embed.spatial_num_patches)
                # spatial mean pooling
                x = x.mean(-1) # (B, C, T)
                # temporal upsample: 8 -> 16, for patch embedding reduction
                x = torch.nn.functional.interpolate(
                    x, scale_factor=self.patch_embed.tubelet_size,
                    mode='linear'
                )
                x = rearrange(x, 'b c t -> b t c')
                return self.fc_norm(x)
            else:
                return self.fc_norm(x.mean(1))
        else:
            return x[:, 0]

    def forward(self, x, save_feature=False):
        x = self.forward_features(x)
        if save_feature:
            feature = x
        
        
        if isinstance(self.head, nn.ModuleDict):
            return {
                'pitch': self.head['pitch'](x),
                'yaw': self.head['yaw'](x)
            }
            
        x = self.head(x)
        # me: add head activation function support
        x = self.head_activation_func(x)
        # me: add frame-level prediction support
        if self.keep_temporal_dim:
            x = x.view(x.size(0), -1) # (B,T,C) -> (B,T*C)
        if save_feature:
            return x, feature
        else:
            return x




@register_model
def vit_base_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def vit_base_dim512_no_depth_patch16_160(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=160,
        patch_size=16, embed_dim=512, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

