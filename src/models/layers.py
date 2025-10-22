from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange, repeat
from src.utils.config import get_cfg

USE_LORA = [True]*16
SCALE = [16]*16
RANK = [8]*16

class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, scale=16, blk_id=None, attn_type=None, layer_type=None):
        super().__init__()
        self.rank = rank
        self.scale = scale
        # print(in_features, out_features, rank, alpha)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        self.load_lora_weights(blk_id=blk_id, attn_type=attn_type, layer_type=layer_type, in_features=in_features, out_features=out_features, rank=rank)
        # self.lora_dropout = nn.Dropout(p=0.02)
        # nn.init.kaiming_uniform_(self.lora_A, a=0.0)
        # nn.init.zeros_(self.lora_B)

    def load_lora_weights(self, blk_id, attn_type, layer_type, in_features, out_features, rank):
        lora_weights_pth_path = "/root/lfz/Facial-Foundation-Model/output/lora/gaze_d16/checkpoint-best.pth"
        if blk_id < 4:
            # 从 .pth 文件加载权重
            print(f"Loading LoRA weights for block {blk_id}, attention {attn_type}, layer {layer_type} from {lora_weights_pth_path}")
            # 加载 .pth 文件
            state_dict = torch.load(lora_weights_pth_path, map_location='cpu')
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            # 构建键名
            key_A = f"module.blocks.{blk_id}.{attn_type}.{layer_type}.lora_A"
            key_B = f"module.blocks.{blk_id}.{attn_type}.{layer_type}.lora_B"
            # print(f"Looking for keys: {key_A}, {key_B}")

            if key_A in state_dict and key_B in state_dict:
                loaded_A = state_dict[key_A]
                loaded_B = state_dict[key_B]

                # 检查加载的张量形状是否匹配
                loaded_A_shape = loaded_A.shape
                loaded_B_shape = loaded_B.shape
                expected_A_shape = torch.Size([rank, in_features])
                expected_B_shape = torch.Size([out_features, rank])

                if loaded_A_shape != expected_A_shape or loaded_B_shape != expected_B_shape:
                    raise ValueError(
                        f"Shape mismatch for block {blk_id}, attention {attn_type}, layer {layer_type}. "
                        f"Expected A: {expected_A_shape}, B: {expected_B_shape}, "
                        f"Got A: {loaded_A_shape}, B: {loaded_B_shape}"
                    )

                # 使用加载的张量初始化参数
                self.lora_A = nn.Parameter(loaded_A.clone())
                self.lora_B = nn.Parameter(loaded_B.clone())
                print(f"Successfully loaded LoRA weights for block {blk_id}, attention {attn_type}, layer {layer_type}")
                # 注意：这里没有从 .pth 文件加载 scale，因为 .pth 文件存储的是 state_dict，
                # 而 scale 通常不作为参数保存在 state_dict 中，而是作为模块属性。
                # 如果你的 .pth 文件中包含 scale 信息，需要相应调整加载逻辑。
            else:
                # raise KeyError(f"LoRA weights not found for block {blk_id}, attention {attn_type}, layer {layer_type} in {lora_weights_pth_path}")
                nn.init.kaiming_uniform_(self.lora_A, a=0.0)
                nn.init.zeros_(self.lora_B)
                print(f"Use randomly initialized LoRA weights for block {blk_id}, attention {attn_type}, layer {layer_type}")

    def forward(self, x, original_weight):
        if self.scale == 100:
            lora_weight = (self.lora_B @ self.lora_A)
            return F.linear(x, lora_weight)
        else:
            lora_weight = (self.lora_B @ self.lora_A) * self.scale
            return F.linear(x, original_weight + lora_weight)
        # return F.linear(x, lora_weight)

    def get_lora_weights(self):
        return {
            'lora_A': self.lora_A.detach().cpu().clone(),
            'lora_B': self.lora_B.detach().cpu().clone(),
            'scale': self.scale
        }

    def is_lora_layer(self): # 添加标识方法
        return True

class DoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=1.0):
        super().__init__()
        self.rank = 8
        self.alpha = alpha
        # 低秩矩阵 A 和 B，与 LoRA 一致
        self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        # 初始化方式与原 LoRA 保持一致
        nn.init.kaiming_uniform_(self.lora_A, a=0.0)
        nn.init.zeros_(self.lora_B)
        # 幅度参数 m，初始化为全 1，形状为 (out_features, 1)
        self.m = nn.Parameter(torch.ones(out_features, 1))
        # 原 LoRA 的 scale 参数在 DoRA 中不再需要，改为由 m 控制

    def orthogonalize(self):
        # 计算方向矩阵 V = lora_B @ lora_A
        V = self.lora_B @ self.lora_A
        # 对 V 进行标准化（简化的正交化处理）
        V_norm = F.normalize(V, dim=1)  # 按行标准化，确保方向单位化
        return V_norm

    def forward(self, x, original_weight):
        # 获取正交化的方向矩阵 V
        V = self.orthogonalize()
        # 计算权重更新 ΔW = m * V
        lora_weight = self.m * V
        # 应用到原始权重
        return F.linear(x, original_weight + lora_weight)


class BottleneckAdapter(nn.Module):
    def __init__(self, dim=1024, hidden_dim=128, drop=0.):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: (B*N, S, C)
        residual = x
        x = self.linear1(x)  # (B*N, S, hidden_dim)
        x = self.ln(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.linear2(x)  # (B*N, S, C)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement 
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_lora=False, scale=16, rank=8, blk_id=0, attn_type=None):
        super().__init__()
        cfg  = get_cfg()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_lora = use_lora    
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if cfg.TRAINING.USE_LORA and use_lora:
            self.lora_qkv = LoRALayer(in_features=dim, out_features=all_head_dim * 3, rank=rank, scale=scale, blk_id=blk_id, attn_type=attn_type, layer_type='lora_qkv')
            # self.lora_proj = LoRALayer(in_features=all_head_dim, out_features=dim, rank=8, scale=4)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        cfg  = get_cfg()
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if cfg.TRAINING.USE_LORA and self.use_lora:
            qkv = self.lora_qkv(x, self.qkv.weight)
            if qkv_bias is not None:
                qkv += qkv_bias
            print("use Lora in Attention")
        else:
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # me: support window mask
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        # if cfg.TRAINING.USE_LORA:
        #     x = self.lora_proj(x, self.proj.weight)
        # else:
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None, use_lora=False, blk_id=0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, use_lora=USE_LORA[blk_id], scale=SCALE[blk_id], rank=RANK[blk_id], blk_id=blk_id, attn_type='self_attn')
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x), mask=mask))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x



"""
adapted from https://github.com/lucidrains/perceiver-pytorch/blob/main/perceiver_pytorch/perceiver_pytorch.py
"""
# support cross attention
class GeneralAttention(nn.Module):
    def __init__(
            self, dim, context_dim=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, use_lora=False, rank=8, scale=16, blk_id=0, attn_type=None):
        super().__init__()
        cfg  = get_cfg()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.use_lora = use_lora
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.kv = nn.Linear(dim if context_dim is None else context_dim, all_head_dim * 2, bias=False)
        if cfg.TRAINING.USE_LORA and use_lora:
            self.lora_q = LoRALayer(in_features=dim, out_features=all_head_dim, rank=rank, scale=scale, blk_id=blk_id, attn_type=attn_type, layer_type='lora_q')
            self.lora_kv = LoRALayer(in_features=dim if context_dim is None else context_dim, out_features=all_head_dim * 2, rank=rank, scale=scale, blk_id=blk_id, attn_type=attn_type, layer_type='lora_kv')
            # self.lora_proj = LoRALayer(in_features=all_head_dim, out_features=dim, rank=8, scale=4)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context=None):
        cfg  = get_cfg()
        B, T1, C = x.shape
        q_bias, kv_bias = self.q_bias, None
        if self.q_bias is not None:
            kv_bias = torch.cat((torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        
        if cfg.TRAINING.USE_LORA and self.use_lora:
            q = self.lora_q(x, self.q.weight)
            if q_bias is not None:
                q = q + q_bias
            kv = self.lora_kv(x if context is None else context, self.kv.weight)   
            # print("use Lora in GeneralAttention")
            # kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)

            if kv_bias is not None:
                kv = kv + kv_bias
        else:
            q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
            kv = F.linear(input=x if context is None else context, weight=self.kv.weight, bias=kv_bias)

        
        q = q.reshape(B, T1, self.num_heads, -1).transpose(1,2) # me: (B, H, T1, C//H)
        _, T2, _ = kv.shape
        kv = kv.reshape(B, T2, 2, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple), me： (B, H, T2, C//H)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # me: (B, H, T1, T2)

        attn = attn.softmax(dim=-1) # (B, H, T1, T2)
        attn = self.attn_drop(attn) # (B, H, T1, T2)

        x = (attn @ v).transpose(1, 2).reshape(B, T1, -1) # (B, H, T1, C//H) -> (B, T1, H, C//H) -> (B, T1, C)
        # if cfg.TRAINING.USE_LORA:
        #     x = self.lora_proj(x, self.proj.weight)
        # else:
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn 



# local + global
class LGBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attn_head_dim=None,
                 # new added
                 first_attn_type='self', third_attn_type='cross',
                 attn_param_sharing_first_third=False, attn_param_sharing_all=False,
                 no_second=False, no_third=False, blk_id=0
                 ):

        super().__init__()

        assert first_attn_type in ['self', 'cross'], f"Error: invalid attention type '{first_attn_type}', expected 'self' or 'cross'!"
        assert third_attn_type in ['self', 'cross'], f"Error: invalid attention type '{third_attn_type}', expected 'self' or 'cross'!"
        self.first_attn_type = first_attn_type
        self.third_attn_type = third_attn_type
        self.attn_param_sharing_first_third = attn_param_sharing_first_third
        self.attn_param_sharing_all = attn_param_sharing_all
        self.blk_id = blk_id

        # Attention layer
        ## perform local (intra-region) attention, update messenger tokens
        ## (local->messenger) or (local<->local, local<->messenger)
        self.first_attn_norm0 = norm_layer(dim)
        if self.first_attn_type == 'cross':
            self.first_attn_norm1 = norm_layer(dim)
        self.first_attn = GeneralAttention(
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, use_lora=USE_LORA[self.blk_id], scale=SCALE[self.blk_id], rank=RANK[self.blk_id], attn_type='first_attn', blk_id=self.blk_id)
        # self.adapter = BottleneckAdapter(dim=dim, hidden_dim=64, drop=drop)
        ## perform global (inter-region) attention on messenger tokens
        ## (messenger<->messenger)
        self.no_second = no_second
        if not no_second:
            self.second_attn_norm0 = norm_layer(dim)
            if attn_param_sharing_all:
                self.second_attn = self.first_attn
            else:
                self.second_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, use_lora=USE_LORA[self.blk_id], scale=SCALE[self.blk_id], rank=RANK[self.blk_id], attn_type='second_attn', blk_id=self.blk_id)

        ## perform local (intra-region) attention to inject global information into local tokens
        ## (messenger->local) or (local<->local, local<->messenger)
        self.no_third = no_third
        if not no_third:
            self.third_attn_norm0 = norm_layer(dim)
            if self.third_attn_type == 'cross':
                self.third_attn_norm1 = norm_layer(dim)
            if attn_param_sharing_first_third or attn_param_sharing_all:
                self.third_attn = self.first_attn
            else:
                self.third_attn = GeneralAttention(
                    dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim, use_lora=USE_LORA[self.blk_id], scale=SCALE[self.blk_id], rank=RANK[self.blk_id], attn_type='third_attn', blk_id=self.blk_id)

        # FFN layer
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if init_values > 0:
            self.gamma_0 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        else:
            self.gamma_0, self.gamma_1, self.gamma_2 = None, None, None


    def forward(self, x, b):
        """
        :param x: (B*N, S, C),
            B: batch size
            N: number of local regions
            S: 1 + region size, 1: attached messenger token for each local region
            C: feature dim
        param b: batch size
        :return: (B*N, S, C),
        """
        bn = x.shape[0]
        n = bn // b # number of local regions
        attns = []
        if self.gamma_1 is None:
            # Attention layer
            ## perform local (intra-region) self-attention
            if self.first_attn_type == 'self':
                y, attn = self.first_attn(self.first_attn_norm0(x))
                attns.append(attn)
                x = x + self.drop_path(y)
            else: # 'cross'
                x[:,:1] = x[:,:1] + self.drop_path(
                    self.first_attn(
                        self.first_attn_norm0(x[:,:1]), # (b*n, 1, c)
                        context=self.first_attn_norm1(x[:,1:]) # (b*n, s-1, c)
                    )
                )

            ## perform global (inter-region) self-attention
            if not self.no_second:
                # messenger_tokens: representative tokens
                # .clone(): fix in-place error in higher pytorch version, please refer to https://github.com/sunlicai/MAE-DFER/issues/3#issuecomment-1809834219
                messenger_tokens = rearrange(x[:,0].clone(), '(b n) c -> b n c', b=b) # attn on 'n' dim
                y, attn = self.second_attn(self.second_attn_norm0(messenger_tokens))  # (B, N, C), (B, num_heads, N, N)
                messenger_tokens = messenger_tokens + self.drop_path(
                    y)
                attns.append(attn)
                x[:,0] = rearrange(messenger_tokens, 'b n c -> (b n) c')
            else: # for usage in the third attn
                # .clone(): fix in-place error in higher pytorch version, please refer to https://github.com/sunlicai/MAE-DFER/issues/3#issuecomment-1809834219
                messenger_tokens = rearrange(x[:,0].clone(), '(b n) c -> b n c', b=b) # attn on 'n' dim

            ## perform local-global interaction
            if not self.no_third:
                if self.third_attn_type == 'self':
                    y, attn = self.third_attn(self.third_attn_norm0(x))
                    attns.append(attn)
                    x = x + self.drop_path(y)
                else:
                    # .clone(): fix in-place error in higher pytorch version, please refer to https://github.com/sunlicai/MAE-DFER/issues/3#issuecomment-1809834219
                    local_tokens = rearrange(x[:,1:].clone(), '(b n) s c -> b (n s) c', b=b)# NOTE: n merges into s (not b), (B, N*(S-1), D)
                    y, attn = self.third_attn(
                        self.third_attn_norm0(local_tokens), # (b, n*(s-1), c)
                        context=self.third_attn_norm1(messenger_tokens) # (b, n*1, c)
                    )
                    attns.append(attn)
                    local_tokens = local_tokens + self.drop_path(y)
                    x[:,1:] = rearrange(local_tokens, 'b (n s) c -> (b n) s c', n=n)

            # FFN layer
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            # adapter_output = self.adapter(x)  # (B*N, S, C)
            # x = x + self.drop_path(                                                                                   _output)
        else:
            raise NotImplementedError
        return x, attns


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // self.tubelet_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # me: for more attention types
        self.temporal_seq_len = num_frames // self.tubelet_size
        self.spatial_num_patches = num_patches // self.temporal_seq_len
        self.input_token_size = (num_frames // self.tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim, 
                            kernel_size = (self.tubelet_size,  patch_size[0],patch_size[1]), 
                            stride=(self.tubelet_size,  patch_size[0],  patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # print("PatchEmbed: B, C, T, H, W:", B, C, T, H, W)
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
    
class EnhancedPatchEmbed(nn.Module):
    def __init__(self, img_size = 160, pad_ker_str_tem_chan = [[0,2,2,1,32],[0,2,2,1,64],[0,2,2,1,256],[0,2,2,2,768]], 
             in_chans=3, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.tubelet_size = int(tubelet_size)
        tem_size = 1
        img_sizes = [img_size]
        for pad_ker_str_tem in pad_ker_str_tem_chan:
            padding, kernel_size, stride, temporal_stride, chan = pad_ker_str_tem
            img_size = list(img_size)
            img_size[0] = (img_size[0] + 2 * padding - kernel_size) // stride + 1
            img_size[1] = (img_size[1] + 2 * padding - kernel_size) // stride + 1
            img_sizes.append(img_size)
            tem_size = tem_size * temporal_stride
        assert tem_size == self.tubelet_size, \
            f"Input temporal size ({tem_size}) doesn't match model ({self.tubelet_size})."
        self.img_sizes = img_sizes
        
        self.temporal_seq_len = num_frames // self.tubelet_size
        self.spatial_num_patches = img_sizes[-1][0]  * img_sizes[-1][1] 
        self.num_patches = self.spatial_num_patches * self.temporal_seq_len
        self.input_token_size = (self.temporal_seq_len, img_sizes[-1][0], img_sizes[-1][1])
        
        self.patch_size = (img_sizes[0][0]//img_sizes[-1][0], img_sizes[0][1]//img_sizes[-1][1])
        
        self.proj = nn.ModuleList()
        for i, (padding, kernel_size, stride, temporal_stride, now_embed_dim) in enumerate(pad_ker_str_tem_chan):
            self.proj.append(
                nn.Conv3d(in_channels=in_chans if i == 0 else pad_ker_str_tem_chan[i-1][-1], 
                          out_channels=embed_dim if i == len(pad_ker_str_tem_chan) - 1 else now_embed_dim, 
                          kernel_size=(temporal_stride, kernel_size, kernel_size), 
                          stride=(temporal_stride, stride, stride), 
                          padding=(0, padding, padding))
            )
            self.proj.append(
                nn.BatchNorm3d(embed_dim if i == len(pad_ker_str_tem_chan) - 1 else now_embed_dim)
            )
            self.proj.append(nn.ReLU(inplace=True))
            
    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_sizes[0][0] and W == self.img_sizes[0][1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_sizes[0][0]}*{self.img_sizes[0][1]})."
        for i in range(len(self.proj)):
            x = self.proj[i](x)
            # print(f"After layer {i+1}, x.shape: {x.shape}")
        x = x.flatten(2).transpose(1, 2)
        return x
        

# sin-cos position encoding
# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py#L31
def get_sinusoid_encoding_table(n_position, d_hid): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 

    return torch.FloatTensor(sinusoid_table).unsqueeze(0) 

