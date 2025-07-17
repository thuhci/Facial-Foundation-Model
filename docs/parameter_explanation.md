# 训练参数详解

本文档详细解释了训练过程中各种参数的含义，帮助理解和配置训练过程。

## 数据增强参数 (Augmentation Parameters)

### Mixup 参数
- **用途**: 提高模型泛化能力，减少过拟合
- **原理**: 将两个训练样本按权重混合，标签也按相同权重混合
- **参数说明**:
  - `mixup`: Mixup的强度，0表示不使用，通常设为0.8
  - `cutmix`: CutMix的强度，类似Mixup但是在空间上混合
  - `mixup_prob`: 执行mixup的概率
  - `mixup_switch_prob`: 在mixup和cutmix间切换的概率
  - `mixup_mode`: 混合模式，'batch'表示批次内混合

### Random Erase 参数
- **用途**: 随机擦除图像的部分区域，提高模型鲁棒性
- **原理**: 在训练图像上随机选择矩形区域并用随机像素值填充
- **参数说明**:
  - `reprob`: 随机擦除的概率，0.25表示25%的概率
  - `remode`: 擦除模式，'pixel'表示用随机像素值填充
  - `recount`: 每张图像擦除的区域数量
  - `resplit`: 是否在数据增强的第一次分割中跳过擦除

### Auto Augmentation 参数
- **用途**: 自动学习最佳的数据增强策略
- **原理**: 使用预定义的增强策略组合，自动选择最优的增强方法
- **参数说明**:
  - `aa`: 自动增强策略字符串，如'rand-m7-n4-mstd0.5-inc1'
    - `m7`: magnitude=7，增强强度
    - `n4`: num_layers=4，每张图像应用的增强操作数
    - `mstd0.5`: magnitude_std=0.5，强度的标准差
    - `inc1`: 使用递增的增强策略

### 其他增强参数
- `color_jitter`: 颜色抖动强度，影响亮度、对比度、饱和度
- `train_interpolation`: 训练时的插值方法，'bicubic'表示双三次插值
- `smoothing`: 标签平滑参数，减少过拟合

## 优化参数 (Optimization Parameters)

### 学习率调度 (Learning Rate Scheduling)
- `lr`: 基础学习率
- `min_lr`: 最小学习率
- `warmup_lr`: 预热阶段的学习率
- `warmup_epochs`: 预热轮数
- `warmup_steps`: 预热步数（如果设置则覆盖warmup_epochs）

### 优化器参数
- `opt`: 优化器类型，'adamw'表示AdamW优化器
- `opt_eps`: 优化器的epsilon值，避免除零错误
- `opt_betas`: Adam优化器的beta参数，控制动量
- `weight_decay`: 权重衰减，L2正则化强度
- `momentum`: SGD优化器的动量参数

### 梯度处理
- `clip_grad`: 梯度裁剪阈值，防止梯度爆炸
- `layer_decay`: 层级学习率衰减，不同层使用不同学习率

## 模型参数 (Model Parameters)

### 基础模型参数
- `model`: 模型架构名称
- `input_size`: 输入图像尺寸
- `num_frames`: 视频帧数
- `sampling_rate`: 帧采样率
- `tubelet_size`: 时间管道大小

### 正则化参数
- `drop`: 通用dropout率
- `attn_drop_rate`: 注意力层dropout率
- `drop_path`: 路径dropout率，用于残差连接

### 注意力机制参数
- `attn_type`: 注意力类型，'local_global'表示局部-全局注意力
- `lg_region_size`: 局部-全局注意力的区域大小
- `lg_first_attn_type`: 第一层注意力类型
- `lg_third_attn_type`: 第三层注意力类型

## 训练参数 (Training Parameters)

### 基础训练参数
- `epochs`: 训练轮数
- `batch_size`: 批次大小
- `num_workers`: 数据加载工作进程数
- `pin_memory`: 是否将数据加载到锁页内存

### 模型EMA (Exponential Moving Average)
- `model_ema`: 是否使用模型EMA
- `model_ema_decay`: EMA衰减率
- `model_ema_force_cpu`: 是否强制在CPU上计算EMA

### 评估参数
- `disable_eval_during_finetuning`: 是否在微调期间禁用验证
- `dist_eval`: 是否使用分布式评估
- `val_metric`: 验证指标，用于选择最佳模型

## 系统参数 (System Parameters)

### 分布式训练
- `world_size`: 分布式训练的进程数
- `local_rank`: 本地进程排名
- `dist_url`: 分布式训练的URL

### 其他系统参数
- `device`: 训练设备，'cuda'或'cpu'
- `seed`: 随机种子
- `output_dir`: 输出目录
- `log_dir`: 日志目录

## 任务特定参数

### 凝视估计参数 (Gaze Estimation)
- `use_l2cs`: 是否使用L2CS方法
- `num_bins`: L2CS方法的分箱数量
- `alpha_reg`: L2CS中回归损失的权重
- `bin_width`: 每个分箱的宽度（度）

## 使用建议

### 对于分类任务
```yaml
augmentation:
  mixup: 0.8
  cutmix: 1.0
  random_erase_prob: 0.25
  label_smoothing: 0.1
```

### 对于回归任务（如凝视估计）
```yaml
augmentation:
  mixup: 0.0  # 禁用mixup
  cutmix: 0.0  # 禁用cutmix
  random_erase_prob: 0.1  # 降低随机擦除概率
  label_smoothing: 0.0  # 禁用标签平滑
```

### 对于小数据集
```yaml
augmentation:
  mixup: 0.4  # 降低mixup强度
  random_erase_prob: 0.1  # 降低随机擦除概率
optimization:
  weight_decay: 0.01  # 降低权重衰减
```

### 对于大数据集
```yaml
augmentation:
  mixup: 1.0  # 提高mixup强度
  random_erase_prob: 0.5  # 提高随机擦除概率
optimization:
  weight_decay: 0.1  # 提高权重衰减
```
