# 训练参数详解

[本文档基本由 copilot 生成]
本文档详细解释了基于 YACS 配置系统的训练过程中各种参数的含义，帮助理解和配置训练过程。

## 模型配置 (MODEL)

### 基础模型参数
- `NAME`: 模型架构名称，默认为 'vit_base_patch16_224'
- `TUBELET_SIZE`: 时间管道大小，默认为 2
- `INPUT_SIZE`: 输入图像尺寸，默认为 224
- `DEPTH`: 模型深度，None 表示使用默认深度
- `USE_CHECKPOINT`: 是否使用梯度检查点节省显存
- `USE_MEAN_POOLING`: 是否使用均值池化，默认为 False
- `INIT_SCALE`: 初始缩放因子，默认为 0.001
- `WITH_CP`: 是否使用检查点，默认为 False
- `COS_ATTN`: 是否使用余弦注意力，默认为 False

### 正则化参数
- `DROP`: 通用 dropout 率，默认为 0.0
- `ATTN_DROP_RATE`: 注意力层 dropout 率，默认为 0.0
- `DROP_PATH`: 路径 dropout 率，用于随机深度，默认为 0.1

### 掩码配置
- `MASK_TYPE`: 掩码类型，默认为 'tube'
- `WINDOW_SIZE`: 窗口大小，默认为 [8, 14, 14]
- `MASK_RATIO`: 掩码比例，默认为 0.75
- `PART_WIN_SIZE`: 部分窗口大小，默认为 [8, 7, 7]
- `PART_APPLY_SYMMETRY`: 是否应用对称性，默认为 True

### 注意力机制参数
- `ATTN_TYPE`: 注意力类型，默认为 'local_global'
- `LG_REGION_SIZE`: 局部-全局注意力的区域大小，默认为 [2, 2, 10]
- `LG_FIRST_ATTN_TYPE`: 第一层注意力类型，默认为 'self'
- `LG_THIRD_ATTN_TYPE`: 第三层注意力类型，默认为 'cross'
- `LG_ATTN_PARAM_SHARING_FIRST_THIRD`: 第一和第三层是否共享参数，默认为 False
- `LG_ATTN_PARAM_SHARING_ALL`: 是否所有层共享参数，默认为 False
- `LG_CLASSIFY_TOKEN_TYPE`: 分类 token 类型，默认为 'org'
- `LG_NO_SECOND`: 是否禁用第二层，默认为 False
- `LG_NO_THIRD`: 是否禁用第三层，默认为 False

## 数据配置 (DATA)

### 数据集参数
- `DATASET_NAME`: 数据集名称，默认为 'Kinetics-400'
- `DATA_PATH`: 数据路径，默认为 '/path/to/data'
- `EVAL_DATA_PATH`: 验证数据路径，默认为 None
- `NUM_CLASSES`: 分类数量，默认为 400
- `NUM_SEGMENTS`: 视频片段数量，默认为 1
- `NUM_FRAMES`: 视频帧数，默认为 16
- `SAMPLING_RATE`: 帧采样率，默认为 4
- `IMAGENET_DEFAULT_MEAN_AND_STD`: 是否使用 ImageNet 默认均值和标准差，默认为 True

### 数据加载参数
- `BATCH_SIZE`: 批次大小，默认为 64
- `NUM_WORKERS`: 数据加载工作进程数，默认为 4
- `PIN_MEMORY`: 是否将数据加载到锁页内存，默认为 True
- `SHORT_SIDE_SIZE`: 短边尺寸，默认为 320

### 测试参数
- `TEST_NUM_SEGMENT`: 测试时的片段数量，默认为 5
- `TEST_NUM_CROP`: 测试时的裁剪数量，默认为 3

## 数据增强配置 (AUGMENTATION)

### Mixup 参数
- **用途**: 提高模型泛化能力，减少过拟合
- **原理**: 将两个训练样本按权重混合，标签也按相同权重混合
- **参数说明**:
  - `MIXUP`: Mixup的强度，0表示不使用，默认为 0.8
  - `CUTMIX`: CutMix的强度，类似Mixup但是在空间上混合，默认为 1.0
  - `CUTMIX_MINMAX`: CutMix 的最小最大值，默认为 None
  - `MIXUP_PROB`: 执行mixup的概率，默认为 1.0
  - `MIXUP_SWITCH_PROB`: 在mixup和cutmix间切换的概率，默认为 0.5
  - `MIXUP_MODE`: 混合模式，'batch'表示批次内混合，默认为 'batch'

### Random Erase 参数
- **用途**: 随机擦除图像的部分区域，提高模型鲁棒性
- **原理**: 在训练图像上随机选择矩形区域并用随机像素值填充
- **参数说明**:
  - `RANDOM_ERASE_PROB`: 随机擦除的概率，默认为 0.25
  - `RANDOM_ERASE_MODE`: 擦除模式，'pixel'表示用随机像素值填充，默认为 'pixel'
  - `RANDOM_ERASE_COUNT`: 每张图像擦除的区域数量，默认为 1
  - `RANDOM_ERASE_SPLIT`: 是否在数据增强的第一次分割中跳过擦除，默认为 False

### Auto Augmentation 参数
- **用途**: 自动学习最佳的数据增强策略
- **原理**: 使用预定义的增强策略组合，自动选择最优的增强方法
- **参数说明**:
  - `AUTO_AUGMENT`: 自动增强策略字符串，如'rand-m7-n4-mstd0.5-inc1'
    - `m7`: magnitude=7，增强强度
    - `n4`: num_layers=4，每张图像应用的增强操作数
    - `mstd0.5`: magnitude_std=0.5，强度的标准差
    - `inc1`: 使用递增的增强策略

### 其他增强参数
- `COLOR_JITTER`: 颜色抖动强度，影响亮度、对比度、饱和度，默认为 0.4
- `TRAIN_INTERPOLATION`: 训练时的插值方法，'bicubic'表示双三次插值，默认为 'bicubic'
- `NUM_SAMPLE`: 采样数量，默认为 2
- `LABEL_SMOOTHING`: 标签平滑参数，减少过拟合，默认为 0.1
- `NO_AUGMENTATION`: 是否禁用数据增强，默认为 False

## 优化配置 (OPTIMIZATION)

### 学习率调度
- `SCHED`: 学习率调度器类型，默认为 'cosine'
- `LR`: 基础学习率，默认为 1e-3
- `MIN_LR`: 最小学习率，默认为 1e-6
- `WARMUP_LR`: 预热阶段的学习率，默认为 1e-6
- `WARMUP_EPOCHS`: 预热轮数，默认为 5
- `WARMUP_STEPS`: 预热步数，-1表示使用轮数，默认为 -1

### 优化器参数
- `OPTIMIZER`: 优化器类型，默认为 'adamw'
- `OPT_EPS`: 优化器的epsilon值，避免除零错误，默认为 1e-8
- `OPT_BETAS`: Adam优化器的beta参数，控制动量，默认为 [0.9, 0.999]
- `WEIGHT_DECAY`: 权重衰减，L2正则化强度，默认为 0.05
- `WEIGHT_DECAY_END`: 权重衰减结束值，默认为 None
- `MOMENTUM`: SGD优化器的动量参数，默认为 0.9

### 梯度处理
- `CLIP_GRAD`: 梯度裁剪阈值，防止梯度爆炸，默认为 None
- `LAYER_DECAY`: 层级学习率衰减，不同层使用不同学习率，默认为 0.75

## 训练配置 (TRAINING)

### 基础训练参数
- `EPOCHS`: 训练轮数，默认为 30
- `START_EPOCH`: 开始训练的轮数，默认为 0
- `UPDATE_FREQ`: 更新频率，默认为 1
- `SAVE_CKPT_FREQ`: 保存检查点的频率，默认为 100
- `SAVE_CKPT`: 是否保存检查点，默认为 True
- `AUTO_RESUME`: 是否自动恢复训练，默认为 True
- `RESUME`: 恢复训练的检查点路径，默认为 ''
- `FINETUNE`: 微调的预训练模型路径，默认为 ''

### 模型EMA (Exponential Moving Average)
- `MODELEMA_`: 是否使用模型EMA，默认为 False
- `MODEL_EMA_DECAY`: EMA衰减率，默认为 0.9999
- `MODEL_EMA_FORCE_CPU`: 是否强制在CPU上计算EMA，默认为 False

### 评估参数
- `DISABLE_EVAL_DURING_FINETUNING`: 是否在微调期间禁用验证，默认为 False
- `EVAL_ONLY`: 是否只进行评估，默认为 False
- `DIST_EVAL`: 是否使用分布式评估，默认为 False
- `VAL_METRIC`: 验证指标，用于选择最佳模型，默认为 'acc1'

## 凝视估计配置 (GAZE)

### L2CS 方法参数
- `USE_L2CS`: 是否使用L2CS方法，默认为 False
- `NUM_BINS`: L2CS方法的分箱数量，默认为 90
- `ALPHA_REG`: L2CS中回归损失的权重，默认为 1.0
- `BIN_WIDTH`: 每个分箱的宽度（度），默认为 2.0

## 系统配置 (SYSTEM)

### 基础系统参数
- `DEVICE`: 训练设备，'cuda'或'cpu'，默认为 'cuda'
- `SEED`: 随机种子，默认为 0
- `OUTPUT_DIR`: 输出目录，默认为 ''
- `LOG_DIR`: 日志目录，默认为 None

### 分布式训练参数
- `WORLD_SIZE`: 分布式训练的进程数，默认为 1
- `LOCAL_RANK`: 本地进程排名，默认为 -1
- `DIST_ON_ITP`: 是否在ITP上进行分布式训练，默认为 False
- `DIST_URL`: 分布式训练的URL，默认为 'env://'
- `DISTRIBUTED`: 是否启用分布式训练，默认为 False
- `DIST_BACKEND`: 分布式训练的后端，默认为 'nccl'

### 高级系统参数
- `ENABLE_DEEPSPEED`: 是否启用DeepSpeed，默认为 False
- `SAVE_FEATURE`: 是否保存特征，默认为 False
- `GPU`: 指定GPU设备，默认为 -1

## 配置文件使用方法

### 基本使用
```python
from src.utils.config import get_cfg, merge_config_file, freeze_cfg

# 获取默认配置
cfg = get_cfg()

# 从YAML文件合并配置
cfg = merge_config_file(cfg, 'configs/gaze360_finetune.yaml')

# 冻结配置使其不可变
freeze_cfg()
```

### 命令行参数覆盖
```bash
python run_finetuning_with_yacs.py \
    --config configs/gaze360_finetune.yaml \
    --output_dir output/ \
```

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


## 注意事项

1. **YACS配置系统**: 新版本使用YACS配置系统，配置项名称采用大写形式
2. **配置冻结**: 建议在训练开始前冻结配置，防止意外修改
3. **类型转换**: 注意YAML文件中的数值类型，科学计数法可能被解析为字符串
4. **向后兼容**: 提供了从argparse参数转换到YACS配置的兼容函数
