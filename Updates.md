# 更新日志

## 7.17 代码重构
1. 拆分了文件夹结构，把根目录下的大多文件分类放入 `src` 文件夹下。
2. 用 yacs.config 替换了原有的配置文件管理方式。详细测试、workflow 见 `test_yacs_config.py`。

## 7.18 继续重构
现在单卡 finetune 可以运行, 即
```bash
python run_finetune_with_yacs.py --config configs/gaze360_finetune.yaml
```

但是 loss 没有下降，可能代码还有 bug.

## 7.19
之前的 finetune 的 bug 修了一些。现在 50 epochs 的 err 会降到 30° 左右，并且仍然会慢速下降。

加了 pretraining 的代码，测试了单卡 pretrain 可以运行。

多卡的 nccl 分布式训调试了很久，仍然无法运行，换成了 gloo 后端，可以运行，但是运行极慢。仍待调式。

## 7.20
gloo 后端的分布式训练修好了，代码如：
```bash
torchrun --nproc_per_node=4 --master_port=29501 run_finetuning_with_yacs.py --config configs/gaze360_finetune.yaml
```
记得把 `configs/gaze360_finetune.yam` 中的 `distributed` 设置为 `true`，并且 `world_size` 设置为您的 GPU 数量。

## 7.21
修复了 fnetune 中学习率缩放和模型加载的 bug，现在可以复现原论文的 finetune 结果。

当前的文件结构:
```
.
├── configs
│   └── gaze360_finetune.yaml                                            # 配置文件可参考该文件
├── src
│   ├── dataset                                       # 数据集相关代码  
│   │   ├── augment                                   # 数据增强以及transform相关代码,还没完全分好类
│   │   │   ├── functional.py
│   │   │   ├── rand_augment.py
│   │   │   ├── random_erasing.py
│   │   │   ├── video_transforms.py
│   │   │   └── volume_transforms.py
│   │   ├── datasets.py                              # 数据集入口
│   │   ├── kinetics.py                              # 具体的视频加载 以及 getitem
│   │   ├── masking_generator.py
│   │   ├── ssv2.py
│   │   └── transforms.py
│   ├── engine                                       # 定义了训练、测试逻辑，为【核心】代码
│   │   ├── pretrain_engine.py
│   │   ├── train_engine.py
│   │   └── val_engine.py
│   ├── models                                       # 模型架构，仍待进一步重构
│   │   ├── ViT.py
│   │   ├── ViT_pretrain.py
│   │   └── layers.py
│   ├── optim                                        # 优化器相关代码
│   │   ├── mixup.py
│   │   └── optim_factory.py
│   └── utils                                        # 工具函数  
│       ├── config.py                                # 配置文件相关函数，本次重构的【核心】
│       ├── ddp.py                                   # 分布式训练相关函数， 暂时没有用到                         
│       ├── evaluation.py                            # 评估的工具函数，计算一堆 metric
│       ├── gaze.py                                  # 凝视估计相关函数               
│       ├── logger.py                                # 日志记录相关函数
│       └── utils.py                                 # 一些未分类的工具函数，主要和 ddp、模型加载 相关
├── [discarded] run_class_finetuning.py              # 之前的脚本，留着debug
├── [discarded] run_gaze360_finetuning.py
├── [discarded] run_mae_pretraining.py  
├── run_finetuning_with_yacs.py                     # 新的 finetune 脚本， 【核心】入口
├── run_pretraining_with_yacs.py                     # 新的 pretain 脚本， 【核心】入口
├── test_config.py                                  # 测试配置文件的脚本
└── test_yacs_config.py                             # 测试配置文件的脚本，本次重构的【核心】用例在这里面