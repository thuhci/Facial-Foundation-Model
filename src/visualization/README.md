# 视频ViT眼部注视角度模型 - 注意力可视化工具

这个工具包专门为基于视频ViT的眼部注视角度回归模型设计，提供了全面的注意力机制可视化功能。

## 🎯 功能特点

### 1. 空间注意力可视化
- **显示内容**: 模型在图像空间中关注的区域
- **应用场景**: 了解模型主要关注眼部的哪些区域（瞳孔、眼角、眉毛等）
- **输出格式**: 热力图叠加在原始图像上

### 2. 时间注意力可视化  
- **显示内容**: 模型在时间序列中关注的帧
- **应用场景**: 了解哪些时刻的眼部状态对注视角度预测最重要
- **输出格式**: 时间轴上的注意力权重分布图

### 3. 注意力-预测相关性分析
- **显示内容**: 注意力中心与预测注视角度的关联性
- **应用场景**: 验证模型是否真正"看向"了正确的方向
- **输出格式**: 相关性数值和可视化对比

### 4. 多层注意力对比
- **显示内容**: 不同Transformer层的注意力模式差异
- **应用场景**: 理解模型的层级特征学习过程
- **输出格式**: 多层注意力图的并排对比

## 📁 文件结构

```
src/visualization/
├── attention_visualizer.py    # 核心可视化类
├── attention_utils.py         # 辅助工具函数
├── demo_attention.py         # 演示脚本
└── README.md                 # 说明文档
```

## 🚀 快速开始

### 1. 基本使用

```python
from src.visualization.attention_visualizer import AttentionVisualizer
from src.visualization.attention_utils import load_video_for_visualization

# 加载模型
model = YourViTModel.load_from_checkpoint('model.pth')
model.eval()

# 创建可视化器
visualizer = AttentionVisualizer(model, save_dir='./visualizations')

# 加载视频数据
video_tensor = load_video_for_visualization('test_video.mp4')

# 进行预测
with torch.no_grad():
    prediction = model(video_tensor)

# 生成完整的可视化报告
report_dir = visualizer.generate_comprehensive_report(
    video_tensor, prediction, sample_name='sample_001'
)

# 清理资源
visualizer.remove_hooks()
```

### 2. 运行演示

```bash
cd /home/qzk/Facial-Foundation-Model
python src/visualization/demo_attention.py
```

## 🔧 API 参考

### AttentionVisualizer 类

#### 主要方法

**`__init__(model, save_dir)`**
- 初始化可视化器
- `model`: 训练好的ViT模型
- `save_dir`: 保存可视化结果的目录

**`visualize_spatial_attention(video_tensor, layer_idx, head_idx, frame_idx)`**
- 可视化空间注意力
- `video_tensor`: 输入视频张量 (B, C, T, H, W)
- `layer_idx`: 层索引，-1表示最后一层
- `head_idx`: 注意力头索引，None表示平均所有头
- `frame_idx`: 帧索引

**`visualize_temporal_attention(video_tensor, layer_idx)`**
- 可视化时间注意力
- 返回每个时间帧的注意力权重

**`analyze_attention_pattern(video_tensor, prediction)`**
- 分析注意力模式与预测的关系
- 计算注意力中心和相关性

**`generate_comprehensive_report(video_tensor, prediction, sample_name)`**
- 生成包含所有可视化的综合报告
- 自动保存所有图表和分析结果

### 辅助工具函数

**`load_video_for_visualization(video_path, target_size, num_frames)`**
- 从视频文件加载数据

**`load_image_sequence_for_visualization(image_dir, target_size, num_frames)`**
- 从图像序列加载数据

**`create_attention_comparison_plot(attention_maps, layer_names, original_frame)`**
- 创建多层注意力对比图

## 💡 模型适配说明

### 支持的模型结构
- 标准ViT模型（使用`Block`）
- 带Local-Global交互的模型（使用`LGBlock`）
- 自定义Attention层

### 模型要求
1. 模型必须有可访问的attention层
2. 支持3D卷积的patch embedding
3. 输出为2D回归结果（yaw, pitch）

### 如果你的模型结构不同
可以修改`AttentionVisualizer`中的钩子注册逻辑：

```python
# 在modify_attention_for_visualization方法中
# 根据你的模型结构调整attention层的识别和修改逻辑
```

## 📊 可视化结果解读

### 1. 空间注意力热力图
- **红色/黄色区域**: 高注意力权重，模型重点关注的区域
- **蓝色/黑色区域**: 低注意力权重，模型忽略的区域
- **理想情况**: 注意力应集中在眼部关键区域（瞳孔、虹膜）

### 2. 时间注意力图表
- **峰值帧**: 对预测最重要的时间点
- **权重分布**: 反映模型的时间建模策略
- **理想情况**: 应该关注眼动变化的关键时刻

### 3. 相关性分析
- **X-Yaw相关性**: 注意力中心X坐标与预测yaw角的相关性
- **Y-Pitch相关性**: 注意力中心Y坐标与预测pitch角的相关性
- **理想情况**: 相关性应该较高（>0.5）

## 🔍 调试和优化建议

### 1. 如果注意力可视化失败
- 检查模型结构是否包含标准的attention层
- 确认输入数据格式正确
- 查看控制台错误信息

### 2. 如果注意力模式不合理
- 检查模型是否充分训练
- 验证数据预处理是否正确
- 考虑调整模型架构或训练策略

### 3. 性能优化
- 可视化时使用较小的batch size
- 只可视化关键层以节省内存
- 使用GPU加速计算

## 🛠️ 扩展功能

### 添加新的可视化类型
在`AttentionVisualizer`类中添加新方法：

```python
def your_custom_visualization(self, video_tensor, **kwargs):
    # 实现自定义可视化逻辑
    pass
```

### 支持新的模型架构
修改`modify_attention_for_visualization`方法以支持新的attention实现。

## 📝 注意事项

1. **内存使用**: 可视化会增加内存使用，建议使用较小的batch size
2. **钩子管理**: 使用完毕后记得调用`remove_hooks()`
3. **模型状态**: 可视化会修改模型的forward方法，注意在训练时禁用
4. **坐标系**: 注意图像坐标系与角度坐标系的转换

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具包！

---

*这个工具包专门为眼部注视角度回归任务设计，可以帮助研究者更好地理解和改进模型。*
