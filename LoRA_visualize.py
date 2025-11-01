import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_lora_stats(lora_params):


    # 提取统计量
    blocks = set()
    attentions = set()
    lora_types = set()
    stats = {}

    for name, tensor in lora_params.items():
        # 解析名称: module.blocks.X.attention_type.lora_layer_type.lora_A/B
        parts = name.split('.')
        if len(parts) >= 6 and parts[0] == 'module' and parts[1] == 'blocks':
            block_id = int(parts[2])
            attn_type = parts[3] # first_attn, second_attn, third_attn
            lora_layer_type = parts[4] # lora_q, lora_kv
            matrix_type = parts[5] # lora_A, lora_B

            blocks.add(block_id)
            attentions.add(attn_type)
            lora_types.add(lora_layer_type)

            key = (block_id, attn_type, lora_layer_type, matrix_type)
            # 计算 Frobenius 范数, 均值, 标准差
            stats[key] = {
                'norm': torch.norm(tensor, p='fro').item(),
                'mean': tensor.mean().item(),
                'std': tensor.std().item(),
                'max': tensor.max().item(),
                'min': tensor.min().item()
            }

    blocks = sorted(list(blocks))
    attentions = sorted(list(attentions))
    lora_types = sorted(list(lora_types))

    # 准备绘图数据
    stats_to_plot = ['norm', 'mean', 'std']
    output_dir = "./lora_visualizations2"
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for stat_name in stats_to_plot:
        # 创建一个矩阵，行是 block，列是 (attention_type, lora_layer_type, matrix_type)
        matrix_types = ['lora_A', 'lora_B']
        col_labels = []
        for attn_type in attentions:
            for lora_type in lora_types:
                for mat_type in matrix_types:
                    col_labels.append(f"{attn_type}\n{lora_type}\n{mat_type}")

        plot_matrix = np.full((len(blocks), len(col_labels)), np.nan) # 使用 NaN 填充缺失值

        for row_idx, block_id in enumerate(blocks):
            for col_idx, col_label in enumerate(col_labels):
                parts = col_label.split('\n')
                attn_type, lora_type, mat_type = parts[0], parts[1], parts[2]
                key = (block_id, attn_type, lora_type, mat_type)
                if key in stats:
                    plot_matrix[row_idx, col_idx] = stats[key][stat_name]

        # --- 绘制单个统计量的热图 ---
        plt.figure(figsize=(20, 10)) # 可以根据需要调整大小

        sns.heatmap(plot_matrix, xticklabels=col_labels, yticklabels=blocks,
                    annot=True, fmt='.4f', cbar_kws={'label': stat_name}, cmap='viridis')
        plt.title(f'LoRA {stat_name.upper()} Heatmap')
        plt.xlabel('Attention Layer / LoRA Type / Matrix')
        plt.ylabel('Block ID')

        # 保存图像
        filename = f"lora_{stat_name}_heatmap.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight') # 保存为高分辨率 PNG
        print(f"Saved {stat_name} heatmap to {filepath}")

        # 显示图像（可选，如果你想在保存的同时看到图像）
        # plt.show()

        # 清除当前图形，为下一个图做准备
        plt.clf()
        plt.close()

    print(f"All visualizations saved to {output_dir}")


def visualize_lora_update_ratio_vs_original(name, state_dict_path, output_dir="./lora_update_comparison", scale_values=[0.5]*16):
    """
    可视化比较每个 block 中 LoRA 更新的权重与初始权重的 Frobenius 范数比值。
    LoRA 更新 = (lora_B @ lora_A) * scale
    比值 = ||LoRA 更新|| / ||原始权重||
    :param state_dict_path: 包含模型权重（包括 LoRA 和原始权重）的 .pth 文件路径
    :param output_dir: 保存图像的目录
    :param scale_values: 一个字典，键为 block_id (int)，值为该 block 内 LoRA 的 scale 值。
                         如果为 None，则假设 scale=1.0。你需要根据你的训练配置传入正确的 scale。
                         例如: {0: 16.0, 1: 16.0, ..., 7: 0.5, 8: 16.0, ...}
    """
    # 加载 state_dict
    state_dict = torch.load(state_dict_path, map_location='cpu')
    model_state = state_dict["model"] 
    state_dict = model_state
    # 筛选出 LoRA 参数和对应的原始参数
    lora_params = {k: v for k, v in state_dict.items() if 'lora_' in k and ('lora_A' in k or 'lora_B' in k)}
    original_params = {k: v for k, v in state_dict.items() if 'lora_' not in k and ('q.weight' in k or 'kv.weight' in k)}
    # print(original_params)
    # 提取 block ID 和统计量
    blocks = set()
    update_ratios = {}

    for lora_name, lora_tensor in lora_params.items():
        # 解析 LoRA 名称: module.blocks.X.attention_type.lora_layer_type.lora_A/B
        parts = lora_name.split('.')
        if len(parts) >= 6 and parts[0] == 'module' and parts[1] == 'blocks':
            block_id = int(parts[2])
            attn_type = parts[3] # first_attn, second_attn, third_attn
            lora_layer_type = parts[4] # lora_q, lora_kv
            matrix_type = parts[5] # lora_A, lora_B

            blocks.add(block_id)

            # 找到对应的原始权重名称
            # 例如，将 module.blocks.0.first_attn.lora_q.lora_A -> module.blocks.0.first_attn.q.weight
            # 或者 module.blocks.0.first_attn.lora_kv.lora_A -> module.blocks.0.first_attn.kv.weight
            if lora_layer_type == 'lora_q':
                original_name = f"module.blocks.{block_id}.{attn_type}.q.weight"
            elif lora_layer_type == 'lora_kv':
                original_name = f"module.blocks.{block_id}.{attn_type}.kv.weight"
            else:
                continue # 跳过非 q 或 kv 的 LoRA (虽然根据你的结构，应该只有这两个)

            if original_name not in original_params:
                print(f"Warning: Original weight {original_name} not found for LoRA {lora_name}")
                continue

            original_tensor = original_params[original_name]

            # 获取对应的另一个 LoRA 矩阵 (A 对应 B, B 对应 A)
            other_lora_name = lora_name.replace('.lora_A', '.lora_B') if 'lora_A' in lora_name else lora_name.replace('.lora_B', '.lora_A')
            if other_lora_name not in lora_params:
                 print(f"Warning: Other LoRA matrix {other_lora_name} not found for {lora_name}")
                 continue

            other_lora_tensor = lora_params[other_lora_name]

            # 计算 LoRA 更新矩阵 (lora_B @ lora_A) * scale
            # 确保矩阵顺序正确 (B @ A)
            if 'lora_A' in lora_name:
                lora_A = lora_tensor
                lora_B = other_lora_tensor
            else: # 'lora_B' in lora_name
                lora_B = lora_tensor
                lora_A = other_lora_tensor

            # 获取 scale (这里需要你传入正确的 scale_values 字典)
            scale = scale_values.get(block_id, 1.0) if scale_values else 1.0

            lora_update_matrix = (lora_B @ lora_A) * scale

            # 计算 Frobenius 范数
            lora_update_norm = torch.norm(lora_update_matrix, p='fro').item()
            original_norm = torch.norm(original_tensor, p='fro').item()

            # 计算比值
            if original_norm == 0:
                print(f"Warning: Original norm for {original_name} is zero. Cannot compute ratio.")
                ratio = float('inf')
            else:
                ratio = lora_update_norm / original_norm

            # 存储比值
            key = (block_id, attn_type, lora_layer_type)
            update_ratios[key] = ratio

    blocks = sorted(list(blocks))
    attn_types = sorted(list(set(k[1] for k in update_ratios.keys())))
    lora_types = sorted(list(set(k[2] for k in update_ratios.keys())))

    # 准备绘图数据
    # 创建一个矩阵，行是 block，列是 (attention_type, lora_layer_type)
    col_labels = []
    for attn_type in attn_types:
        for lora_type in lora_types:
            col_labels.append(f"{attn_type}\n{lora_type}")

    plot_matrix = np.full((len(blocks), len(col_labels)), np.nan) # 使用 NaN 填充缺失值

    for row_idx, block_id in enumerate(blocks):
        for col_idx, col_label in enumerate(col_labels):
            parts = col_label.split('\n')
            attn_type, lora_type = parts[0], parts[1]
            key = (block_id, attn_type, lora_type)
            if key in update_ratios:
                plot_matrix[row_idx, col_idx] = update_ratios[key]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # --- 绘制热图 ---
    plt.figure(figsize=(15, 10)) # 可以根据需要调整大小

    # 使用对数刻度可能有助于观察差异很大的比值
    # plot_matrix_log = np.log10(plot_matrix + 1e-8) # 加一个很小的数避免 log(0)
    # sns.heatmap(plot_matrix_log, xticklabels=col_labels, yticklabels=blocks,
    #             annot=plot_matrix, fmt='.2e', cbar_kws={'label': 'log10(Ratio)'}, cmap='viridis')

    # 使用线性刻度
    sns.heatmap(plot_matrix, xticklabels=col_labels, yticklabels=blocks,
                annot=plot_matrix, fmt='.4f', cbar_kws={'label': '||LoRA Update||_F / ||Original||_F'}, cmap='viridis')
    plt.title(f'LoRA Update Magnitude Ratio vs Original Weights')
    plt.xlabel('Attention Layer / LoRA Type')
    plt.ylabel('Block ID')

    # 保存图像
    filename = name
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight') # 保存为高分辨率 PNG
    print(f"Saved LoRA update ratio heatmap to {filepath}")

    # 显示图像（可选）
    # plt.show()

    # 清除当前图形
    plt.clf()
    plt.close()

    print(f"Visualization saved to {output_dir}")


def load_lora_weights_from_file(filepath):
    """
    从 .pth 文件中加载 LoRA 权重
    :param filepath: 权重文件的路径
    :return: 权重字典
    """
    try:
        weights_data = torch.load(filepath, map_location='cpu')
        return weights_data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def visualize_lora_from_file(weights_data, epoch, layer_names=None, save_dir=None, save=True):
    """
    从加载的权重数据中可视化 LoRA 权重
    :param weights_data: torch.load 加载的权重字典
    :param epoch: 当前 epoch (用于标题和文件名)
    :param layer_names: 要可视化的层名列表。如果为 None，则可视化所有层。
    :param save_dir: 保存图像的目录
    :param save: 是否保存图像
    """
    if weights_data is None:
        print("No weights data to visualize.")
        return

    if layer_names is None:
        layer_names = list(weights_data.keys())

    num_layers = len(layer_names)
    if num_layers == 0:
        print("No layers specified for visualization.")
        return

    fig, axes = plt.subplots(num_layers, 2, figsize=(15, 5 * num_layers))
    if num_layers == 1:
        axes = axes.reshape(1, -1)

    for i, layer_name in enumerate(layer_names):
        if layer_name not in weights_data:
            print(f"Layer {layer_name} not found in loaded weights.")
            continue

        layer_weights = weights_data[layer_name]
        ax_A = axes[i, 0]
        ax_B = axes[i, 1]

        # 可视化 lora_A
        if 'lora_A' in layer_weights:
            lora_A = layer_weights['lora_A']
            sns.heatmap(lora_A.numpy(), ax=ax_A, cmap="viridis", cbar_kws={'label': 'Value'})
            ax_A.set_title(f'{layer_name} - lora_A (Epoch {epoch})')
            ax_A.set_xlabel('Input Features')
            ax_A.set_ylabel('Rank')
        else:
            ax_A.text(0.5, 0.5, 'lora_A not found', horizontalalignment='center', verticalalignment='center', transform=ax_A.transAxes)
            ax_A.set_title(f'{layer_name} - lora_A (Epoch {epoch})')

        # 可视化 lora_B
        if 'lora_B' in layer_weights:
            lora_B = layer_weights['lora_B']
            sns.heatmap(lora_B.numpy(), ax=ax_B, cmap="viridis", cbar_kws={'label': 'Value'})
            ax_B.set_title(f'{layer_name} - lora_B (Epoch {epoch})')
            ax_B.set_xlabel('Rank')
            ax_B.set_ylabel('Output Features')
        else:
            ax_B.text(0.5, 0.5, 'lora_B not found', horizontalalignment='center', verticalalignment='center', transform=ax_B.transAxes)
            ax_B.set_title(f'{layer_name} - lora_B (Epoch {epoch})')

    plt.tight_layout()

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        vis_file = os.path.join(save_dir, f"lora_weights_epoch_{epoch}.png")
        plt.savefig(vis_file, dpi=150, bbox_inches='tight')
        print(f"Saved visualization for epoch {epoch} to {vis_file}")
    elif save:
        print("Warning: 'save' is True but 'save_dir' is not provided. Image not saved.")

    plt.show()

def plot_lora_evolution_from_files(filepaths_and_epochs, layer_name, save_dir=None, save=True):
    """
    从多个 .pth 文件中加载数据，绘制指定 layer 的 LoRA 权重变化
    :param filepaths_and_epochs: [(filepath, epoch), ...] 列表
    :param layer_name: 要可视化的 layer 名称
    :param save_dir: 保存图像的目录
    :param save: 是否保存图像
    """
    epochs = []
    metrics_A_mean = []
    metrics_B_mean = []
    metrics_A_std = []
    metrics_B_std = []
    metrics_A_norm = []
    metrics_B_norm = []

    for filepath, epoch in filepaths_and_epochs:
        weights_data = load_lora_weights_from_file(filepath)
        if weights_data and layer_name in weights_data:
            layer_weights = weights_data[layer_name]
            if 'lora_A' in layer_weights and 'lora_B' in layer_weights:
                lora_A = layer_weights['lora_A']
                lora_B = layer_weights['lora_B']
                epochs.append(epoch)
                metrics_A_mean.append(lora_A.mean().item())
                metrics_B_mean.append(lora_B.mean().item())
                metrics_A_std.append(lora_A.std().item())
                metrics_B_std.append(lora_B.std().item())
                metrics_A_norm.append(torch.norm(lora_A).item())
                metrics_B_norm.append(torch.norm(lora_B).item())
            else:
                print(f"Warning: lora_A or lora_B not found for {layer_name} in file {filepath}")

    if not epochs:
        print(f"No valid data found for {layer_name} across the specified files.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    # Plot Mean
    axes[0].plot(epochs, metrics_A_mean, label='lora_A Mean', marker='o')
    axes[0].plot(epochs, metrics_B_mean, label='lora_B Mean', marker='s')
    axes[0].set_title(f'{layer_name} - Mean Evolution')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Value')
    axes[0].legend()
    axes[0].grid(True)

    # Plot Std
    axes[1].plot(epochs, metrics_A_std, label='lora_A Std', marker='o')
    axes[1].plot(epochs, metrics_B_std, label='lora_B Std', marker='s')
    axes[1].set_title(f'{layer_name} - Std Evolution')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Std Value')
    axes[1].legend()
    axes[1].grid(True)

    # Plot Norm
    axes[2].plot(epochs, metrics_A_norm, label='lora_A Norm', marker='o')
    axes[2].plot(epochs, metrics_B_norm, label='lora_B Norm', marker='s')
    axes[2].set_title(f'{layer_name} - Frobenius Norm Evolution')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Frobenius Norm')
    axes[2].legend()
    axes[2].grid(True)

    # Plot Histogram of final epoch's lora_A
    # Load the final epoch's data again to get the matrix for histogram
    final_epoch_path = filepaths_and_epochs[-1][0]
    final_weights_data = load_lora_weights_from_file(final_epoch_path)
    if final_weights_data and layer_name in final_weights_data and 'lora_A' in final_weights_data[layer_name]:
        final_lora_A = final_weights_data[layer_name]['lora_A'].flatten().numpy()
        axes[3].hist(final_lora_A, bins=50, alpha=0.7, label='lora_A Values (Last Epoch)')
        axes[3].set_title(f'{layer_name} - lora_A Value Distribution (Last Epoch)')
        axes[3].set_xlabel('Value')
        axes[3].set_ylabel('Frequency')
        axes[3].legend()
        axes[3].grid(True)
    else:
        axes[3].text(0.5, 0.5, 'lora_A data for histogram not found', horizontalalignment='center', verticalalignment='center', transform=axes[3].transAxes)
        axes[3].set_title(f'{layer_name} - lora_A Value Distribution (Last Epoch)')

    plt.tight_layout()

    if save and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        evolution_file = os.path.join(save_dir, f"lora_evolution_{layer_name.replace('.', '_')}.png")
        plt.savefig(evolution_file, dpi=150, bbox_inches='tight')
        print(f"Saved evolution visualization for {layer_name} to {evolution_file}")
    elif save:
        print("Warning: 'save' is True but 'save_dir' is not provided. Image not saved.")

    plt.show()


# --- 使用示例 ---

# 1. 可视化单个 epoch 的权重
# 假设你的权重文件在 "./my_lora_weights/epoch_X/lora_weights.pth"
# epoch_to_visualize = 0
# weights_file_path = f"./my_lora_weights/epoch_{epoch_to_visualize}/lora_weights.pth"
# weights_data = load_lora_weights_from_file(weights_file_path)

# if weights_data:
#     # 可视化所有层
#     visualize_lora_from_file(weights_data, epoch=epoch_to_visualize, save_dir="./visualizations", save=True)

#     # 或者可视化特定层
#     # selected_layers = ["blocks.0.attn.lora_qkv", "blocks.1.attn.lora_qkv"] # 根据你的实际层名调整
#     # visualize_lora_from_file(weights_data, epoch=epoch_to_visualize, layer_names=selected_layers, save_dir="./visualizations", save=True)


# # 2. 可视化多个 epoch 的权重变化
# # 构建文件路径和对应 epoch 的列表
# num_epochs = 3 # 假设你训练了 3 个 epoch (0, 1, 2)
# filepaths_and_epochs = []
# for epoch in range(num_epochs):
#     path = f"/root/lfz/Facial-Foundation-Model/output/lora/dfew_scale16_2/checkpoint-{epoch}.pth"
#     if os.path.exists(path): # 确保文件存在
#         filepaths_and_epochs.append((path, epoch))
#     else:
#         print(f"Warning: File {path} does not exist. Skipping epoch {epoch}.")

# if filepaths_and_epochs:
#     # 选择一个具体的 layer name 来查看其变化
#     # 你需要从你的模型或第一个 epoch 的权重文件中获取实际的 layer name
#     # 例如，假设 weights_data 是从 epoch 0 加载的
#     if weights_data:
#         first_layer_name = list(weights_data.keys())[0] # 取第一个层名作为例子
#         plot_lora_evolution_from_files(filepaths_and_epochs, layer_name=first_layer_name, save_dir="./visualizations", save=True)

#         # 或者选择一个你感兴趣的特定层名
#         # specific_layer_name = "blocks.0.attn.lora_qkv" # 根据你的实际层名调整
#         # plot_lora_evolution_from_files(filepaths_and_epochs, layer_name=specific_layer_name, save_dir="./visualizations", save=True)
#     else:
#         print("No initial weights data loaded to get layer names.")

import torch
import pprint

# 加载 .pth 文件
file_path = "/root/lfz/Facial-Foundation-Model/output/lora/dfew_scale16_2/checkpoint-best.pth" # 替换为你的 .pth 文件路径
loaded_data = torch.load(file_path, map_location='cpu')
model_weight = loaded_data['model']

# 使用 pformat 获取格式化后的字符串
# formatted_string = pprint.pformat(loaded_data, indent=2, width=100, depth=None)
# for k,v in model_weight.items():
#     print(k,v.shape)
lora_params = {k: v for k, v in model_weight.items() if 'lora' in k}
print(lora_params)
visualize_lora_stats(lora_params)
# # 将字符串写入文件
# output_file_path = "cpt49_pth_structure.txt" # 指定输出文件名
# with open(output_file_path, 'w', encoding='utf-8') as f:
#     f.write(formatted_string)

# print(f"Structure printed to {output_file_path}")
# SCALE_VALUES = {i: 16 for i in range(16)}
# visualize_lora_update_ratio_vs_original(name="dfew_basic", state_dict_path="/root/lfz/Facial-Foundation-Model/output/lora/dfew_scale16_2/checkpoint-best.pth", scale_values=SCALE_VALUES)