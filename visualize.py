import json
import os
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def extract_test_angular_error_from_file(file_path):
    """从单个文件中提取test_angular_error数据"""
    test_angular_errors = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                try:
                    data = json.loads(line)
                    test_angular_errors.append(data['test_war'])
                except json.JSONDecodeError:
                    print(f"警告：无法解析文件 {file_path} 中的行: {line}")
                except KeyError:
                    print(f"警告：文件 {file_path} 中的行缺少 'test_war' 键: {line}")

    return test_angular_errors

def process_all_files(directory_path, output_csv_path):
    """处理所有实验文件并将结果保存到CSV"""
    # 获取目录中所有.txt文件
    txt_files = list(Path(directory_path).glob("*.txt"))
    
    if not txt_files:
        print(f"在目录 {directory_path} 中没有找到 .txt 文件")
        return
    
    # 存储所有实验的数据
    all_data = {}
    
    # 读取每个文件的数据
    for file_path in txt_files:
        print(f"处理文件: {file_path.name}")
        test_angular_errors = extract_test_angular_error_from_file(file_path)
        all_data[file_path.stem] = test_angular_errors  # 使用文件名（不含扩展名）作为列名
    
    # 找到最大epoch数以确定行数
    max_epochs = max(len(errors) for errors in all_data.values()) if all_data else 0
    
    # 创建DataFrame
    df_data = {}
    
    for exp_name, errors in all_data.items():
        # 如果某个实验的epoch数不足，用NaN填充
        padded_errors = errors + [float('nan')] * (max_epochs - len(errors))
        df_data[exp_name] = padded_errors
    
    # 创建DataFrame，行索引为epoch
    df = pd.DataFrame(df_data)
    df.index.name = 'epoch'
    
    # 保存到CSV文件
    df.to_csv(output_csv_path, index=True)
    print(f"结果已保存到: {output_csv_path}")
    print(f"CSV文件形状: {df.shape}")
    print("\n前几行预览:")
    print(df.head())


def plot_angular_error_curves(csv_file_path, output_image_path=None):
    """
    从CSV文件绘制test_angular_error折线图（修改版）
    
    Parameters:
    csv_file_path: CSV文件路径
    output_image_path: 输出图片路径，如果为None则显示图片
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, index_col='epoch')
    
    # 设置中文字体支持（如果需要）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    # 创建图形 - 增加宽度，保持高度
    plt.figure(figsize=(16, 8))  # 宽度从12增加到16
    
    # 获取所有实验列名
    experiment_columns = df.columns.tolist()
    
    # 使用seaborn的颜色调色板，确保颜色区分度高
    colors = sns.color_palette("husl", len(experiment_columns))
    
    # 绘制每条折线
    for i, col in enumerate(experiment_columns):
        # 获取该实验的数据（去除NaN值）
        series = df[col].dropna()
        epochs = series.index
        errors = series.values
        
        plt.plot(epochs, errors, 
                marker='o', 
                linewidth=2, 
                markersize=3,  # 减小点的大小从6到3
                label=col,
                color=colors[i])
    
    # 设置图表标题和轴标签
    plt.title('Test WAR Comparison Across Experiments', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Test WAR', fontsize=14)

    # 添加网格
    plt.grid(True, alpha=0.3)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 调整布局，避免图例被截断
    plt.tight_layout()
    
    # 设置y轴范围，让趋势更明显
    all_errors = df.values.flatten()
    all_errors = all_errors[~np.isnan(all_errors)]  # 移除NaN值
    if len(all_errors) > 0:
        y_min, y_max = np.min(all_errors), np.max(all_errors)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
    
    # 保存或显示图片
    if output_image_path:
        plt.savefig(output_image_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {output_image_path}")
    else:
        plt.show()

def plot_angular_error_curves_advanced(csv_file_path, output_image_path=None, 
                                     figsize=(14, 10), dpi=300):
    """
    高级版本的绘图函数，提供更多的自定义选项
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, index_col='epoch')
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图形
    fig, ax = plt.subplots(figsize=figsize)
    
    # 获取所有实验列名
    experiment_columns = df.columns.tolist()
    
    # 使用更丰富的颜色和样式
    colors = sns.color_palette("tab10", len(experiment_columns)) if len(experiment_columns) <= 10 else \
             sns.color_palette("husl", len(experiment_columns))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x']
    
    # 绘制每条折线
    for i, col in enumerate(experiment_columns):
        # 获取该实验的数据（去除NaN值）
        series = df[col].dropna()
        epochs = series.index
        errors = series.values
        
        marker_style = markers[i % len(markers)]
        
        ax.plot(epochs, errors, 
               marker=marker_style,
               linewidth=2.5, 
               markersize=8,
               label=col,
               color=colors[i],
               alpha=0.8)
    
    # 设置图表标题和轴标签
    ax.set_title('Test Angular Error Comparison Across Experiments', 
                fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Angular Error (°)', fontsize=14, fontweight='bold')
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # 设置y轴格式，避免科学计数法
    ax.ticklabel_format(style='plain', axis='y')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图片
    if output_image_path:
        plt.savefig(output_image_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"高级版图片已保存到: {output_image_path}")
    else:
        plt.show()



def main():
    # 设置输入目录和输出文件路径
    input_directory = "/root/lfz/Facial-Foundation-Model/e"
    output_csv = "block.csv"
    
    print(f"输入目录: {input_directory}")
    print(f"输出文件: {output_csv}")
    print("-" * 50)
    
    # 处理文件
    process_all_files(input_directory, output_csv)

    csv_file = "/root/lfz/Facial-Foundation-Model/block.csv"

    
    # 检查文件是否存在
    if not Path(csv_file).exists():
        print(f"错误：文件 {csv_file} 不存在！")
        return
    
    
    output_image = "/root/lfz/Facial-Foundation-Model/block.png"
    
   
    plot_angular_error_curves(csv_file, output_image)

if __name__ == "__main__":
    main()