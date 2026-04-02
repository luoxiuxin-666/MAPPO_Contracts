import numpy as np
import os
import matplotlib
import math

# 设置无头模式后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 配置目录
ENV_NAME = "SFL_PPO_Contract"
RESULT_DIR = os.path.join("results", ENV_NAME, "plots")
os.makedirs(RESULT_DIR, exist_ok=True)


def smooth_data(data, window_size=10):
    """
    计算滑动平均，用于平滑曲线
    """
    if len(data) < window_size:
        return data

    # 使用卷积计算滑动平均
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='valid')

    # 填充前面被切掉的数据，保持长度一致以便绘图
    padding = data[:window_size - 1]
    return np.concatenate([padding, smoothed])


def preprocess_metrics(metrics_dict, current_episode, window_size):
    """
    将原始字典数据处理成绘图所需的 x 轴和平滑后的 y 轴数据
    """
    processed_data = {}

    for key, data_list in metrics_dict.items():
        if not data_list:
            continue

        # 1. 统一提取数据 (安全处理 tensor.item() 和嵌套列表)
        cleaned_list = []
        for item in data_list:
            if isinstance(item, (list, tuple, np.ndarray)):
                cleaned_list.append([x.item() if hasattr(x, 'item') else float(x) for x in item])
            else:
                cleaned_list.append([item.item() if hasattr(item, 'item') else float(item)])

        # 2. 寻找当前指标数据中最长的子序列长度
        lengths = [len(x) for x in cleaned_list]
        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            continue

        # 3. 使用最后一个元素对齐补全短数据
        padded_data = []
        for item_list in cleaned_list:
            if len(item_list) == 0:
                padded_data.append([np.nan] * max_len)
            elif len(item_list) < max_len:
                last_element = item_list[-1]
                padding_length = max_len - len(item_list)
                padded_data.append(item_list + [last_element] * padding_length)
            else:
                padded_data.append(item_list)

        # 4. 转换为规范的 float 数组并展平
        raw_data = np.array(padded_data, dtype=np.float64).flatten()

        if len(raw_data) == 0:
            continue

        # 安全处理 NaN
        if np.isnan(raw_data).any():
            raw_data = np.nan_to_num(raw_data)

        # 5. 动态计算当前指标的 X 轴
        x_axis = np.linspace(0, current_episode, len(raw_data))

        # 6. 计算平滑数据
        real_window = min(window_size, len(raw_data)) if len(raw_data) > 1 else 1
        smoothed_y = smooth_data(raw_data, real_window)

        processed_data[key] = {
            'x': x_axis,
            'y': smoothed_y,
            'window': real_window
        }

    return processed_data


import numpy as np
import os
import matplotlib.pyplot as plt

# 假设 RESULT_DIR 已经定义
RESULT_DIR = "results/plots"
os.makedirs(RESULT_DIR, exist_ok=True)
from matplotlib.ticker import ScalarFormatter

def smooth_data(data, window_size):
    """
    内部平滑函数
    """
    if window_size <= 1:
        return data

    # 动态调整窗口大小，防止超过数据长度
    window_size = min(window_size, len(data))

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(data, kernel, mode='valid')

    # 为了保持长度一致，前面填充原始数据
    padding = data[:window_size - 1]
    return np.concatenate([padding, smoothed])


def preprocess_single_metric(name, values, current_episode, window_size):
    """
    对单个指标进行预处理：计算X轴坐标和平滑处理
    """
    raw_data = np.array(values).flatten()
    if len(raw_data) == 0:
        return None

    # 计算 X 轴
    step_interval = current_episode / len(raw_data)
    x_axis = np.arange(1, len(raw_data) + 1) * step_interval

    # 执行平滑
    y_axis = smooth_data(raw_data, window_size)

    return {'x': x_axis, 'y': y_axis}


def plot_learning_curves_(metrics_dict, current_episode, picture_name=None, window_size=20, combine_plots=False):
    """
    改进版绘图函数：
    :param window_size: 可以是 int (全局统一), 也可以是 dict {'Total_Reward': 50, 'Learning_Rate': 1}
    """
    if not metrics_dict:
        print("[Plotter] Warning: metrics_dict is empty.")
        return

    # 1. 准备处理后的数据字典
    processed_data = {}
    for key, values in metrics_dict.items():
        # 获取该指标对应的特定 window_size
        if isinstance(window_size, dict):
            # 如果字典里没设，默认用 1 (不平滑)
            specific_ws = window_size.get(key, 1)
        else:
            specific_ws = window_size

        # 假设 preprocess_single_metric 在外部已定义
        res = preprocess_single_metric(key, values, current_episode, specific_ws)
        if res:
            processed_data[key] = res

    if not processed_data:
        return

    keys = list(processed_data.keys())
    num_metrics = len(keys)

    # ================= 模式1：联合展示 (所有数据画在同一个坐标系) =================
    if combine_plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle(f'{picture_name if picture_name else "Metrics"} Comparison (Episode {current_episode})',
                     fontsize=16)

        color_palette = ['red', 'blue', 'green', 'darkorange', 'purple',
                         'brown', 'magenta', 'teal', 'olive', 'black']

        # --- Y轴格式设置 ---
        ax.set_yscale('linear')  # 确保使用线性坐标

        # 禁用科学计数法（例如显示 40000 而不是 4e4）
        # plain 表示不使用偏移量和科学计数法
        ax.ticklabel_format(style='plain', axis='y')

        # 如果数值非常大，上面的 plain 可能失效，强制使用标量格式化
        y_formatter = ScalarFormatter(useOffset=False)
        y_formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(y_formatter)

        for i, key in enumerate(keys):
            data = processed_data[key]
            line_color = color_palette[i % len(color_palette)]

            # 绘制平滑曲线
            ax.plot(data['x'], data['y'], linewidth=1.5, color=line_color, label=key)

        ax.set_xlabel('Episodes')
        ax.set_ylabel('Value')

        # 注意：这里删除了原先错误的 ax.set_yscale('log') 调用，防止覆盖线性设置

        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(loc='best', fontsize='medium')

        name = picture_name if picture_name else 'training_curves_combined'
        save_path = os.path.join(RESULT_DIR, name + '.png')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        print(f"[Plotter] Combined comparison curve updated at {save_path}")

    # ================= 模式2：单独展示 (每个数据独立存为一张图片) =================
    else:
        for i, key in enumerate(keys):
            fig, ax = plt.subplots(figsize=(6, 4))
            data = processed_data[key]

            # 获取颜色
            line_color = plt.cm.tab10(i % 10)

            ax.plot(data['x'], data['y'], linewidth=2.5, color=line_color)

            # 单独展示时也建议检查是否需要禁用科学计数法
            if np.max(data['y']) > 1000:
                ax.ticklabel_format(style='plain', axis='y')

            ax.set_title(f"{key} (ws={window_size.get(key, 1) if isinstance(window_size, dict) else window_size})")
            ax.set_xlabel('Episodes')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.6)

            plt.tight_layout()
            safe_filename = key.replace('/', '_').replace(' ', '_') + '.png'
            save_path = os.path.join(RESULT_DIR, safe_filename)
            plt.savefig(save_path, dpi=100)
            plt.close(fig)

        print(f"[Plotter] {num_metrics} individual curve images updated.")