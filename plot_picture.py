import numpy as np
import os
import matplotlib
import math
import pandas as pd
import seaborn as sns
import os
import pickle
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
    # (简单策略：前面几个点用原始数据填充，或者用逐渐增大的窗口平均)
    padding = data[:window_size - 1]
    return np.concatenate([padding, smoothed])


def plot_learning_curves(metrics_dict, current_episode,mode, window_size=20):
    """
    改进版绘图函数：
    1. 自适应子图布局
    2. 双线绘制 (Raw + Smooth)
    3. 自动 X 轴推断
    4. 自动处理不等长嵌套数据（使用末尾元素补齐）
    """
    if not metrics_dict:
        return

    # 1. 确定指标数量和布局
    num_metrics = len(metrics_dict)
    if num_metrics == 0:
        return

    # 自动计算列数和行数 (最多3列)
    cols = 3 if num_metrics >= 3 else num_metrics
    rows = math.ceil(num_metrics / cols)

    # 创建画布
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    fig.suptitle(f'Training Status (Episode {current_episode})', fontsize=16)

    # 统一处理 axes 为列表，方便遍历 (处理只有1个子图的情况)
    if num_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    keys = list(metrics_dict.keys())

    # 2. 遍历绘制每个指标
    for i, key in enumerate(keys):
        ax = axes[i]
        data_list = metrics_dict[key]

        if not data_list:
            continue

        # 【核心修正】: 强制安全地转换为 numpy array
        try:
            # 1. 先尝试将列表中的元素都取 .item()（如果是 tensor 或 numpy 标量）
            clean_list = [x.item() if hasattr(x, 'item') else float(x) for x in metrics_dict[key]]
            # 2. 转换为 numpy array
            raw_data = np.array(clean_list, dtype=np.float32)
        except Exception as e:
            print(f"[Plot Error] Could not process metric '{key}'. Data format is invalid. Error: {e}")
            continue  # 跳过这个画不出来的图

        # ==================== 核心修改区域 ====================
        # 步骤 1：寻找当前指标数据中最长的子序列长度
        lengths = [len(item) if isinstance(item, (list, tuple, np.ndarray)) else 1 for item in data_list]
        if not lengths:
            continue
        max_len = max(lengths)

        # 步骤 2：使用最后一个元素对齐补全短数据
        padded_data = []
        for item in data_list:
            if not isinstance(item, (list, tuple, np.ndarray)):
                item_list = [item]  # 将单个数字包装成列表
            else:
                item_list = list(item)

            # 如果遇到空列表，用 NaN 填满；否则用原列表最后一个元素补齐至 max_len
            if len(item_list) == 0:
                item_list = [np.nan] * max_len
            elif len(item_list) < max_len:
                last_element = item_list[-1]
                padding_length = max_len - len(item_list)
                item_list.extend([last_element] * padding_length)

            padded_data.append(item_list)

        # 步骤 3：转换为规范的 float 数组并展平
        raw_data = np.array(padded_data, dtype=np.float64).flatten()
        # ====================================================

        # 检查是否为空
        if len(raw_data) == 0:
            continue

        # 安全处理 NaN
        if np.isnan(raw_data).any():
            raw_data = np.nan_to_num(raw_data)

        # 动态计算当前指标的 X 轴 (解决展平后数据长度变化的问题)
        # 将展平后的总步数均匀映射到当前的 Episode 进度上
        x_axis = np.linspace(0, current_episode, len(raw_data))

        # A. 绘制原始数据 (浅色，透明度高)
        ax.plot(x_axis, raw_data, alpha=0.3, color='gray', label='Raw')

        # B. 绘制平滑数据 (深色，主趋势)
        # 根据数据长度动态调整平滑窗口，防止初期数据太少报错
        real_window = min(window_size, len(raw_data)) if len(raw_data) > 1 else 1
        smoothed_data = smooth_data(raw_data, real_window)

        # 使用不同的颜色循环
        line_color = plt.cm.tab10(i % 10)
        ax.plot(x_axis, smoothed_data, linewidth=2, color=line_color, label=f'Smooth (w={real_window})')

        # C. 装饰图表
        ax.set_title(key.replace('_', ' ').title())
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.6)

        # 只在第一个图显示图例，避免遮挡
        if i == 0:
            ax.legend(loc='best', fontsize='small')

    # 3. 删除多余的空子图
    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    # 4. 保存
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 留出标题空间
    name = mode + '_' + 'training_curves.png'
    save_path = os.path.join(RESULT_DIR, name)
    plt.savefig(save_path, dpi=100)

    plt.close(fig)  # 极其重要：关闭图像释放内存
    print(f"[Plotter] Curves updated at {save_path}")


def plot_ic_verification(env, utility_matrix, node_type='DN', save_dir="./results/plots"):
    """
    绘制 IC 验证图，并优化数据展示。
    """
    os.makedirs(save_dir, exist_ok=True)

    # 获取节点数量
    N = len(env.DN_list) if node_type == 'DN' else len(env.CN_list)

    # =========================================================
    # 【1. 样式与颜色配置区】 - 你可以在这里修改颜色和大小
    # =========================================================
    # 颜色：使用 'tab10' 或 'Set1' 这种区分度大的色系
    color_map = plt.cm.get_cmap('tab10')
    line_colors = [color_map(i % 10) for i in range(N)]

    line_width = 2.0  # 线条粗细 (数值越大越粗)
    marker_size = 9  # 节点标记点大小 (o, s, ^ 等)
    highlight_size = 280  # 红圈的大小 (数值越大圈越大)
    highlight_width = 2  # 红圈线条的粗细
    title_size = 15  # 标题字体大小
    label_size = 12  # 坐标轴标签大小
    # =========================================================

    if node_type == 'DN':
        title = "Incentive Compatibility (IC) Verification for Data Nodes"
        xlabel = "Contract Menu Index (Sorted by $D_n$)"
    else:
        title = "Incentive Compatibility (IC) Verification for Compute Nodes"
        xlabel = "Contract Menu Index (Sorted by $\\beta_m$)"

    # =========================================================
    # 【2. 数据优化处理区】 - 让曲线看起来更“正比例”，不被极小值拉长
    # =========================================================

    # =========================================================
    # 【3. 绘图逻辑】
    # =========================================================
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'x', '+']

    for i in range(N):
        # 使用原始数据找最大值，但为了处理“相同”的情况，进行精度缩放
        # np.isclose 结合 np.where 可以精准找到第一个匹配最大值的索引
        y_values_raw = utility_matrix[i, :]
        max_val = np.max(y_values_raw)

        # 核心修改：找到所有接近最大值的索引，取第一个 [0]
        best_indices = np.where(np.isclose(y_values_raw, max_val, atol=1e-4))[0]
        best_menu_idx = best_indices[0]

        # 绘图时使用优化后的显示数据
        y_display = utility_matrix[i, :]

        # 画折线
        ax.plot(range(N), y_display,
                marker=markers[i % len(markers)],
                markersize=marker_size,
                color=line_colors[i],
                linewidth=line_width,
                label=f"True Type {i}",
                alpha=0.85)

        # 在“第一个”最大值位置画红圈
        # 注意：如果最大值被截断了，圈会画在截断处，但通常 IC 验证点都是正值
        ax.scatter(best_menu_idx, y_display[best_menu_idx],
                   s=highlight_size,
                   facecolors='none',
                   edgecolors='red',
                   linewidths=highlight_width,
                   zorder=10)

    # 装饰
    ax.set_title(title, fontsize=title_size, fontweight='bold', pad=15)
    ax.set_xlabel(xlabel, fontsize=label_size)
    ax.set_ylabel("Utility ($U$)", fontsize=label_size)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"Menu {j}" for j in range(N)])
    ax.grid(True, linestyle='--', alpha=0.5)

    # 图例放在右侧
    ax.legend(title="Node True Type", bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'IC_Verification_{node_type}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"[{node_type}] IC plot saved to {save_path} (First Max Circled)")

    # =========================================================
    # 【4. 终端打印】 (严格显示哪一个是第一个最大值)
    # =========================================================
    print(f"\n=== {node_type} Utility Matrix ( * = Marked First Max ) ===")
    for i in range(N):
        row_raw = utility_matrix[i, :]
        max_v = np.max(row_raw)
        idx_first = np.where(np.isclose(row_raw, max_v, atol=1e-4))[0][0]

        row_str = f"Node {i:<2} | "
        for j in range(N):
            val = row_raw[j]
            if j == idx_first:
                row_str += f"{val:>7.1f}* | "
            else:
                row_str += f"{val:>8.1f} | "
        print(row_str)



class ICDataLogger:
    def __init__(self, base_dir="results/ic_verification"):
        self.base_dir = base_dir
        self.csv_dir = os.path.join(base_dir, "csv")
        self.plot_dir = os.path.join(base_dir, "plots")
        self.consolidated_file = os.path.join(base_dir, "all_valid_matrices.pkl")

        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        # 如果是重新训练，清空或初始化合并文件
        self.successful_records = []

    def log_matrix(self, matrix, node_type, episode):
        """
        保存单个 Episode 的矩阵为 CSV 和 热力图
        """
        num_nodes = matrix.shape[0]
        # 1. 转换为 DataFrame 增加可读性
        df = pd.DataFrame(
            matrix,
            index=[f"Node_{i}(Type)" for i in range(num_nodes)],
            columns=[f"Contract_{j}" for j in range(num_nodes)]
        )

        # 2. 保存为 CSV
        csv_path = os.path.join(self.csv_dir, f"ep{episode}_{node_type}_utility.csv")
        df.to_csv(csv_path)

        # 3. 绘制热力图并保存
        plt.figure(figsize=(8, 6))
        sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
        plt.title(f"Utility Matrix {node_type} - Episode {episode}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"ep{episode}_{node_type}_heatmap.png"))
        plt.close()

    def save_to_consolidated(self, dn_matrix, cn_matrix, episode):
        """
        将满足 IC 的矩阵存入合并列表
        """
        record = {
            'episode': episode,
            'dn_matrix': dn_matrix,
            'cn_matrix': cn_matrix
        }
        self.successful_records.append(record)

        # 定期持久化保存到磁盘（防止程序崩溃丢失数据）
        with open(self.consolidated_file, 'wb') as f:
            pickle.dump(self.successful_records, f)

    # --- 在 main 循环中的调用方式 ---
    # 初始化 logger (在循环外)


        # 2. 加入合并数据集
