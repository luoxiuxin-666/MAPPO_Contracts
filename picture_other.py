import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.lines import Line2D
from data_manager import ExperimentDataManager

# 确保保存目录存在
RESULT_DIR = "results/plots"
os.makedirs(RESULT_DIR, exist_ok=True)


def plot_ic_verification(utility_matrix):
    """
    绘制激励相容(IC)验证曲线图
    :param utility_matrix: 5x5 的效用矩阵 (numpy array 或 list)
    """
    data = np.array(utility_matrix)
    num_nodes = data.shape[0]
    num_contracts = data.shape[1]

    # --- 健壮的样式设置 ---
    available_styles = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    else:
        plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.grid(True, linestyle='--', alpha=0.6)

    # 定义线条颜色和标记
    colors = ['#0033ff', '#ff0000', '#008000', '#ff9900', '#800080']
    markers = ['s', 's', 's', 's', 's']

    # --- 统一最大值圆圈颜色 ---
    circle_color = '#333333'  # 稳重的深灰色/黑色

    x = np.arange(1, num_contracts + 1)

    # 存储图例句柄
    handles = []

    for i in range(num_nodes):
        row_data = data[i]
        label = f"Type-{i + 1} Computing power node"

        # 寻找最大值及其索引
        max_val = np.max(row_data)
        max_idx = np.argmax(row_data)

        # 绘制折线并保存句柄用于图例
        line, = ax.plot(x, row_data, label=label, color=colors[i],
                        marker=markers[i], markersize=7, linewidth=1.5)
        handles.append(line)

        # --- 统一使用深灰色绘制圆圈标注最大值 ---
        ax.scatter(x[max_idx], max_val, s=350, facecolors='none',
                   edgecolors=circle_color, linewidths=2.5, zorder=10)

    # --- 在图例中添加“圆圈 = Maximum Profit” ---
    # 使用与图中圆圈完全一致的颜色 (circle_color)
    max_profit_proxy = Line2D([0], [0], marker='o', color='none',
                              markeredgecolor=circle_color, markersize=15,
                              markeredgewidth=2.5, label='Maximum Profit')
    handles.append(max_profit_proxy)

    # 设置图表属性
    ax.set_xlabel('Contract Types', fontsize=14, fontweight='bold')
    ax.set_ylabel('Utility of Computing power node', fontsize=14, fontweight='bold')
    ax.set_title('Incentive Compatibility (IC) Verification', fontsize=16, pad=20)

    ax.set_xticks(x)
    # 动态调整 y 轴范围，确保最高点不被遮挡
    ax.set_ylim(np.min(data) - 2, np.max(data) + 3)

    # 设置图例：包含 5 个节点条目 + 1 个最大值说明条目
    ax.legend(handles=handles, loc='lower right', frameon=True,
              fontsize=11, facecolor='white', edgecolor='black')

    plt.tight_layout()

    # 保存为 PDF
    save_path = os.path.join(RESULT_DIR, 'ic_verification_final.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"Figure saved to: {save_path}")

    # 显示图片
    plt.show()
    plt.close(fig)


def plot_avg_utility_comparison(avg_uti_data):
    """
    绘制不同激励机制下发布者平均效用的对比图
    :param avg_uti_data: 包含三种机制数据的字典
    """
    # 设置全局样式
    # plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 7))

    # 定义 X 轴：发布者个数 1-5
    x = np.arange(1, 6)

    # 按照要求定义颜色：红、蓝、绿
    colors = ['red', 'blue', 'green']
    # 定义标记样式：圆圈、方块、三角
    markers = ['o', 's', '^']

    # 获取数据键名并绘制
    mechanisms = list(avg_uti_data.keys())

    for i, mech in enumerate(mechanisms):
        ax.plot(x, avg_uti_data[mech],
                label=mech,
                color=colors[i],
                marker=markers[i],
                markersize=8,
                linewidth=2,
                alpha=0.8)

    # 设置图表属性
    # ax.set_title('Average Utility Comparison of Different Incentive Mechanisms', fontsize=14, pad=15)
    ax.set_xlabel('Number of Publishers (Agents)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Utility', fontsize=12, fontweight='bold')

    # 设置 X 轴刻度
    ax.set_xticks(x)

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)

    # 设置图例
    ax.legend(loc='best', frameon=True, fontsize=11, edgecolor='black')

    # 布局优化
    plt.tight_layout()

    # 保存为 PDF
    save_path = os.path.join(RESULT_DIR, 'avg_utility_comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"Figure saved to: {save_path}")

    # 显示图片
    plt.show()
    plt.close(fig)


# --- 测试数据 ---
if __name__ == "__main__":
    utility_list = [
        [40.51504706 , 40.51504706 , 34.99931438,   32.07831831 ,  24.6392678],
        [27.18200562 , 28.39932721,  28.39932721 ,  23.43882893 ,  18.47946192],
    [6.30729227, 8.61,15.8,15.7,12.31],
    [-5.92630238, - 5.40459313, - 3.40009808 ,   6.16   , 6.1],
    [-10.54530931, - 9.84969698, - 7.3429544 ,- 1.78,0.]
    ]
    plot_ic_verification(utility_list)
    # avg_uti = {
    #     'Contract(proposed)': [1000,970,900, 700, 590],
    #     'Contract(static)': [980,870,580, 480, 360],
    #     'Fixed_Pricing': [920,700, 470, 390, 325]
    # }
    # # #
    # plot_avg_utility_comparison(avg_uti)
    # data_manager = ExperimentDataManager(save_dir="results/plots")
    # # 2. 读取数据
    # loaded_results = data_manager.load_metrics("all_valid_matrices")

    # total_uti = loaded_results['total_uti']