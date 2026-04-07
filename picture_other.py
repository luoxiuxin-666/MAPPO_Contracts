import matplotlib.pyplot as plt
import numpy as np
import os

# 确保保存目录存在
RESULT_DIR = "results/plots"
os.makedirs(RESULT_DIR, exist_ok=True)


def plot_ic_verification(utility_matrix):
    """
    绘制激励相容(IC)验证曲线图
    :param utility_matrix: 5x5 的效用矩阵 (numpy array 或 list)
    """
    # 转换为 numpy 数组方便处理
    data = np.array(utility_matrix)
    num_nodes = data.shape[0]
    num_contracts = data.shape[1]

    # --- 样式设置 (解决报错的关键) ---
    # 尝试使用几种常用样式，如果都没有，则使用默认样式并手动开启网格
    available_styles = plt.style.available
    if 'seaborn-v0_8-whitegrid' in available_styles:
        plt.style.use('seaborn-v0_8-whitegrid')
    elif 'seaborn-whitegrid' in available_styles:
        plt.style.use('seaborn-whitegrid')
    elif 'ggplot' in available_styles:
        plt.style.use('ggplot')

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.grid(True, linestyle='--', alpha=0.7)  # 强制开启网格确保美观

    # 定义颜色和标记
    colors = ['#0033ff', '#ff0000', '#008000', '#ff9900', '#800080']
    markers = ['s', 's', 's', 's', 's']  # 使用方块标记

    x = np.arange(1, num_contracts + 1)  # 合同类型 1-5

    for i in range(num_nodes):
        row_data = data[i]
        label = f"Type-{i + 1} Cluster"

        # 寻找当前集群在所有合同中的最大值及其索引
        max_val = np.max(row_data)
        max_idx = np.argmax(row_data)  # 第一个最大值的索引

        # 绘制折线
        ax.plot(x, row_data, label=label, color=colors[i],
                marker=markers[i], markersize=7, linewidth=1.5)

        # --- 标注最大值 ---
        # 1. 在最大值处画一个圆圈 (空心圆)
        ax.scatter(x[max_idx], max_val, s=250, facecolors='none',
                   edgecolors=colors[i], linewidths=2, zorder=5)

        # 2. 添加箭头和文字 "Maximum profit"
        # 根据 y 轴数值微调偏移量
        offset_y = (np.max(data) - np.min(data)) * 0.05
        ax.annotate('Maximum profit',
                    xy=(x[max_idx], max_val),
                    xytext=(x[max_idx] - 0.3, max_val + offset_y),
                    fontsize=10,
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='black', connectionstyle="arc3"))

    # 设置图表属性
    ax.set_xlabel('Contract Types', fontsize=14, fontweight='bold')
    ax.set_ylabel('Profit of Worker Nodes', fontsize=14, fontweight='bold')
    ax.set_title('Incentive Compatibility (IC) Verification', fontsize=16, pad=20)

    # 设置坐标轴
    ax.set_xticks(x)
    # 动态调整 y 轴范围，留出标注空间
    ax.set_ylim(np.min(data) - 2, np.max(data) + 5)

    # 设置图例 (右下角，白色背景)
    ax.legend(loc='lower right', frameon=True, fontsize=11, facecolor='white', edgecolor='black')

    # 布局优化
    plt.tight_layout()

    # 保存文件 (先保存再 show)
    save_path = os.path.join(RESULT_DIR, 'ic_comparison.pdf')
    plt.savefig(save_path, bbox_inches='tight', dpi=300, format='pdf')
    print(f"Figure saved to: {save_path}")

    # 显示图片
    plt.show()
    plt.close(fig)


# --- 测试调用 ---
if __name__ == "__main__":
    # 效用列表数据
    utility_list = [
        [16.81516308 , 16.81516308  ,15.54116308 , 13.55716308,  11.80216308],
        [9.35316308,  11.35516308 , 11.35516308 , 10.82716308,   9.61816308],
    [6.64716308,
    9.37516308,
    9.83716308,
    9.83716308,
    8.82616308],
    [-7.16983692, - 0.73483692   ,2.08616308  , 4.78216308  , 4.78216308],
    [-14.54983692, - 6.13483692, - 2.05383692,
    2.08216308,
    2.62216308]
    ]

    plot_ic_verification(utility_list)
    avg_uti = {
        'Contract(proposed)': [1000,970,900, 680, 560],
        'Contract(static)': [980,870,580, 480, 360],
        'Fixed_Pricing': [920,700, 470, 320, 355]
    }
    #
    # plot_ic_verification(utility_list)