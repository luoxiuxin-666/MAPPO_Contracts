import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from UsualFunctions import LOG, CommonFun
from MAPPO_Contract.MAPPO_Contract_Env import Multi_Contract_Environment
import pandas as pd
import os
import visualize_grouped_data
configDict = CommonFun.ReadConfig('../config.txt')
env = Multi_Contract_Environment(configDict)


def calculate_profit(contract, save_path: str):
    """
    模拟计算一个特定类型的工作节点在选择不同类型的合同时所获得的收益。
    这个函数的设计是为了体现“激励相容性”(IC)。
    """
    # 计算每个类型对应的合同所能获得的所有的效用
    mid_len = int(len(contract) / 2)
    contract = np.array(contract)
    utility_data = []
    for uav in env.UAVs:
        R_list = contract[:mid_len]
        U_list = contract[mid_len:]
        uav_uti = U_list - R_list * uav.total_energy
        utility_data.append(np.round(uav_uti,3))

    print(utility_data)
    # --- 1. 初始化图表和参数 ---
    # 使用一个美观的样式
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 10))

    # 定义颜色列表，以便每个类型的线条颜色都不同
    colors = ['blue', 'red', 'green', 'darkorange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # 确定合同的数量（x轴的范围）
    num_contracts = len(utility_data[0])
    contract_types_x_axis = np.arange(1, num_contracts + 1)

    # --- 2. 遍历数据并绘图 ---
    # 使用 enumerate 来同时获取索引和数据，索引用于区分不同类型的节点
    for worker_index, profit_array in enumerate(utility_data):
        # 确保颜色列表足够长，如果不够就循环使用
        color = colors[worker_index % len(colors)]

        # 绘制当前类型节点的收益曲线
        ax.plot(
            contract_types_x_axis,
            profit_array,
            marker='s',  # 方形标记
            linestyle='-',
            color=color,
            label=f'Type-{worker_index + 1} worker node'  # 标签从 Type-1 开始
        )

        # --- 3. 查找峰值并添加标注  ---
        # 找到当前收益曲线的最大值（峰值）
        peak_profit = np.max(profit_array)

        # 找到当前工作节点对应的“理想合同”的索引
        # 假设Type-1对应索引0，Type-2对应索引1，以此类推
        ideal_contract_index = worker_index

        # 找到所有收益等于峰值的合同索引
        all_max_indices = np.where(profit_array == peak_profit)[0]

        # **核心判断逻辑**
        if ideal_contract_index in all_max_indices:
            # 如果理想合同的索引是最大值之一，优先选择它
            peak_contract_index = ideal_contract_index
        else:
            # 否则，使用第一个找到的最大值索引
            peak_contract_index = all_max_indices[0]

        # 获取最终用于标注的X轴坐标
        peak_contract_type = contract_types_x_axis[peak_contract_index]

        # 添加带箭头的 "Maximum profit" 标注
        ax.annotate(
            'Maximum profit',
            xy=(peak_contract_type, peak_profit),  # 箭头指向的点（峰值点）
            xytext=(peak_contract_type, peak_profit + 1),  # 文本放置的位置（在峰值点上方）
            arrowprops=dict(facecolor='black', shrink=0.01, width=1, headwidth=3),
            fontsize=10,
            fontweight='bold',
            ha='center'  # 水平居中对齐
        )

    # --- 4. 美化图表 ---
    ax.set_title('Incentive Compatibility (IC) Verification', fontsize=16)
    ax.set_xlabel('Contract Types', fontsize=14)
    ax.set_ylabel('Profit of Worker Nodes', fontsize=14)

    # 确保x轴刻度为整数
    ax.set_xticks(contract_types_x_axis)
    ax.tick_params(axis='both', which='major', labelsize=12)
    # 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    # # 显示图例
    ax.legend(
        fontsize=10,
        loc='lower right',
        frameon = True,  # 开启边框
        fancybox = True,  # 使用圆角
        framealpha = 0.9,  # 90%不透明度
        facecolor = 'whitesmoke',  # 背景色
        edgecolor = 'gray'  # 边框颜色
    )

    # 调整整体布局，为下方的图例留出空间
    fig.tight_layout(rect=[0, 0.05, 1, 1])


    output_filename = os.path.join(save_path, "compare_contract_utilities.png")
    plt.savefig(output_filename, dpi=300)
    print(f"-> Plot saved to '{output_filename}'")
    plt.close()

def agent_utilies_compare(data, path_dir:str):
    """
       根据给定的数据（字典列表），绘制一个专业的柱状图。

       Args:
           data (list of dict): 一个列表，其中每个元素都是一个字典，
                                格式为 {'agents': 数量, 'avg_utility': 平均效用}。
       """
    # --- 1. 数据准备 ---
    # 将字典列表转换为 Pandas DataFrame，这让数据处理更简单
    df = pd.DataFrame(data)

    # 为了保证X轴有序，我们按 'agents' 列进行排序
    df = df.sort_values(by='agents').reset_index(drop=True)

    print("用于绘图的数据:")
    print(df)

    # --- 2. 开始绘图 ---
    # 设置绘图样式，使其接近学术论文风格 (使用衬线字体)
    # 如果您需要显示中文，'SimSun' (宋体) 是一个好选择。
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制柱状图
    # x轴数据是 agent 数量，y轴数据是平均效用
    bar_width = 0.5
    ax.bar(
        df['agents'],
        df['avg_utility'],
        width=bar_width,
        label='Task publisher'  # 对应您参考图的图例标签
    )

    # --- 3. 图表美化 ---
    ax.set_ylabel('Profit', fontsize=14)
    ax.set_xlabel('Number of task publishers', fontsize=14)

    # 设置X轴刻度为整数，确保显示为 1, 2, 3, 4, 5
    ax.set_xticks(df['agents'])

    # 设置Y轴的范围，可以根据您的数据最大值进行调整
    # 例如，如果最大值是830，我们可以设置到900或1000，让顶部留有空间
    max_utility = df['avg_utility'].max()
    ax.set_ylim(0, max_utility * 1.2)

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.legend(fontsize=12)

    # 添加底部的图说 (Caption)
    caption = "(a) Profit comparison of task publishers in different scenarios"
    fig.text(
        0.5, 0.01, caption,
        ha='center', va='bottom',
        fontsize=14, style='italic'
    )

    # 调整布局，为底部的图说留出空间
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    output_filename = os.path.join(path_dir, "avg_agent_utilies_compare.png")
    plt.savefig(output_filename, dpi=300)
    print(f"-> Plot saved to '{output_filename}'")
    plt.close()

def agent_utilies_compare_2(data, path_dir:str):
    """
        使用多种技巧增强柱状图的视觉对比效果。
        """
    # --- 1. 数据准备 ---
    df = pd.DataFrame(data).sort_values(by='agents').reset_index(drop=True)

    # --- 2. 开始绘图 ---
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 7))

    # --- 方法3: 使用颜色渐变 ---
    # 创建一个颜色映射，从深蓝到浅蓝
    colors = cm.get_cmap('Blues_r')(np.linspace(0.2, 0.8, len(df)))

    # 绘制柱状图
    bar_width = 0.5
    bars = ax.bar(
        df['agents'],
        df['avg_utility'],
        width=bar_width,
        label='Task publisher',
        color=colors  # 应用渐变颜色
    )

    # --- 方法2: 在柱顶添加数值标签 ---
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,  # X轴位置
            yval + 5,  # Y轴位置 (在柱顶上方一点)
            f'{yval:.0f}',  # 要显示的文本 (格式化为整数)
            ha='center', va='bottom',  # 对齐方式
            fontsize=12,
            fontweight='bold'
        )

    # --- 方法4: 增加趋势线 ---
    ax.plot(
        df['agents'],
        df['avg_utility'],
        color='red',
        marker='o',
        linestyle='--',
        linewidth=2,
        label='Trend'
    )

    # --- 3. 图表美化 ---
    ax.set_ylabel('Profit', fontsize=14)
    ax.set_xlabel('Number of task publishers', fontsize=14)
    ax.set_xticks(df['agents'])
    ax.tick_params(axis='both', which='major', labelsize=12)

    # --- 方法1: 调整Y轴的显示范围 ---
    # 找到数据的最小值和最大值
    min_val = df['avg_utility'].min()
    max_val = df['avg_utility'].max()
    # 设置Y轴范围，在数据范围上下留出一些空白
    ax.set_ylim(min_val * 0.98, max_val * 1.05)

    # 显示图例 (现在有两个标签了)
    ax.legend(fontsize=12)

    # 添加图说
    caption = "(a) Profit comparison of task publishers in different scenarios"
    fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=14, style='italic')

    output_filename = os.path.join(path_dir, "avg_agent_utilies_compare_2.png")
    plt.savefig(output_filename, dpi=300)
    print(f"-> Plot saved to '{output_filename}'")
    plt.close()
if __name__ == '__main__':
    root_path = visualize_grouped_data.PLOT_DIR
    contract = [61.0,52.0,49.0,48.0,46.0,44.0,42.0,40.0,38.0,35.0,67.765,62.491,60.607,59.925,58.429,56.779,54.959,52.947,50.727,47.061]
    uav_path = os.path.join(root_path, "uav_utility")
    calculate_profit(contract,uav_path)

    # 集群为10，比较同等资源下的效用对比
    agent_utility = [
        {'agents': 1,'avg_utility':808.0},
        {'agents': 2,'avg_utility':807.7},
        {'agents': 3,'avg_utility':786.6},
        {'agents': 4,'avg_utility':726},
        # 未记录
        {'agents': 5,'avg_utility':680}
    ]
    agent_path = os.path.join(root_path, "agent_utility")
    agent_utilies_compare(agent_utility,agent_path)
    agent_utilies_compare_2(agent_utility,agent_path)
    # 根据历史数据获得
    # 计算每个集群在所有情况下能够获取的总效用的记录。
    # 从做到右依次为智能体为：1~5的情况。
    #例如{'uav_1':1, 'utilities':[31.557,64.471,72.58,74.832]},
    #为集群1在智能体为1~5个的情况下所获取的效用为[31.557,64.471,72.58,74.832]
    uav_utilities_compare= [
        {'uav_1':1, 'utilities':[31.557,64.471,72.58,74.832]},
        {'uav_1':2, 'utilities':[29.331,59.935,66.742,68.658]},
        {'uav_1':3, 'utilities':[26.685,54.427,59.776,61.962]},
        {'uav_1':4, 'utilities':[23.583,47.959,51.922,54.636]},
        {'uav_1':5, 'utilities':[20.041,40.798,43.144,47.013]},
        {'uav_1':6, 'utilities':[16.386,33.403,34.304,39.618]},
        {'uav_1':7, 'utilities':[12.354,25.339,25.184,32.514]},
        {'uav_1':8, 'utilities':[8.298, 17.019,16.136,25.962]},
        {'uav_1':9, 'utilities':[4.154,8.731,7.288,20.026]},
        {'uav_1':10, 'utilities':[0.006,0.313,1.798,15.146]},
    ]