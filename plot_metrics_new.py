#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/5 0005 9:39
# @Author  : ZhengHao
# @File    : plot_metrics.py
import numpy as np
import os
import matplotlib

# 设置后端为 'Agg'，必须在导入 pyplot 之前
# 这会阻止 Matplotlib 尝试创建任何 GUI 窗口
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# 创建结构化的结果目录
env_name = "contract_mappo"
result_base_dir = "results"
result_dir = os.path.join(result_base_dir, env_name)
plots_dir = os.path.join(result_dir, "plots")
# 确保目录存在
os.makedirs(plots_dir, exist_ok=True)


# 函数签名恢复为两个参数
def plot_all_metrics(metrics_dict, episode):
    """
    将所有指标绘制到一个包含多个子图的图表中。
    - 直接绘制原始数据点，无平滑。
    - 横坐标根据传入的当前 episode 和数据点数量自动计算。
    - 图像直接保存，不显示。

    参数:
    metrics_dict: 包含所有指标数据的字典，格式为 {metric_name: values_list}
    episode: 当前的总 episode 数
    """
    # 检查字典是否为空
    if not metrics_dict:
        print("Warning: metrics_dict is empty. Nothing to plot.")
        return

    # --- 关键修改: 自动计算横坐标 ---
    # 从字典中任意取出一个指标列表来获取数据点的数量
    any_metric_values = next(iter(metrics_dict.values()))
    num_data_points = len(any_metric_values)

    # 如果没有数据点，则直接返回
    if num_data_points == 0:
        print("Warning: No data points to plot.")
        return

    # 计算记录间隔 (假设是等间隔的)
    log_interval = episode / num_data_points

    # 根据间隔和数据点数量，生成横坐标刻度
    # 例如：如果间隔是50，有3个点，则生成 [50, 100, 150]
    episode_ticks = np.arange(1, num_data_points + 1) * log_interval

    # 创建一个2x3的子图布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Training Metrics of {env_name} (Up to Episode {episode})', fontsize=16)

    # 压平axes数组以便迭代
    axes = axes.flatten()

    # 在每个子图中绘制一个指标
    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        if i >= len(axes):
            break

        ax = axes[i]

        # 使用自动计算出的 episode_ticks 作为横坐标
        ax.plot(episode_ticks, values, '-')

        ax.set_title(metric_name.replace('_', ' '))
        ax.set_xlabel('Episodes')
        ax.set_ylabel(metric_name.replace('_', ' '))
        ax.grid(True, alpha=0.5)

    # 删除未使用的子图
    for j in range(len(metrics_dict), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = os.path.join(plots_dir, f'mappo_training_metrics.png')
    plt.savefig(save_path)
    plt.close(fig)
