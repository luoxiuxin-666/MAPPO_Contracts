#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/18 0018 9:34
# @Author  : ZhengHao
# @File    : visualize_grouped_data.py.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast  # 用于安全地将字符串 "[...]" 转换为列表
import matplotlib.lines as mlines # <<< 新增：导入用于创建代理线条的模块
import matplotlib.patheffects as path_effects
# --- 可配置参数 ---
# 注意：您的CSV文件名可能和截图中的列名不一样，请按实际情况修改

# --- 可配置参数 ---
# 指定包含CSV文件的实验结果文件夹路径
result_dir = "/results/contract_mappo/data/"
plot_dir = "/results/contract_mappo/plots/"
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.normpath(root_dir+result_dir)
PLOT_DIR = os.path.normpath(root_dir+plot_dir)
Agent_Utility = "agent_results.csv"
UAV_Utility = "uav_results.csv"
# 新增：移动平均的窗口大小。值越大，线条越平滑，但对变化的反应越慢。
SMOOTHING_WINDOW = 5
MARKER_SUBSAMPLING_RATE = 3

def plot_utility_over_time_smoothed(df: pd.DataFrame, save_path: str):
    """
    绘制平滑处理后的、颜色对比强烈的智能体效用趋势图。
    """
    print("Generating plot: Smoothed Utility Over Time...")

    plot_df = df.copy()
    plot_df['episode_scaled'] = plot_df['episode']
    # plot_df['episode_scaled'] = plot_df['episode'] / 100
    # plot_df['agent_id'] = plot_df['agent_id'].astype('category')

    # --- 1: 创建更具描述性的标签列 ---
    plot_df['agent_label'] = 'Agent_' + plot_df['agent_id'].astype(str)
    plot_df['agent_label'] = plot_df['agent_label'].astype('category')

    # --- 核心修改 1: 计算移动平均值 ---
    # 我们需要为每个agent独立计算移动平均，所以先按agent_id分组
    # min_periods=1 确保即使在窗口未满的开头，也能计算出平均值
    plot_df['utility_smoothed'] = plot_df.groupby('agent_id')['agent_utility'] \
        .transform(lambda x: x.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean())

    # --- 2: 定义高对比度的颜色 ---
    # 为每个agent_id手动指定一个清晰的颜色
    # 常见的组合: 蓝色, 橙色, 绿色
    custom_palette = {
        0: '#0072B2',  # 深蓝色
        1: '#CE0000',  # 橙色
        2: '#009100',  # 绿色
        3: '#A6A600',  # 绿色
        4: '#FF7F50',  # 绿色
    }

    # 定义标记样式
    marker_styles = ['o', 'X', '>', '3','*']
    linestyle_list = ['--', '--', '--', '--','--']

    # --- 定义颜色、标记和线型 ---
    num_agents = len(plot_df['agent_id'].unique())
    agent_ids = sorted(plot_df['agent_id'].unique())

    # --- 绘图 ---
    plt.figure(figsize=(16, 9))

    # a) 先用较细、半透明的线绘制原始的、抖动的数据作为背景
    sns.lineplot(
        data=plot_df,
        x='episode_scaled',
        y='agent_utility',
        hue='agent_id',
        palette=custom_palette,
        style='agent_id',
        legend=False,  # 不在背景上显示图例
        alpha=0.3,  # 设置透明度
        lw=1.5
    )

    # b) 再用较粗、不透明的线绘制平滑后的趋势数据
    # 绘制平滑后的趋势数据，并自定义标记
    sns.lineplot(
        data=plot_df,
        x='episode_scaled',
        y='utility_smoothed',
        hue='agent_id',
        palette=custom_palette,
        style='agent_id',
        # markers=marker_styles,  # <-- 指定形状
        # markersize=8,  # <-- 指定大小
        dashes=True,
        lw=3.0,
        legend=False,  # 不在背景上显示图例
        errorbar=None
    )

    # 绘制点
    for agent_id in plot_df['agent_id'].unique():
        agent_data = plot_df[plot_df['agent_id'] == agent_id]
        subsampled_data = agent_data.iloc[::MARKER_SUBSAMPLING_RATE]

        plt.plot(
            subsampled_data['episode_scaled'],
            subsampled_data['utility_smoothed'],
            marker=marker_styles[int(agent_id) % len(marker_styles)],
            linestyle='none',
            markersize=7,  # <-- 指定大小
            color=custom_palette[agent_id],
            label='_nolegend_'
        )

    # --- 手动创建并显示图例 ---
    legend_handles = []
    for agent_id, agent_label in zip(agent_ids, sorted(plot_df['agent_label'].unique())):
        handle = mlines.Line2D(
            [], [],
            color=custom_palette[agent_id],
            marker=marker_styles[int(agent_id) % len(marker_styles)],
            linestyle=linestyle_list[int(agent_id) % len(linestyle_list)],
            markersize=7,
            label=agent_label
        )
        legend_handles.append(handle)

    legend = plt.legend(
        handles=legend_handles,
        title='Agent ID',
        fontsize=14,
        title_fontsize=15,
        loc='best'
    )

    plt.title('Comparison of Agent Utilities Over Training (Smoothed Trend)', fontsize=20, pad=20)
    # plt.xlabel('Training Epoch (x100 Episodes)', fontsize=16)
    plt.xlabel('Training Epoch', fontsize=16)
    plt.ylabel('Agent Final Utility (Γi)', fontsize=16)

    plt.grid(
        axis='y',  # 只在Y轴上应用
        which='major',  # 只在主刻度上
        linestyle='--',  # 虚线样式
        linewidth=1.0,  # 线条加粗
        color='gray',  # 灰色
        alpha=0.6  # 半透明
    )
    # 我们可以选择性地关闭X轴的网格线，以突出Y轴
    plt.grid(axis='x', which='both', visible=False)  # b=False 表示不显示

    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_path_effects([path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.5)])

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    output_filename = os.path.join(save_path, "utility_trends_smoothed.png")
    plt.savefig(output_filename, dpi=300)
    print(f"-> Plot saved to '{output_filename}'")
    plt.close()

def plot_uav_total_utility(df: pd.DataFrame, save_path: str):
    """
       绘制平滑处理后的、颜色对比强烈的智能体效用趋势图。
       """
    print("Generating plot: Smoothed Utility Over Time...")

    plot_df = df.copy()
    plot_df['episode_scaled'] = plot_df['episode']

    # --- 1: 创建更具描述性的标签列 ---
    plot_df['uav_label'] = 'UAV_' + plot_df['uav_id'].astype(str)
    plot_df['uav_label'] = plot_df['uav_label'].astype('category')

    # --- 核心修改 1: 计算移动平均值 ---
    # 我们需要为每个agent独立计算移动平均，所以先按agent_id分组
    # min_periods=1 确保即使在窗口未满的开头，也能计算出平均值
    plot_df['utility_smoothed'] = plot_df.groupby('uav_id')['total_utility'] \
        .transform(lambda x: x.rolling(window=SMOOTHING_WINDOW, min_periods=1).mean())

    # --- 2: 定义高对比度的颜色 ---
    # 为每个agent_id手动指定一个清晰的颜色
    # 常见的组合: 蓝色, 橙色, 绿色
    custom_palette = {
        0: '#0072B2',  # 深蓝色
        1: '#D55E00',  # 橙色
        2: '#00FFFF',  # 绿色
        3: '#FF7F50',  # 绿色
        4: '#00868B',  # 绿色
        5: '#AE8F00',
        6: '#BB3D00',
        7: '#009100',
        8: '#000000',
        9: '#484891',
    }

    # 定义标记样式
    marker_styles = ['o', 'X', '>', '*', '3','+','d','s','h','p']
    linestyle_list = ['--', '--', '--', '--', '--','--', '--', '--', '--', '--']

    # --- 定义颜色、标记和线型 ---
    num_agents = len(plot_df['uav_id'].unique())
    uav_ids = sorted(plot_df['uav_id'].unique())

    # --- 绘图 ---
    plt.figure(figsize=(16, 9))

    # a) 先用较细、半透明的线绘制原始的、抖动的数据作为背景
    sns.lineplot(
        data=plot_df,
        x='episode_scaled',
        y='total_utility',
        hue='uav_id',
        palette=custom_palette,
        style='uav_id',
        legend=False,  # 不在背景上显示图例
        alpha=0.3,  # 设置透明度
        lw=1.5
    )

    # b) 再用较粗、不透明的线绘制平滑后的趋势数据
    # 绘制平滑后的趋势数据，并自定义标记
    sns.lineplot(
        data=plot_df,
        x='episode_scaled',
        y='utility_smoothed',
        hue='uav_id',
        palette=custom_palette,
        style='uav_id',
        # markers=marker_styles,  # <-- 指定形状
        # markersize=8,  # <-- 指定大小
        dashes=True,
        lw=3.0,
        legend=False,  # 不在背景上显示图例
        errorbar=None
    )

    # 绘制点
    for uav_id in plot_df['uav_id'].unique():
        uav_data = plot_df[plot_df['uav_id'] == uav_id]
        subsampled_data = uav_data.iloc[::MARKER_SUBSAMPLING_RATE]

        plt.plot(
            subsampled_data['episode_scaled'],
            subsampled_data['utility_smoothed'],
            marker=marker_styles[int(uav_id) % len(marker_styles)],
            linestyle='none',
            markersize=7,  # <-- 指定大小
            color=custom_palette[uav_id],
            label='_nolegend_'
        )

    # --- 手动创建并显示图例 ---
    legend_handles = []
    for uav_id, uav_label in zip(uav_ids, sorted(plot_df['uav_label'].unique())):
        handle = mlines.Line2D(
            [], [],
            color=custom_palette[uav_id],
            marker=marker_styles[int(uav_id) % len(marker_styles)],
            linestyle=linestyle_list[int(uav_id) % len(linestyle_list)],
            markersize=7,
            label=uav_label
        )
        legend_handles.append(handle)

    legend = plt.legend(
        handles=legend_handles,
        title='UAV ID',
        fontsize=14,
        title_fontsize=15,
        loc='best'
    )

    plt.title('Comparison of UAV Utilities Over Training (Smoothed Trend)', fontsize=20, pad=20)
    # plt.xlabel('Training Epoch (x100 Episodes)', fontsize=16)
    plt.xlabel('Training Epoch', fontsize=16)
    plt.ylabel('UAV Final Utility (Γi)', fontsize=16)

    plt.grid(
        axis='y',  # 只在Y轴上应用
        which='major',  # 只在主刻度上
        linestyle='--',  # 虚线样式
        linewidth=1.0,  # 线条加粗
        color='gray',  # 灰色
        alpha=0.6  # 半透明
    )
    # 我们可以选择性地关闭X轴的网格线，以突出Y轴
    plt.grid(axis='x', which='both', visible=False)  # b=False 表示不显示

    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_path_effects(
        [path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='gray', alpha=0.5)])

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    output_filename = os.path.join(save_path, "uav_utility_trends.png")
    plt.savefig(output_filename, dpi=300)
    print(f"-> Plot saved to '{output_filename}'")
    plt.close()


def plot_utility_distribution(df: pd.DataFrame, save_path: str):
    """
    使用箱形图展示每个智能体最终效用的分布情况。
    """
    print("Generating plot: Utility Distribution...")
    plt.figure(figsize=(10, 6))

    sns.boxplot(data=df, x='agent_id', y='agent_utility', palette='pastel')

    plt.title('Distribution of Final Utilities per Agent', fontsize=16)
    plt.xlabel('Agent ID', fontsize=12)
    plt.ylabel('Agent Utility (Γi)', fontsize=12)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    output_filename = os.path.join(save_path, "utility_distribution_boxplot.png")
    plt.savefig(output_filename, dpi=300)
    print(f"-> Plot saved to '{output_filename}'")
    plt.close()


def plot_utility_comparison_bar(df: pd.DataFrame, save_path: str):
    """
    使用条形图比较每个智能体的平均效用。
    """
    print("Generating plot: Average Utility Comparison...")
    summary = df.groupby('agent_id')['agent_utility'].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(10, 6))

    bars = plt.bar(summary['agent_id'], summary['mean'], yerr=summary['std'],
                   color=sns.color_palette('rocket', len(summary)), capsize=5, alpha=0.8)

    plt.title('Average Final Utility Comparison per Agent (with Std Dev)', fontsize=16)
    plt.xlabel('Agent ID', fontsize=12)
    plt.ylabel('Average Utility (Γi)', fontsize=12)
    plt.xticks(summary['agent_id'])
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}', va='bottom' if yval >= 0 else 'top',
                 ha='center')

    plt.tight_layout()

    output_filename = os.path.join(save_path, "utility_comparison_barchart.png")
    plt.savefig(output_filename, dpi=300)
    print(f"-> Plot saved to '{output_filename}'")
    plt.close()


def create_plot(type):
    """主执行函数"""
    csv_path = ""
    plots_dir = ""
    if type == 'agent':
        csv_path = os.path.join(RESULTS_DIR, Agent_Utility)
        plots_dir = os.path.join(PLOT_DIR, "agent_utility")
    elif type == 'uav':
        csv_path = os.path.join(RESULTS_DIR, UAV_Utility)
        plots_dir = os.path.join(PLOT_DIR, "uav_utility")

    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found at '{csv_path}'")
        return

    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully loaded data from '{csv_path}'. Shape: {df.shape}")
        # 筛选出 episode > 0 的数据，以防有初始状态的记录
        df = df[df['episode'] > 0].copy()
    except Exception as e:
        print(f"ERROR: Failed to read the CSV file. Reason: {e}")
        return

    if df.empty:
        print("WARNING: The CSV file is empty or contains no valid data. No plots will be generated.")
        return



    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nINFO: Plots will be saved in '{plots_dir}'\n")
    if type == 'agent':
        # --- 生成所有图表 ---
        plot_utility_over_time_smoothed(df, plots_dir)
        plot_utility_distribution(df, plots_dir)
        plot_utility_comparison_bar(df, plots_dir)
    elif type == 'uav':
        # 无人机
        plot_uav_total_utility(df, plots_dir)

    print("\nVisualization complete!")

if __name__ == '__main__':
    # 设置一个美观的全局绘图样式
    sns.set_theme(style="whitegrid", context="talk")
    create_plot('agent')
    create_plot('uav')