#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/7/17 0017 14:45
# @Author  : ZhengHao
# @File    : Record_Data.py

import os
import pandas as pd
from datetime import datetime
from typing import List, Dict
import numpy as np
import shutil  # 导入用于删除目录的库


class Record_Experimental_Data:
    """
    一个专门用于记录和保存实验数据的类。
    - 每次初始化（即每次实验运行）都会创建一个唯一的时间戳文件夹。
    - 支持在同一次运行中，多次将数据追加到同一个CSV文件中。
    """

    def __init__(self, env, base_log_dir: str = "experiment_results"):
        """
        初始化记录器。

        Args:
            env: 实验所使用的环境实例，用于获取 uav_num, agent_num 等信息。
            base_log_dir (str): 存储所有实验结果的根目录。
        """
        self.env = env
        self.base_log_dir = base_log_dir

        # # 1. 检查目标目录是否存在
        # if os.path.exists(self.base_log_dir):
        #     print(f"INFO: Found existing log directory '{self.base_log_dir}'. Removing it to start fresh...")
        #     try:
        #         # 2. 如果存在，则递归删除整个目录树
        #         shutil.rmtree(self.base_log_dir)
        #         print(f"INFO: Successfully removed previous log directory.")
        #     except OSError as e:
        #         print(f"ERROR: Failed to remove directory {self.base_log_dir}. Reason: {e}")
        #         # 如果删除失败，程序可以继续，但可能会写入旧目录，或在创建时报错

        # 3. 创建一个全新的、空的目录
        # 无论之前是否存在，这一步都会确保一个空目录被创建
        # os.makedirs(self.base_log_dir, exist_ok=True)
        # print(f"INFO: Created a new, empty log directory: '{self.base_log_dir}'")

        # 4. 初始化一个列表，用于累积所有要记录的数据
        self.agent_data: List[Dict] = []

        self.uav_data: List[Dict] = []

        # 5. 新增一个集合，用于跟踪已经写入过表头的文件
        self._headers_written = set()

    def log_agent_data(self, episode: int):
        """
        记录当前 episode 中所有智能体的最终状态和合同数据。

        Args:
            episode (int): 当前的训练轮次（episode number）。
        """
        # 遍历环境中的每一个智能体
        for i, agent in enumerate(self.env.Agents):
            # 创建一个字典来存储该智能体在当前回合的数据行
            contract = np.hstack([agent.uav_r_list, agent.uav_u_list]).flatten()
            # 使用 np.round 四舍五入
            contracts = '[' + ",".join(str(np.round(x, 3)) for x in agent.uav_r_list) + \
                        "  ,  " + ",".join(str(np.round(x, 3)) for x in agent.uav_u_list) + ']'
            record = {
                'episode': episode,
                'agent_id': i,
                'agent_utility': np.round(agent.utility, 3),
                'contracts_selected_count': np.sum(agent.P_State_List),
                'agent_contract': contracts
            }

            # 将这条记录添加到结果列表中
            self.agent_data.append(record)

    def log_uav_data(self, episode: int):
        """
        记录当前 episode 中所有智能体的最终状态和合同数据。

        Args:
            episode (int): 当前的训练轮次（episode number）。
        """
        # 遍历环境中的每一个智能体
        for i, uav in enumerate(self.env.UAVs):
            # 创建一个字典来存储该智能体在当前回合的数据行
            uav_utility = uav.utility_list
            total_utility = np.round(np.sum(uav_utility), 3)
            select_agent = uav.select_Agent
            # 直接转换所有元素为字符串并用逗号连接
            utility = '[' + ",".join(str(np.round(x, 3)) for x in uav_utility[:len(select_agent)]) + ']'
            agents = '[' + ",".join(str(x) for x in select_agent) + ']'
            record = {
                'episode': episode,
                'uav_id': i,
                'total_utility': total_utility,
                'uav_utility_list': utility,
                'select_agent': agents
            }
            # 将这条记录添加到结果列表中
            self.uav_data.append(record)

    def save_to_csv(self, type, filename: str = "converged_results.csv"):
        """
        将所有已记录的数据转换为pandas DataFrame，并保存为CSV文件。

        Args:
            filename (str): 要保存的CSV文件名。
        """
        results = []
        if type == 'agent':
            results = self.agent_data
            self.agent_data = []
        elif type == 'uav':
            results = self.uav_data
            self.uav_data = []

        if not results:
            print("WARNING: No data to save.")
            return

        # 构造完整的文件路径
        save_path = os.path.join(self.base_log_dir, filename)

        # 将结果列表转换为 DataFrame
        df = pd.DataFrame(results)

        # 检查是否已经为这个文件写入过表头
        write_header = filename not in self._headers_written

        # 使用 mode='a' (append) 来追加数据
        # 保存为 CSV 文件，不包含 pandas 的行索引
        df.to_csv(save_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
        # 标记这个文件已经写过表头了
        if write_header:
            self._headers_written.add(filename)
