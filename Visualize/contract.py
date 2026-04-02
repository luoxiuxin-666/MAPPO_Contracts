import math
import numpy as np
import pandas as pd
from collections import namedtuple
from itertools import combinations_with_replacement
import csv

# 定义合同和类型的数据结构
# 为UAV类型增加 gamma_k 参数，代表发布者对该类型精度的敏感度
Contract = namedtuple("Contract", ["effort_R", "reward_U"])
UAVType = namedtuple("UAVType", ["name", "theta", "prob", "gamma"])


class UAVCluster:
    """代表一个无人机集群（代理人），行为与之前版本相同。"""

    def __init__(self, uav_type: UAVType):
        self.uav_type = uav_type

    def calculate_utility(self, contract: Contract) -> float:
        return contract.reward_U - self.uav_type.theta * contract.effort_R

    def choose_contract(self, contract_menu: list) -> (Contract, float):
        max_utility = -np.inf
        chosen_contract = None
        options = contract_menu + [Contract(effort_R=0, reward_U=0)]

        for contract in options:
            utility = self.calculate_utility(contract)
            if utility > max_utility:
                max_utility = utility
                chosen_contract = contract

        return chosen_contract, max_utility


class MultiTypeTaskPublisher:
    """
    支持多种类型代理人的任务发布者，并使用理论公式计算效用。
    """

    def __init__(self, uav_types: list, possible_efforts: list, L, eta, zeta):
        # 按照成本系数 theta 从低到高（效率从高到低）排序
        self.uav_types = sorted(uav_types, key=lambda x: x.theta)
        self.possible_efforts = sorted(possible_efforts)
        self.num_types = len(uav_types)
        # 存储模型参数
        self.L = L
        self.eta = eta
        self.zeta = zeta

    def calculate_contribution_S(self, effort_R: float, uav_type: UAVType) -> float:
        """
        根据理论公式，计算任务发布者从一个类型的集群获得的价值 S_k。
        """
        if effort_R <= 0:
            return 0.0

        # 步骤 1: 从努力 R 计算精度误差 ε
        epsilon = math.exp(-(1 - self.eta) * effort_R / self.zeta)

        # 步骤 2: 从精度 (1-ε) 计算价值 S
        accuracy = 1 - epsilon
        S_k = self.L * math.log(1 + uav_type.gamma * accuracy)

        return S_k

    def design_optimal_contracts_and_log(self, log_filename='exploration_log.csv') -> (dict, float, int):
        """
        为多种类型设计最优合同菜单, 同时将每一步探索过程记录到CSV文件中。
        返回: 最优合同字典, 最大效用, 找到最优解的轮次
        """
        best_expected_utility = -np.inf
        optimal_contracts = {}
        optimal_round_num = 0  # <--- 新增: 记录最优解所在的轮次

        # 1. 在探索开始前计算并打印总探索次数
        n = len(self.possible_efforts)
        r = self.num_types
        total_combinations = math.comb(n + r - 1, r)
        print(f"总共需要探索的次数: {total_combinations}")

        effort_combinations = combinations_with_replacement(self.possible_efforts, self.num_types)

        # 2. 打开CSV文件并准备写入
        with open(log_filename, 'w', newline='', encoding='utf-8') as csvfile:
            log_writer = csv.writer(csvfile)
            # 写入表头
            log_writer.writerow(['轮次', '合同设计', '发布者效用'])

            for round_num, effort_tuple in enumerate(effort_combinations, 1):
                R_vector = sorted(effort_tuple, reverse=True)

                # ... [计算 U_vector 和 utilities 的代码保持不变] ...
                U_vector = [0.0] * self.num_types
                utilities = [0.0] * self.num_types

                last_idx = self.num_types - 1
                theta_last = self.uav_types[last_idx].theta
                R_last = R_vector[last_idx]
                utilities[last_idx] = 0
                U_vector[last_idx] = theta_last * R_last

                for i in range(self.num_types - 2, -1, -1):
                    theta_i = self.uav_types[i].theta
                    R_i = R_vector[i]
                    utility_if_mimic = utilities[i + 1] + (self.uav_types[i + 1].theta - theta_i) * R_vector[i + 1]
                    utilities[i] = utility_if_mimic
                    U_vector[i] = utilities[i] + theta_i * R_i

                # 计算发布者的期望效用
                current_expected_utility = 0
                for i in range(self.num_types):
                    prob_i = self.uav_types[i].prob
                    uav_type_i = self.uav_types[i]
                    R_i = R_vector[i]
                    U_i = U_vector[i]

                    contribution_S_i = self.calculate_contribution_S(R_i, uav_type_i)
                    publisher_utility_i = contribution_S_i - U_i
                    current_expected_utility += prob_i * publisher_utility_i

                # --- 将当前探索步骤的数据写入CSV ---
                r_rounded = [round(r, 3) for r in R_vector]
                u_rounded = [round(u, 3) for u in U_vector]
                contract_design_str = str(r_rounded + u_rounded)
                utility_rounded = round(current_expected_utility, 3)
                log_writer.writerow([round_num, contract_design_str, utility_rounded])
                # -----------------------------------

                # 更新最优解
                if current_expected_utility > best_expected_utility:
                    best_expected_utility = current_expected_utility
                    optimal_round_num = round_num  # <--- 新增: 更新最优解的轮次
                    optimal_contracts = {
                        self.uav_types[i].name: Contract(effort_R=R_vector[i], reward_U=U_vector[i])
                        for i in range(self.num_types)
                    }

        return optimal_contracts, best_expected_utility, optimal_round_num


# --- 主程序：模拟与执行 (五种类型) ---
if __name__ == "__main__":
    # --- 模型超参数 (根据您的理论模型定义) ---
    L_PARAM = 800.0  # 价值函数 S_k 的缩放系数
    ETA_PARAM = 0.15  # η: 数据异构性参数 (0 <= η < 1)
    ZETA_PARAM = 10.0  # ζ: 反映学习任务难度的常数

    # 1. 定义市场环境 (五种类型，比例相同)
    thetas = [0.586, 0.628, 0.682, 0.748, 0.825]
    gammas = [0.8, 0.8, 0.8, 0.8, 0.8]
    num_types = len(thetas)

    UAV_TYPES = [
        UAVType(name=f"类型{i + 1}", theta=thetas[i], prob=1.0 / num_types, gamma=gammas[i])
        for i in range(num_types)
    ]

    POSSIBLE_EFFORTS = list(range(37, 50, 1))

    # 2. 初始化任务发布者
    publisher = MultiTypeTaskPublisher(
        UAV_TYPES, POSSIBLE_EFFORTS,
        L=L_PARAM, eta=ETA_PARAM, zeta=ZETA_PARAM
    )

    # 3. 设计最优合同并记录过程, 接收所有返回值
    print("正在设计最优合同菜单并记录探索过程到 exploration_log.csv ...")
    optimal_menu_dict, max_publisher_utility, optimal_round = publisher.design_optimal_contracts_and_log()

    # 4. 按照指定格式打印最终结果
    print(f"\n# 最优合同在第 {optimal_round} 轮探索中找到")
    print("# 最后生成的合同")
    print("# 设计的合同及对应的效用分析 (基于理论公式):")
    header = (
        f"{'集群类型':<6} {'成本系数 (θ)':<12} {'敏感度 (γ)':<11} {'设计的任务量 (R)':<16} "
        f"{'设计的报酬 (U)':<15} {'集群获得的净效用':<16} {'发布者价值 (S_k)':<15} {'发布者净效用':<15}"
    )
    print(header)

    optimal_r_list = []
    optimal_u_list = []

    # 必须遍历发布者内部排序后的 uav_types 列表以保证顺序
    for uav_type in publisher.uav_types:
        contract = optimal_menu_dict[uav_type.name]

        # 提取合同参数
        R = contract.effort_R
        U = contract.reward_U

        # 计算各项指标
        cluster_net_utility = U - uav_type.theta * R
        publisher_value_sk = publisher.calculate_contribution_S(R, uav_type)
        publisher_net_utility = publisher_value_sk - U

        # 存储最优 R 和 U 用于最后打印
        optimal_r_list.append(R)
        optimal_u_list.append(U)

        # 格式化输出行
        # 注意：为了完全匹配您的示例输出，此处格式化为4位小数
        row_str = (
            f" {uav_type.name:<7} {uav_type.theta:<10.3f} {uav_type.gamma:<12.1f} {R:<14} "
            f"{U:<15.4f} {cluster_net_utility:<15.4f} {publisher_value_sk:<15.4f} {publisher_net_utility:<15.4f}"
        )
        print(row_str)

    # 准备并打印最终的最优合同设计向量
    # 注意：为了完全匹配您的示例输出，此处格式化为4位小数
    final_u_list_formatted = [f"{u:.4f}" for u in optimal_u_list]
    final_design_str = str(optimal_r_list + final_u_list_formatted)
    # 移除字符串中的引号
    final_design_str = final_design_str.replace("'", "")

    print(f"# 最优合同设计：{final_design_str}，发布者总效用: {max_publisher_utility:.4f}")