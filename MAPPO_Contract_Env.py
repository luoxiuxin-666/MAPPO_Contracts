#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/6/21 0021 10:11
# @Author  : ZhengHao
# @File    : mappo_env_change2.py


import numpy as np

# 固定随机种子
rng = np.random.default_rng()

class UAV:
    """定义UAV（无人机集群）对象。"""

    def __init__(self, uav_id, total_energy, E_u):
        self.uav_id = uav_id
        self.total_energy = total_energy  # E_k^tot: 成本参数
        self.resource_base = 0
        self.resource_limit = 0   # E_k^all: 资源上限
        self.E_u = E_u

        self.select_Agent = []
        self.utility_list = []


class Agent:
    """定义Agent（任务发布者）对象。"""

    def __init__(self, agent_id, uav_num):
        self.agent_id = agent_id

        # --- 状态向量 ---
        self.IR_State_List = np.zeros(uav_num, dtype=int)
        self.Monotonicity_State = np.zeros(uav_num - 1, dtype=int)
        self.P_State_List = np.zeros(uav_num, dtype=int)  # 合同被选择的状态

        # --- 合同内容 ---
        self.uav_r_list = np.zeros(uav_num)
        self.uav_u_list = np.zeros(uav_num)

        # --- 效用和奖励 ---
        self.utility = 0
        self.reward = 0

        self.delta_k = np.full(uav_num, 1.0 / uav_num)


class Multi_Contract_Environment:
    """
    为多任务发布者设计的合同制定强化学习环境。
    融合了动作空间修正、多智能体竞争和自适应课程学习。
    """

    def __init__(self, configDict):
        # --- 加载配置 ---
        # 服务器配置参数，计算贡献
        self.L = float(configDict['L'])
        self.gamma_k = float(configDict['gamma_k'])

        # 计算全局精度
        self.eta = float(configDict['eta'])
        self.zat = float(configDict['zat'])
        self.zat_2 = configDict['zat_2']

        # 智能体个数和集群个数
        self.agent_num = int(configDict['agent_num'])
        self.uav_num = int(configDict['uav_num'])

        # 集群参数配置，计算能耗
        # 本地计算能耗信息
        self.f_k_range = configDict['f_k']
        self.D_n = configDict['D_n']
        self.c_n = configDict['c_n']
        self.ksi = configDict['ksi']
        self.N = configDict['N']

        # 本地通信能耗信息
        self.v_1 = configDict['v_1']
        self.B_1_range = configDict['B_1']
        self.rho_k_range = configDict['rho_k']
        self.h_k_range = configDict['h_k']
        self.N_0 = configDict['N_0']

        # 无人机通信能耗
        self.v_2 = configDict['v_2']
        self.B_2_range = configDict['B_2']
        self.rho_u_range = configDict['rho_u']
        self.h_u_range = configDict['h_u']

        # 无人机传递参数给服务器的能耗
        self.v_3 = configDict['v_3']
        self.B_3_range = configDict['B_3']
        self.h_u2_range = configDict['h_u2']

        # 无人机悬停和聚合能耗
        self.e_h = configDict['e_h']
        self.e_c = configDict['e_c']

        # 其他限制条件
        self.U_max = configDict['U_max']
        self.R_max = configDict['R_max']
        self.R_min = configDict['R_min']
        self.R_last = configDict['R_last']
        self.E_all = configDict['E_all']

        # --- 动作和状态空间定义 ---
        self.action_dim = self.uav_num + 1
        self.state_dim = self.uav_num * 2 + (self.uav_num - 1)  # IR(K) + P(K) + Mono(K-1)

        # --- UAV 和 Agent 初始化 ---

        f_k_list = rng.integers(self.f_k_range[0], self.f_k_range[1],self.uav_num)
        print(f" the f_k_list is {f_k_list}")

        total_range = self.f_k_range[1] - self.f_k_range[0]

        # R5的基准值范围，例如让它在 [10, 30] 之间
        self.R_BASE_RANGE = [self.R_min, self.R_last]
        # 间隔值的缩放因子，控制R之间的差距大小
        self.DELTA = configDict.get('DELTA')

        self.UAVs = []
        self.Agents = []
        # 计算所以集群的能耗，并按照升序排序
        E_tot_list = []
        for i in range(self.uav_num):
            # 获取随机数值
            # 平均CPU消耗功率
            # f_k = rng.integers(self.f_k_range[0], self.f_k_range[1])
            f_k = f_k_list[i]
            # 集群的传输带宽
            B_1_k = rng.integers(self.B_1_range[0], self.B_1_range[1])
            # 集群的功率
            rho_k = np.round(rng.uniform(self.rho_k_range[0],self.rho_k_range[1]), 4)
            # 集群信道增益
            h_k = rng.integers(self.h_k_range[0],self.h_k_range[1])

            # 无人机到集群的带宽
            B_2_k = rng.integers(self.B_2_range[0], self.B_2_range[1])
            # 无人机到集群的功率
            rho_u = np.round(rng.uniform(self.rho_u_range[0], self.rho_u_range[1]), 2)
            # 无人机信道增益,为了方便直接使用h_k
            h_u = h_k

            # 每个集群随机获取参数并计算能耗
            total_energy = self.Calculate_total_energy(f_k, B_1_k, rho_k, h_k, B_2_k, rho_u, h_u)

            E_tot_list.append(total_energy)

        # 计算无人机补偿的能耗
        self.E_u = self.Calculate_E_u()

        # 将集群能耗按照升序排序
        E_tot_list.sort()
        print(f"the E_tot_list is {E_tot_list}")
        print(f"the E_u is {self.E_u}")

        for i in range(1,self.uav_num+1):
            uav = UAV(f'uav_{i}', E_tot_list[i-1], self.E_u)
            uav.resource_limit = E_tot_list[i-1]*self.agent_num * 30
            uav.resource_base = uav.resource_limit
            self.UAVs.append(uav)
        for i in range(1,self.agent_num+1):
            self.Agents.append(Agent(f'agent_{i}', self.uav_num))

        # 奖惩权重
        self.REWARD_SUCCESS = 100.0
        self.PENALTY_INVALID_IN_PHASE2 = -200.0

    def _reconstruct_r_from_action(self, raw_action):
        """
        Args:
            raw_action (np.array): 5维的原始动作向量，来自高斯分布。
                                   语义: (a_R5_base, a_delta_4, a_delta_3, a_delta_2, a_delta_1)
        Returns:
            np.array: 解码后的5维R向量，满足 R_k >= R_{k+1} 且在 [R_MIN, R_MAX] 范围内。
        """
        raw_action = raw_action[:-1]
        R_k = np.zeros(self.uav_num)

        # 1. 解码 R_5
        # 使用 tanh 将第一个动作值压到(-1, 1)，然后线性映射到预设范围
        # tanh(a) -> (-1, 1)  => (tanh(a)+1)/2 -> (0, 1)
        base_val_normalized = (np.tanh(raw_action[0]) + 1) / 2
        R_k[self.uav_num - 1] = self.R_BASE_RANGE[0] + (self.R_BASE_RANGE[1] - self.R_BASE_RANGE[0]) * base_val_normalized

        # 2. 解码间隔值
        # 间隔必须是正数，使用 softplus(x) = log(1+exp(x)) 是一个比exp()更稳定的选择
        # softplus 函数可以将任何实数映射到 (0, +inf)
        deltas = np.log(1 + np.exp(raw_action[1:])) * self.DELTA

        # 3. 从后向前递推计算 R_4, R_3, R_2, R_1
        # R_4 = R_5 + delta_4, R_3 = R_4 + delta_3, ...
        # raw_action[1] is a_delta_4, raw_action[2] is a_delta_3, etc.
        # deltas[0] corresponds to a_delta_4, deltas[1] to a_delta_3...
        for k in range(self.uav_num - 2, -1, -1):
            # deltas的索引是 0,1,2,3 -> 对应 R_4, R_3, R_2, R_1 的间隔
            # k 的索引是 3,2,1,0
            R_k[k] = R_k[k + 1] + deltas[self.uav_num - k - 2]

        # 4. 最后一步：裁剪 (Clip)
        # 确保所有R值都严格落在[R_MIN, R_MAX]的范围内
        R_k = np.round(np.clip(R_k, self.R_min, self.R_max))

        return R_k

    def dbm_to_watts(self, dbm):
        """
        将功率值从 dBm 转换为瓦 (W)。
        """
        return (10 ** (dbm / 10)) / 1000

    def Calculate_total_energy(self, f_k, B_1_k, rho_k, h_k, B_2_k, rho_u, h_u):
        h_k_w = self.dbm_to_watts(h_k)
        h_u_w = self.dbm_to_watts(h_u)
        # 计算本地训练能耗
        E_cmp = self.N * self.ksi * self.c_n * self.D_n * f_k**2
        T_cmp = (self.c_n * self.D_n) / f_k
        # 计算集群的传输能耗
        T_com = self.v_1 / (B_1_k * np.log2(1+((rho_k * h_k_w)/(B_1_k*self.N_0))))
        E_com = self.N * T_com * rho_k
        # 计算无人机的传输能耗
        T_u_com = self.v_2 / (B_2_k * np.log2(1 + ((rho_u * h_u_w)/(B_2_k * self.N_0))))
        E_u_com = self.N * T_u_com * rho_u
        # 计算本地迭代次数
        local_r = np.round(self.zat_2 * np.log(1/self.eta))
        # local_r = np.round(np.log(1/self.eta))
        # 计算总能耗
        print(f" h_k is {h_k_w}; h_u is {h_u_w}; E_cmp is {E_cmp}; T_cmp is {T_cmp}; T_com is {T_com}; \n"
              f"E_com is {E_com}; T_u_com is {T_u_com}; E_u_com is {E_u_com}; local_r is {local_r}")
        E_tot = local_r * E_cmp + E_cmp + E_u_com + (local_r*T_cmp + T_com + T_u_com) * self.e_h + self.e_c
        return np.round(E_tot, 3)

    def Calculate_E_u(self):
        # 无人机到服务器的带宽
        B_3 = rng.integers(self.B_3_range[0], self.B_3_range[1])
        # 无人机到服务器的信道增益
        h_u2 = rng.integers(self.h_u2_range[0],self.h_u2_range[1])
        # 无人机到服务器的传输功率
        rho_u2 = np.round(rng.uniform(self.rho_u_range[0], self.rho_u_range[1]), 2)

        h_u2_w = self.dbm_to_watts(h_u2)

        T_u = self.v_3 / (B_3 * np.log2(1+((rho_u2*h_u2_w)/(B_3*self.N_0))))

        E_u = T_u * rho_u2 + T_u * self.e_h

        return np.round(E_u, 3)

    def _calculate_u_from_r(self, R_k, incentive_U):
        """根据R向量和理论公式，计算U向量。"""
        U_k = np.zeros_like(R_k)
        # tanh(a) -> (-1, 1) -> (0, 5)
        incentive_boost = ((np.tanh(incentive_U) + 1) / 2) * 5.0  # 5.0是超参数
        last_idx = self.uav_num - 1
        U_k[last_idx] = R_k[last_idx] * self.UAVs[last_idx].total_energy
        for k in range(self.uav_num - 2, -1, -1):
            E_k_tot = self.UAVs[k].total_energy
            U_k[k] = U_k[k + 1] + (R_k[k] - R_k[k + 1]) * E_k_tot

        # 添加一个竞争激励
        U_final = U_k + incentive_boost
        return U_final

    def _check_constraints(self, agent):
        """检查R的IC和IR约束，更新agent的内部状态。"""
        R_k, U_k = agent.uav_r_list, agent.uav_u_list
        agent.Monotonicity_State = (R_k[:-1] > R_k[1:]).astype(int)
        energies = np.array([uav.total_energy for uav in self.UAVs])
        net_utilities = U_k - R_k * energies
        agent.IR_State_List = (net_utilities >= -1e-6).astype(int)

    def _uavs_select_contracts_dp(self):
        """
        [最优解] 使用动态规划求解选择问题，为每个UAV找到最优合同组合。
        """
        for agent in self.Agents:
            agent.P_State_List.fill(0)

        for k, uav in enumerate(self.UAVs):
            # 1. 收集该UAV的所有候选合同
            candidate_contracts = []
            for i, agent in enumerate(self.Agents):
                # 同样，只考虑有效的合同
                is_valid = np.all(agent.Monotonicity_State == 1) and agent.IR_State_List[k] == 1
                if is_valid:
                    candidate_contracts.append({
                        'agent_idx': i,
                        'R': int(agent.uav_r_list[k]),
                        'U': agent.uav_u_list[k]
                    })

            if not candidate_contracts:
                continue

            # 2. 求解问题
            n = len(candidate_contracts)
            capacity = int(np.floor(uav.resource_limit / uav.total_energy))
            # dp[i][j] 表示考虑前i个合同，能耗为j时的最大价值
            dp = [[0] * (capacity + 1) for _ in range(n + 1)]

            for i in range(1, n + 1):
                item_idx = i - 1
                weight = candidate_contracts[item_idx]['R']
                value = candidate_contracts[item_idx]['U']
                for j in range(1, capacity + 1):
                    if j < weight:
                        dp[i][j] = dp[i - 1][j]
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weight] + value)

            # 3. 回溯找到哪些合同被选中
            j = capacity
            for i in range(n, 0, -1):
                if dp[i][j] > dp[i - 1][j]:
                    item_idx = i - 1
                    selected_agent_idx = candidate_contracts[item_idx]['agent_idx']
                    self.Agents[selected_agent_idx].P_State_List[k] = 1
                    j -= candidate_contracts[item_idx]['R']

    def _get_one_agent_state(self, agent):
        """拼接得到单个智能体的完整状态向量。"""
        return np.hstack([agent.IR_State_List, agent.Monotonicity_State, agent.P_State_List]).astype(float)

    def Get_Multi_State(self):
        """获取所有智能体的状态列表。"""
        return [self._get_one_agent_state(agent) for agent in self.Agents]

    def Reset(self):
        """重置环境。"""
        for agent in self.Agents:
            agent.IR_State_List.fill(0)
            agent.Monotonicity_State.fill(0)
            agent.P_State_List.fill(0)
            agent.uav_r_list.fill(0)
            agent.uav_u_list.fill(0)
            agent.reward = 0
            agent.utility = 0
        for uav in self.UAVs:
            base = rng.integers(-10, 11)
            uav.resource_limit = uav.resource_base + base
        return self.Get_Multi_State(), {}

    def Step(self, multi_action):
        contracts = []
        # 1. 为每个智能体解码动作，并更新其内部状态
        for i, agent in enumerate(self.Agents):
            # a) 从原始动作重构出满足单调性的R向量
            raw_agent_action = multi_action[i]
            R_k = self._reconstruct_r_from_action(raw_agent_action)
            # b) 根据 R 计算 U
            U_k = self._calculate_u_from_r(R_k, multi_action[i][-1])

            # c) 保存 R 和 U
            agent.uav_r_list = R_k
            agent.uav_u_list = U_k
            contracts.append([R_k, U_k])
            # d) 检查 IR 约束
            self._check_constraints(agent)

        self._uavs_select_contracts_dp()

        multi_reward = self._compute_all_rewards()

        next_multi_state = self.Get_Multi_State()
        # is_valid = np.all(self.Agents[0].Monotonicity_State == 1) and np.all(self.Agents[0].IR_State_List == 1)
        # infos = { "r": self.Agents[0].uav_r_list, "u": self.Agents[0].uav_u_list,"valid_uav":is_valid}

        # 记录一下集群的选择以及效用
        self.record_UAVs_Utility()
        return multi_reward, next_multi_state, contracts

    def _compute_agent_utility(self, agent):
        """计算单个智能体的总效用，只计入被选中的合同。"""
        total_utility = 0
        contracts_chosen_count = np.sum(agent.P_State_List)

        if contracts_chosen_count == 0:
            return 0.0

        for k in range(self.uav_num):
            if agent.P_State_List[k] == 1:
                R, U = agent.uav_r_list[k], agent.uav_u_list[k]
                epsilon_k = np.exp(-((1 - self.eta) * R) / self.zat)
                S_k = self.L * np.log(1 + self.gamma_k * (1 - epsilon_k))
                total_utility += agent.delta_k[k] * (S_k - U)

        return total_utility

    def _compute_all_rewards(self):
        """根据当前课程阶段，计算所有智能体的奖励。"""
        multi_reward = []
        for agent in self.Agents:
            agent.utility = 0.0
            is_valid = np.all(agent.Monotonicity_State == 1) and np.all(agent.IR_State_List == 1)
            if is_valid:
                agent.utility = self._compute_agent_utility(agent)
                reward = self.REWARD_SUCCESS + agent.utility
            else:
                reward = self.PENALTY_INVALID_IN_PHASE2
            agent.reward = reward
            multi_reward.append(reward)
        return multi_reward


    def record_UAVs_Utility(self):
        '''激励一下每轮集群的选择，以及能耗'''
        for i, uav in enumerate(self.UAVs):
            uav.select_Agent = []
            uav.utility_list = []
            for j, agent in enumerate(self.Agents):
                if agent.P_State_List[i] == 1:
                    uav.select_Agent.append(j)
                    uav.utility_list.append(agent.uav_u_list[i] - agent.uav_r_list[i]*uav.total_energy)

# -----------------
# 独立测试环境的示例
if __name__ == '__main__':
    config = {
        'L': 1500, 'gamma_k': 0.8, 'eta': 0.15, 'zat': 10,
        'uav_num': 5, 'agent_num': 2
    }

    env = Multi_Contract_Environment(config)

    print(f"Action dim: {env.action_dim}, State dim: {env.state_dim}")
