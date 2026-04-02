import numpy as np
from scipy.optimize import differential_evolution


class TraditionalContractOptimizer:
    """
    传统合同优化类：使用差分进化算法寻找满足 IC/IR 约束的最优合同。
    用于作为 MAPPO 算法的对比基准（Baseline）。
    """

    def __init__(self, env):
        self.env = env
        self.uav_num = env.uav_num
        self.UAVs = env.UAVs
        # 假设类型概率分布均匀，总和为 1
        self.p_k = np.full(self.uav_num, 1.0 / self.uav_num)

    def _calculate_S_k(self, R_k, k_idx):
        """
        计算任务贡献度 S_k (对应专利公式)
        S_k = L * w * ln(1 + gamma * (1 - epsilon))
        """
        epsilon_k = np.exp(-((1 - self.env.eta) * R_k) / self.env.zat)
        # 传统方法中，假设聚合权重为均值 1/K
        weight = 1.0 / self.uav_num
        S_k = self.env.L * weight * np.log(1 + self.env.gamma_k * (1 - epsilon_k))
        return S_k

    def _get_incentive_U(self, R_vector):
        """
        核心逻辑：根据合同理论递推公式计算报酬 U，确保 IC 和 IR 约束处于边界。
        确保高效率集群不流向低效率合同。
        U_K = R_K * E_K
        U_k = U_{k+1} + (R_k - R_{k+1}) * E_k
        """
        U_vector = np.zeros(self.uav_num)
        # 获取按照能耗升序排序后的集群能耗值
        energies = np.array([uav.total_energy for uav in self.UAVs])

        # 1. 确定最低档合同(能耗最高)的报酬，使其满足 IR 边界（不亏钱）
        U_vector[-1] = R_vector[-1] * energies[-1]

        # 2. 向上递推，确定高效率合同的报酬，使其满足 IC 边界（不撒谎）
        for k in range(self.uav_num - 2, -1, -1):
            U_vector[k] = U_vector[k + 1] + (R_vector[k] - R_vector[k + 1]) * energies[k]

        return U_vector

    def objective_function(self, R_vector):
        """
        优化器目标函数：最大化 Agent 的总期望效用 Σ p_k * (S_k - U_k)
        """
        # 修改点：不再返回惩罚值，而是直接排序，引导优化器收敛
        # 将随机生成的 R 向量强制解释为单调递减序列：R1 >= R2 >= ... >= RK
        R_sorted = np.sort(R_vector)[::-1]

        U_vector = self._get_incentive_U(R_sorted)
        total_utility = 0
        for k in range(self.uav_num):
            S_k = self._calculate_S_k(R_sorted[k], k)
            total_utility += self.uav_num * self.p_k[k] * (S_k - U_vector[k])

        # differential_evolution 执行的是最小化，所以返回负的效用
        return -total_utility

    def solve(self):
        """
        使用差分进化算法求解最优 R 序列。
        """
        print("Traditional Optimizer is calculating the best contract...")

        # 定义搜索边界：每个 R_k 都在 [R_min, R_max] 之间
        bounds = [(5,50)] * self.uav_num

        # 执行优化
        result = differential_evolution(
            self.objective_function,
            bounds,
            strategy='best1bin',
            popsize=15,
            tol=0.01,
            mutation=(0.5, 1),
            recombination=0.7,
            seed=42  # 固定随机种子以保证结果可重复
        )

        # 后处理：获取排序并取整后的 R
        best_R = np.round(np.sort(result.x)[::-1])

        # 计算对应的报酬 U 和最大效用
        best_U = self._get_incentive_U(best_R)
        best_utility = -result.fun

        return best_R, best_U, best_utility
# --- 测试代码 ---
if __name__ == '__main__':
    from MAPPO_Contract_Env import Multi_Contract_Environment
    from UsualFunctions import LOG, CommonFun
    config = CommonFun.ReadConfig('config.txt')
    env = Multi_Contract_Environment(config)

    optimizer = TraditionalContractOptimizer(env)
    r_best, u_best, util = optimizer.solve()

    print("\n" + "=" * 30)
    print("TRADITIONAL OPTIMAL CONTRACT")
    print(f"Optimal R vector: {r_best}")
    print(f"Optimal U vector: {u_best}")
    print(f"Agent Max Utility: {util:.4f}")
    print("=" * 30)