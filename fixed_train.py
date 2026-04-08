from UsualFunctions import LOG, CommonFun
import numpy as np
configDict = CommonFun.ReadConfig('config.txt')
def decode_ppo_to_ru(raw_action,env):
    """uav_num, R_min, R_max, P_min, P_max, DELTA
    根据 PPO 生成的原始动作获取 R 序列和 U 序列
    raw_action: PPO 输出的 numpy 数组 [a1, a2, ..., aK, ap]
    """
    uav_num = env.uav_num
    R_min = 20
    R_max = 50
    P_min = env.UAVs[0].total_energy
    P_max = 1.1*env.UAVs[uav_num-1].total_energy
    DELTA = env.DELTA
    # 1. 分离动作：前 K 位给 R，最后 1 位给 P
    raw_r_part = raw_action[:uav_num]
    raw_p_val = raw_action[-1]

    # 2. 获取单价 P (映射到 [P_min, P_max])
    # 使用 tanh 将无界输出压到 [-1, 1]，再线性映射
    unit_price = ((np.tanh(raw_p_val) + 1) / 2) * (P_max - P_min) + P_min

    # 3. 重构 R 序列 (确保单调性: R1 >= R2 >= ... >= RK)
    R_k = np.zeros(uav_num)

    # a) 确定最低档 R_K (能耗最高的那类)
    # 映射到 [R_min, R_max*0.3] 的基准范围
    base_val = (np.tanh(raw_r_part[0]) + 1) / 2
    R_k[-1] = R_min + (R_max * 0.3 - R_min) * base_val

    # b) 向上累加增量 (Delta)，确保 R_k >= R_{k+1}
    # 使用 exp 或 softplus 保证增量为正
    deltas = np.log(1 + np.exp(raw_r_part[1:])) * DELTA

    for k in range(uav_num - 2, -1, -1):
        # 索引对应：deltas[0] 是 R_{K-1} 的增量
        R_k[k] = R_k[ k +1] + deltas[uav_num - k - 2]

    # c) 裁剪并取整
    R_k = np.round(np.clip(R_k, R_min, R_max))

    # 4. 根据固定定价公式获取 U 序列
    # 核心公式：报酬 = 单价 * 任务量
    U_k = R_k * unit_price

    return R_k, U_k, unit_price


def run_fixed_train(ppo,env,state):
    batch_size = configDict['BATCH']
    buffers = {
        'states': [],
        'actions': [],
        'rewards': [],
        'next_states': [],
        'log_probs': []
    }
    for step_i in range(batch_size):
        actions, log_probs = ppo.take_action(state)
        R_k, U_k, unit_price = decode_ppo_to_ru(actions,env)
        for i,uav in enumerate(env.UAVs):
            if U_k[i] < uav.total_energy * R_k[i]:
                # 不满足IR不接受
                U_k[i], R_k[i] = 0, 0

        multi_reward, next_state, contracts,acceptance_rate = env.step_2(R_k, U_k,2)

        buffers['states'].append(state)
        buffers['actions'].append(actions)
        buffers['next_states'].append(next_state)
        buffers['rewards'].append(np.mean(multi_reward))
        buffers['log_probs'].append(log_probs)

        state = next_state

    ppo.update(buffers)

    return np.mean(multi_reward),[R_k,U_k], unit_price,acceptance_rate
