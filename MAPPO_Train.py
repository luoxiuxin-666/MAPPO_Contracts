#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/4/22 0022 11:11
# @File    : Multi_Contract_RL.py
import copy
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau
from traditional_contracts import TraditionalContractOptimizer
import numpy as np
import matplotlib.pyplot as plt
from UsualFunctions import LOG, CommonFun
from MAPPO_Contract_Env import Multi_Contract_Environment
from Record_Data import Record_Experimental_Data
from MAPPO import MAPPO
from collections import deque
import torch
import os
import shutil
from plot_picture_2 import plot_learning_curves_
from plot_picture import plot_learning_curves
import datetime
from fixed_ppo import PPO
from fixed_train import run_fixed_train
from data_manager import ExperimentDataManager

# 创建结构化的结果目录
result_base_dir = "results"
env_name = "contract_mappo"
result_dir = os.path.join(result_base_dir, env_name)
# 创建子目录
logs_dir = os.path.join(result_dir, "logs")
plots_dir = os.path.join(result_dir, "plots")
weights_dir = os.path.join(result_dir, "weights")
data_dir = os.path.join(result_dir, "data")

# 确保目录存在
for directory in [logs_dir, plots_dir, weights_dir, data_dir]:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

log = LOG()
log.LogInitialize()

def Log(message, Flag=True):
    mode = 'mappo'
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.LogRecord(f"mode {mode}:{message} - 时间：{current_time} ", Flag)

def Multi_Contract_Play():

    # 1.1 加载日志信息
    log = LOG()
    log.LogInitialize()
    log.LogRecord('FL and Multi Contract code is running now ......')

    log.LogRecord('Reading config.')
    configDict = CommonFun.ReadConfig('config.txt')
    log.LogRecord('Read config success!')

    # 1.2 其他信息
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device:", device)

    # 1.3 加载环境信息和智能体信息
    log.LogRecord(' Loading env now ......')
    env = Multi_Contract_Environment(configDict)
    record_data = Record_Experimental_Data(env, data_dir)
    state_dim = env.state_dim
    action_dim = env.action_dim
    total_episode = configDict['total_episode']
    actor_lr = configDict['actor_lr']
    critic_lr = configDict['critic_lr']
    gamma = configDict['GAMMA']
    lmbda = configDict['LAMBDA']
    eps = configDict['eps']
    batch_size = configDict['BATCH']
    tau_1 = configDict['tau_1']
    tau_2 = configDict['tau_2']
    agent_num = configDict['agent_num']
    K_epochs = configDict['K_epochs']
    R_min = configDict['R_min']
    R_max = configDict['R_max']
    R_RANGE = (R_min, R_max)
    Fixed_PPO = True
    Traditional_Contract = True

    # 使用双端队列来记录指标,最大元素为每一批次的步数
    avg_reward_per_episode = deque(maxlen=batch_size)
    avg_utility_per_episode = deque(maxlen=batch_size)

    # 记录每20轮的激励，用来判断学习率是否需要更新
    rewards_window = deque(maxlen=10)
    best_mean_reward = -float('inf')

    metrics_dict = {
        # "avg_agents_utility": [],
        "mappo_contract": [],
        "traditional_contract": [],
        "fixed_pricing": [],
        "Average_Policy_Loss": [],
        "Average_Value_Loss": [],
        "Average_Entropy": []
    }
    Optimizer_Change = True
    # 创建MAPPO智能体（共有agent_num个actor, 一个共享critic）
    mappo = MAPPO(agent_num, state_dim, action_dim, actor_lr, critic_lr, lmbda, gamma, eps, K_epochs, device, R_RANGE,total_episode)
    if Fixed_PPO:
        fixed_state_dim = env.agent_num * env.uav_num
        fixed_action_dim = env.uav_num + 1
        ppo = PPO(fixed_state_dim,fixed_action_dim)
        fixed_env = copy.deepcopy(env)
        if Optimizer_Change:
            scheduler_actor = ReduceLROnPlateau(
                    ppo.actor_optimizer,  # 监控它的优化器
                    mode='max',  # 目标是 Reward 最大化
                    factor=0.6,
                    patience=100,
                    threshold=1e-4,
                    min_lr=1e-6,
                    verbose=True
                )

    if Traditional_Contract:
        traditional_env = copy.deepcopy(env)

    log.LogRecord('Initialize agents success......')

    log.LogRecord('Training is on the way.')
    plt.close('all')
    plt.figure(figsize=(15, 8))
    plt.ion()   # 开启实时绘制窗口
    all_steps = 0
    max_reward = 0
    for episode in range(1, total_episode+1):
        # 初始化Trajectory buffer
        buffers = [{
            'states': [],
            'actions_raw': [],
            'next_states': [],
            'rewards': [],
            'log_probs': []
        } for _ in range(agent_num)]

        #重置环境
        multi_states, _ = env.Reset()
        episode_reward = 0.0
        if(episode%500==0):
            max_reward = 0

        for step_i in range(batch_size):
            all_steps += 1
            # 获取动作
            actions, log_probs = mappo.take_action(multi_states)

            # 根据多个智能体的动作，获取下一步的状态。
            multi_reward, next_multi_state, contracts,mappo_acceptance_rate,total_utility_matrix,ic_total = env.Step(actions)

            # 将平均奖励作为每个智能体的奖励
            avg_reward = np.mean(multi_reward)
            std_reward = np.std(multi_reward)
            shared_reward = tau_1 * avg_reward - tau_2 * std_reward
            episode_reward += shared_reward

            # 存储经验
            for agent_i in range(agent_num):
                buffers[agent_i]['states'].append(np.array(multi_states[agent_i]))
                buffers[agent_i]['actions_raw'].append(actions[agent_i])
                buffers[agent_i]['next_states'].append(np.array(next_multi_state[agent_i]))
                buffers[agent_i]['rewards'].append(shared_reward)
                buffers[agent_i]['log_probs'].append(log_probs[agent_i])

            multi_states = next_multi_state

            # 记录每一步的智能体平均效用
            avg_utility = np.mean([agent.utility for agent in env.Agents])
            avg_utility_per_episode.append(avg_utility)

            # 记录平均reward
            avg_reward_per_episode.append(shared_reward)


        # 使用mappo更新参数
        a_loss, c_loss, entropy,actors_current_lr,critic_current_lr = mappo.update(buffers)

        # 记录奖励值
        rewards_window.append(np.mean(avg_reward_per_episode))

        if Traditional_Contract:
            traditional_env.Reset()
            optimizer = TraditionalContractOptimizer(traditional_env)
            r_best, u_best, util = optimizer.solve()
            tra_multi_reward, tra_next_multi_state, tra_contracts,tra_acceptance_rate = traditional_env.step_2(r_best,u_best,1)
            traditional_reward = np.mean(tra_multi_reward)

        if Fixed_PPO:
            fixed_env.Reset()
            fixed_state = [random.randint(0, 1) for _ in range(env.uav_num*env.agent_num)]
            fixed_reward,fixed_contract, unit_price,fixed_acceptance_rate = run_fixed_train(ppo, fixed_env, fixed_state)

        # 每10轮修改一次学习率
        if episode%10 ==0:
            Log("episode：{}, AVG_Reward:{}, True_reward:{},actors_lr{},critic_lr{}".format(
                episode, avg_reward_per_episode[-1], avg_reward_per_episode[-1], actors_current_lr, critic_current_lr
            ))
            ## 打印数据
            for i, contract in enumerate(contracts):
                Log(f"agent[{i+1}]==>contract :{contract}",False)
            Log(f"acceptance_rate==> mappo:{mappo_acceptance_rate}\n"
                f"traditional_contract:{tra_acceptance_rate} and contract is {tra_contracts}\n"
                f"fixed_pricing_ppo:{fixed_acceptance_rate} and contract is {fixed_contract} unit_price == >{unit_price} \n",False)
            # 计算平滑后的平均奖励
            mean_reward = np.mean(rewards_window)
            # 检查是否是历史最佳
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                # print(f"Episode {episode}: New best mean reward: {mean_reward:.2f}") # 可选：打印新高
            # --- 核心步骤：让调度器根据平滑后的奖励来决策 ---
            for scheduler in mappo.actor_schedulers:
                scheduler.step(mean_reward)
            mappo.critic_scheduler.step(mean_reward)

            metrics_dict['mappo_contract'].append(avg_reward)
            metrics_dict['traditional_contract'].append(traditional_reward)
            metrics_dict['fixed_pricing'].append(fixed_reward)
            if Fixed_PPO:
                if Optimizer_Change:
                    scheduler_actor.step(fixed_reward)

            # 调用新的绘图函数
            # plot_all_metrics(metrics_dict, episode)
            plot_learning_curves(metrics_dict, episode, 'mappo', window_size=20)

            # ic约束
            if ic_total == env.uav_num * env.agent_num and episode%300 ==0:
                Log(f"IC_uti:{total_utility_matrix[0]}",False)

            # 记录数据
            record_data.log_agent_data(episode)
            record_data.log_uav_data(episode)

            # 存储数据
            record_data.save_to_csv('agent', 'agent_results.csv')
            record_data.save_to_csv('uav', 'uav_results.csv')

        # 保存模型的权重参数
        if episode % 500 == 0:
            mappo.save_model()


    return metrics_dict


        # ------------------------------------------------------------------
if __name__ == '__main__':
    metrics_dict = Multi_Contract_Play()
 # 1. 实例化管理器
    data_manager = ExperimentDataManager(save_dir="results/my_experiments")

    # 2. 调用保存接口 (直接用 变量名=变量 的形式传进去，非常方便)
    data_manager.save_metrics(
        filename="uti_compare",
        total_uti=metrics_dict
    )
    plot_learning_curves_(metrics_dict,3000,'TOTAL_UTILTY',window_size=20,combine_plots=True)