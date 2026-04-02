from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
# --- 1. 环境和模型配置 ---
# 建议将这些作为可配置参数传入，这里为了演示设为全局常量
 # R动作的范围 (min, max)

result_base_dir = "results"
env_name = "contract_mappo"
result_dir = os.path.join(result_base_dir, env_name)
weights_dir = os.path.join(result_dir, "weights")


# --- 2. 策略网络 (PolicyNet) ---
class PolicyNet(nn.Module):
    def __init__(self, state_dim, actor_lr, action_dim,total_episode, trainable=True):
        self.total_episode = total_episode
        super(PolicyNet, self).__init__()
        self.action_dim = action_dim

        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 256)
        self.l3 = nn.Linear(256, 512)
        self.mean_layer = nn.Linear(512, action_dim)
        self.log_std_layer = nn.Linear(512, action_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=actor_lr)

        # 3. 创建学习率调度器
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_decay_schedule)

        # --- 2. 创建自适应调度器 ---
        # 监控一个指标，当它停止上升时 ('max' mode)，降低学习率
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 我们监控奖励，所以希望它越大越好
            factor=0.5,  # 当触发时，学习率变为原来的一半
            patience=10,  # 如果连续10次评估，平均奖励都没有创下新高，就降低LR
            min_lr=1e-8,  # 学习率最低不会低于 1e-6
            verbose=True  # 降低时打印提示
        )

        for param in self.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))

        # 让网络直接输出无界的均值
        mean = self.mean_layer(x)

        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def save_checkpoint(self, checkpoint_file):
        torch.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file))

    # 2. 定义线性衰减的 lambda 函数
    #    这个函数接收当前步数，返回一个学习率的乘法因子
    def linear_decay_schedule(self,step):
        progress = step/self.total_episode
        # 乘法因子从1线性下降到0
        return max(0.0, 1.0 - progress)


# --- 3. 中心化价值网络 (CentralValueNet) ---
class CentralValueNet(nn.Module):
    def __init__(self, global_state_dim, critic_lr,total_episode):
        super(CentralValueNet, self).__init__()
        self.total_episode = total_episode
        self.layers = nn.Sequential(
            nn.Linear(global_state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)

        # 3. 创建学习率调度器
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.linear_decay_schedule)

        # --- 2. 创建自适应调度器 ---
        # 监控一个指标，当它停止上升时 ('max' mode)，降低学习率
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # 我们监控奖励，所以希望它越大越好
            factor=0.5,  # 当触发时，学习率变为原来的一半
            patience=10,  # 如果连续10次评估，平均奖励都没有创下新高，就降低LR
            min_lr=1e-7,  # 学习率最低不会低于 1e-6
            verbose=True  # 降低时打印提示
        )


    def forward(self, state):
        return self.layers(state)

    def linear_decay_schedule(self,step):
        progress = step/self.total_episode
        # 乘法因子从1线性下降到0
        return max(0.0, 1.0 - progress)

# --- 4. MAPPO 算法核心类 ---
class MAPPO:
    def __init__(self, agent_num, state_dim, action_dim, actor_lr, critic_lr,
                 lmbda, gamma, eps, K_epochs, device, R_RANGE,total_episode, entropy_coef=0.01, mini_batch_size=64):
        self.total_episode = total_episode
        self.agent_num = agent_num
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = state_dim * agent_num
        self.lmbda = lmbda
        self.gamma = gamma
        self.eps = eps
        self.K_epochs = K_epochs
        self.device = device
        self.entropy_coef = entropy_coef
        self.mini_batch_size = mini_batch_size
        self.R_RANGE = R_RANGE

        # 初始化网络
        self.actor = [PolicyNet(state_dim, actor_lr, action_dim,self.total_episode).to(device) for _ in range(agent_num)]
        self.critic = CentralValueNet(self.global_state_dim, critic_lr,self.total_episode).to(device)
        self.actor_optimizers = [actor.optimizer for actor in self.actor]
        self.actor_schedulers = [actor.scheduler for actor in self.actor]
        self.critic_optimizer = self.critic.optimizer
        self.critic_scheduler = self.critic.scheduler


    def save_model(self, path=None):
        if path is None: path = weights_dir
        if not os.path.exists(path): os.makedirs(path)
        for i, actor in enumerate(self.actor):
            torch.save(actor.state_dict(), os.path.join(path, f"actor_{i}.pth"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def load_model(self, path=None):
        if path is None: path = weights_dir
        for i, actor in enumerate(self.actor):
            actor_path = os.path.join(path, f"actor_{i}.pth")
            if os.path.exists(actor_path):
                actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        critic_path = os.path.join(path, "critic.pth")
        if os.path.exists(critic_path):
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

    @torch.no_grad()
    def take_action(self, multi_state):
        """
            将原始动作将直接发送给环境进行解码。
        """
        actions_list, log_probs_list = [], []

        for i, actor in enumerate(self.actor):
            state = torch.tensor(np.array([multi_state[i]]), dtype=torch.float, device=self.device)

            mean, log_std = actor(state)
            dist = Normal(mean, log_std.exp())

            # 采样原始动作 a_raw。
            # (a_R5_base, a_delta_4, a_delta_3, a_delta_2, a_delta_1)
            action = dist.sample()

            # log_prob 的计算
            log_prob = dist.log_prob(action).sum(dim=-1)

            # 我们直接将原始的、无界的 action作为要发送给环境的动作。
            actions_list.append(action.cpu().numpy().flatten())
            log_probs_list.append(log_prob.cpu().numpy())

        return actions_list, log_probs_list

    def _process_action_for_env(self, action):
        """
        将[-1, 1]范围的 squashed 动作映射到[R_RANGE, U_RANGE]并对R取整。
        这个函数的输入现在是 torch.tanh(action_raw)
        """
        r_scaled = self._scale_action(action, self.R_RANGE[0], self.R_RANGE[1])

        r_integer = torch.round(r_scaled)

        return r_integer

    # _scale_action 函数本身是正确的，无需修改
    def _scale_action(self, action_val, low, high):
        """Helper to scale action from [-1, 1] to [low, high]."""
        return low + (0.5 * (action_val + 1.0) * (high - low))

    def compute_advantage(self, td_delta):
        """计算GAE优势函数。"""
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(advantage_list, dtype=torch.float, device=self.device)

    def update(self, transition_dicts):
        # 1. 数据准备：从buffer中提取数据并转换为Tensor
        T = len(transition_dicts[0]['states'])

        # 全局状态
        global_states_list = [np.concatenate([d['states'][t] for d in transition_dicts]) for t in range(T)]
        global_states = torch.tensor(np.array(global_states_list), dtype=torch.float, device=self.device)

        # 个体数据
        states = torch.tensor(np.array([d['states'] for d in transition_dicts]), dtype=torch.float,
                              device=self.device).transpose(0, 1)
        actions = torch.tensor(np.array([d['actions_raw'] for d in transition_dicts]), dtype=torch.float,
                                   device=self.device).transpose(0, 1)
        old_log_probs = torch.tensor(np.array([d['log_probs'] for d in transition_dicts]), dtype=torch.float,
                                     device=self.device).transpose(0, 1)

        # 奖励
        rewards = torch.tensor(np.array([[d['rewards'][t] for d in transition_dicts] for t in range(T)]),
                               dtype=torch.float, device=self.device)

        # 只需要最后一个 next_state 来做 bootstrap
        last_next_global_state_list = [d['next_states'][-1] for d in transition_dicts]

        last_next_global_state = torch.tensor(np.concatenate(last_next_global_state_list), dtype=torch.float,
                                              device=self.device).unsqueeze(0)
        # ---

        # 2. 计算优势函数
        with torch.no_grad():
            # Critic输出 (T, 1)
            values = self.critic(global_states).squeeze(-1)  # (T, 1) -> (T,)

            last_next_value = self.critic(last_next_global_state).squeeze()  # (1,1) -> scalar

            # 拼接得到所有时间步的 next_values
            next_values = torch.cat([values[1:], last_next_value.unsqueeze(0)])

            # 由于没有 done, (1-dones) 恒为 1
            td_target = rewards + self.gamma * next_values.unsqueeze(1)  # (T, 1) + scalar * (T, 1) -> (T, agent_num)

            # Critic 输出的 values 需要 unsqueeze 以匹配 td_target 的形状
            td_delta = td_target - values.unsqueeze(1)  # (T, agent_num) - (T, 1) -> (T, agent_num)

            # 计算 GAE
            advantages_per_agent = [self.compute_advantage(td_delta[:, i]) for i in range(self.agent_num)]
            advantages = torch.stack(advantages_per_agent, dim=1)  # (T, agent_num)
            # 优势归一化
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. K-Epochs 循环更新 (Mini-batch SGD)
        all_action_loss, all_critic_loss, all_entropy = [], [], []

        # 3. K-Epochs 循环更新
        for _ in range(self.K_epochs):
            indices = np.arange(T)
            np.random.shuffle(indices)

            for start in range(0, T, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_indices = indices[start:end]

                # --- 更新 Critic ---
                # Critic 的 target 是 td_target 的均值，或者直接用 rewards + gamma * V(s')
                # Critic的目标是逼近回报的期望，而回报对于所有agent是共享的，所以可以用td_target的均值
                # 或者更直接地，Critic的目标是 V(s_t) -> E[R_t + gamma * V(s_t+1)]
                # td_target 是 (T, agent_num), values 是 (T,). 我们需要 (T, 1)
                batch_td_target = td_target[batch_indices]  # (batch_size, agent_num)
                batch_global_states = global_states[batch_indices]  # (batch_size, global_dim)

                #  critic_loss 计算
                batch_values = self.critic(batch_global_states)  # (batch_size, 1)
                critic_loss = F.mse_loss(batch_values, batch_td_target.mean(dim=1, keepdim=True))

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optimizer.step()
                all_critic_loss.append(critic_loss.item())

                # --- 更新 Actors ---
                total_actor_loss = 0
                total_entropy = 0
                for i in range(self.agent_num):
                    batch_states_i = states[batch_indices, i]
                    batch_actions_i = actions[batch_indices, i]
                    # old_log_probs 是在 take_action 时计算的，已经包含了校正项
                    batch_old_log_probs_i = old_log_probs[batch_indices, i].squeeze()
                    batch_advantages_i = advantages[batch_indices, i]

                    # 从当前策略网络获取分布
                    mean, log_std = self.actor[i](batch_states_i)
                    dist = Normal(mean, log_std.exp())

                    # 1. 计算原始高斯分布的 log_prob
                    new_log_probs = dist.log_prob(batch_actions_i).sum(dim=-1)

                    # Squashed Gaussian 的熵是高斯熵 + 校正项的期望
                    # 这里为了简化，我们通常只使用高斯熵作为近似，并通过 entropy_coef 调整
                    # 严格的熵计算比较复杂，通常在SAC算法中实现。在PPO中，仅使用高斯熵是可接受的实践。
                    entropy = dist.entropy().mean()

                    # 计算 PPO 目标
                    ratio = torch.exp(new_log_probs - batch_old_log_probs_i)
                    surr1 = ratio * batch_advantages_i
                    surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantages_i

                    actor_loss = -torch.min(surr1, surr2).mean()

                    total_actor_loss += actor_loss
                    total_entropy += entropy
                    all_action_loss.append(actor_loss.item())

                for i in range(self.agent_num): self.actor_optimizers[i].zero_grad()
                final_loss = total_actor_loss - self.entropy_coef * total_entropy
                final_loss.backward()
                for i in range(self.agent_num):
                    nn.utils.clip_grad_norm_(self.actor[i].parameters(), 0.5)
                    self.actor_optimizers[i].step()
                all_entropy.append(total_entropy.item() / self.agent_num)



        # 返回一下actor的学习率以及critic的学习率
        actor_lr = [scheduler.optimizer.param_groups[0]['lr'] for scheduler in self.actor_schedulers]
        # actor_lr = [ scheduler.get_last_lr()[0] for scheduler in self.actor_schedulers]
        critic_lr = self.critic_scheduler.optimizer.param_groups[0]['lr']
        return np.mean(all_action_loss), np.mean(all_critic_loss), np.mean(all_entropy),actor_lr,critic_lr