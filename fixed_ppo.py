import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


# --- 1. 网络结构定义 ---

class PolicyNet(nn.Module):
    """Actor 网络：负责输出动作的均值和标准差"""

    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        # 均值输出层
        self.mean_layer = nn.Linear(128, action_dim)
        # 标准差输出层（使用 log_std 保证标准差恒正且数值稳定）
        self.log_std_layer = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.mean_layer(x)
        log_std = torch.clamp(self.log_std_layer(x), -20, 2)
        return mean, log_std.exp()


class ValueNet(nn.Module):
    """Critic 网络：负责评估状态的价值 V(s)"""

    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.layers(x)


# --- 2. PPO 算法核心类 ---

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=3e-4, lr_critic=1e-3,
                 gamma=0.99, lmbda=0.95, eps=0.2, epochs=10, device='cpu'):
        self.device = device
        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE 参数
        self.eps = eps  # PPO 截断范围 (Clip)
        self.epochs = epochs  # 每次更新时迭代的次数

        # 初始化网络
        self.actor = PolicyNet(state_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim).to(device)

        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def take_action(self, state):
        """输入状态，输出动作及对应的对数概率"""
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        # 计算该动作在当前分布下的 log_prob，用于后续计算 Ratio
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().cpu().numpy().flatten(), log_prob.detach().cpu().numpy()

    def compute_advantage(self, td_delta):
        """使用 GAE 计算优势函数"""
        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.array(advantage_list), dtype=torch.float).to(self.device)

    def update(self, transition_dict):
        """核心更新方法"""
        # 1. 转换 Buffer 数据为 Tensor
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(transition_dict['actions']), dtype=torch.float).to(self.device)
        rewards = torch.tensor(np.array(transition_dict['rewards']), dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        old_log_probs = torch.tensor(np.array(transition_dict['log_probs']), dtype=torch.float).to(self.device)

        # 2. 计算 TD Target 和 TD Delta (优势函数的基础)
        with torch.no_grad():
            td_target = rewards + self.gamma * self.critic(next_states)
            td_delta = td_target - self.critic(states)

        # 3. 计算优势函数 Advantage
        advantage = self.compute_advantage(td_delta)

        # 4. 循环迭代更新多个 Epoch
        for _ in range(self.epochs):
            # 获取当前策略下的分布
            mean, std = self.actor(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)

            # 计算概率比值 Ratio: pi_theta / pi_old
            # 由于使用的是对数概率，相减再取 exp 等于直接相除
            ratio = torch.exp(new_log_probs - old_log_probs)

            # PPO Clipped Objective
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # Actor Loss: 取最小值并取负（因为优化器执行的是最小化，而我们需要最大化目标）
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            # Critic Loss: 均方误差 (MSE)
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 熵奖励 (Entropy Bonus): 鼓励探索，防止策略过快收敛到局部最优
            entropy_loss = torch.mean(dist.entropy().sum(dim=-1))

            # 总损失
            total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

            # 梯度更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            # 梯度裁剪：防止梯度爆炸，提高稳定性
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save_model(self, path):
        torch.save(self.actor.state_dict(), path + "_actor.pth")
        torch.save(self.critic.state_dict(), path + "_critic.pth")

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path + "_actor.pth"))
        self.critic.load_state_dict(torch.load(path + "_critic.pth"))