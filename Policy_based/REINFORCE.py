import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

import numpy as np


NUM_EPISODE = 500
HIDDEN_DIM = 128
EPSILON = 0.01
GAMMA = 0.9
LR = 0.001

class REINFORCE:
    def __init__(self, state_dim, n_action, hidden_dim, eps=EPSILON, gamma=GAMMA) -> None:
        self.state_dim = state_dim
        self.n_action = n_action
        self.hidden_dim = hidden_dim
        self.eps = eps
        self.gamma = gamma

        self.policy = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                    nn.ReLU(),
                                    nn.Linear(hidden_dim, n_action),
                                    nn.Softmax(dim=-1))
        
        self.states, self.actions, self.rewards = [], [], []
    def eps_gs(self, state):
        '''eps greedy search'''
        if np.random.rand() < self.eps:
            return np.random.randint(self.n_action)
        else:
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(state)
            probs = self.policy(state)
            categorical = Categorical(probs)
            return categorical.sample().item()
    
    def add(self, state, action, reward):
        '''记录每一个episode从前到后的state, action, reward'''
        self.states.append(state)
        self.actions.append(int(action))
        self.rewards.append(reward)
        return

    def train(self, optimizer):
        '''一轮episode结束，基于policy gradient 开始更新policy网络的参数'''
        #获得每个state的G（累积折扣回报）
        steps = len(self.states)
        Gs = torch.zeros(steps)
        tmp_G = 0
        for i in range(steps-1, -1, -1):
            Gs[i] = self.gamma * tmp_G + self.rewards[i]
            tmp_G = Gs[i]
        #Gs进行标准化
        Gs = (Gs - Gs.mean()) / Gs.std()

        neg_logits = -torch.log(self.policy(torch.tensor(self.states))).gather(1, torch.tensor(self.actions).reshape((-1,1)))
        loss = (neg_logits.squeeze() * Gs).sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return
    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        return


env = gym.make('CartPole-v1')
state_dim, n_action, hidden_dim = env.observation_space.shape[0], env.action_space.n, HIDDEN_DIM

agent = REINFORCE(state_dim, n_action, hidden_dim)
optimizer = Adam(agent.policy.parameters(), lr=LR)
returns = []

for i in range(NUM_EPISODE):
    state = env.reset()[0]
    epi_return = 0

    while True:
        action = agent.eps_gs(state) #eps greedy search得到action
        next_state, reward, terminated, truncted, _ = env.step(action) #action和环境交互
        agent.add(state, action, reward) #记录状态，动作，reward
        epi_return += reward
        state = next_state

        #如果停止了，开始一轮更新
        if terminated or truncted:
            agent.train(optimizer)
            agent.clear()
            break

    returns.append(epi_return)
    if (i+1) % 10 == 0:
        print("Episode[%d|%d] return:%.4f"%(i+1, NUM_EPISODE, epi_return))
env.close()

#渲染环境，可视化
env = gym.make("CartPole-v1", render_mode="human")

for i in range(10):
    state = env.reset()[0]
    while True:
        env.render()
        action = agent.eps_gs(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # add transition to queue.
        if terminated or truncated:
            break
        state = next_state