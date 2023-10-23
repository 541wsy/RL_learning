import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Categorical
import numpy as np

NUM_EPISODE = 1000
EPSILON = 0.1
HIDDEN_DIM=128
LR = 2e-3
GAMMA = 0.98

class AC:
    def __init__(self, state_dim, n_action, hidden_dim, eps=EPSILON, lr=LR, gamma=GAMMA) -> None:

        self.state_dim = state_dim
        self.n_action = n_action
        self.hidden_dim = hidden_dim

        self.actor_network = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, n_action),
                                           nn.Softmax(dim=-1))
        self.value_network = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hidden_dim, 1))
        self.actor_optimizer = Adam(self.actor_network.parameters(), lr=lr)
        self.value_optimizer = Adam(self.value_network.parameters(), lr=lr)
        
        self.eps = eps
        self.gamma = gamma

    def take_action(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        probs = self.actor_network(state)
        cate = Categorical(probs)
        return cate.sample().item()
    
    def train(self, transitions):
        self.actor_optimizer.zero_grad()
        self.value_optimizer.zero_grad()

        for action, state, next_state, reward, terminated, truncted in transitions:
            action, state, next_state, reward = torch.tensor(action), torch.tensor(state), torch.tensor(next_state), torch.tensor(reward)

            #TD 算法更新value network
            next_probs = self.actor_network(next_state)
            next_action = Categorical(next_probs).sample()
            TD_target = reward + self.gamma * self.value_network(next_state) * (1-terminated) * (1-truncted)
            loss_value = F.mse_loss(self.value_network(state), TD_target)
            
            baseline = TD_target - self.value_network(state)
            #policy gradient更新actor network
            loss_actor = -torch.log(self.actor_network(state)[action]) * baseline.detach()
            
            loss_actor.backward()
            loss_value.backward()
            
        self.value_optimizer.step()
        self.actor_optimizer.step()
            
        return

env = gym.make('CartPole-v1')
state_dim, n_action, hidden_dim = env.observation_space.shape[0], env.action_space.n, HIDDEN_DIM
returns = []
agent = AC(state_dim, n_action, hidden_dim)

for i in range(NUM_EPISODE):
    epi_return = 0
    state = env.reset()[0]
    transitions = []
    while True:
        action = agent.take_action(state)
        next_state, reward, terminated, truncted, _ = env.step(action)
        transitions.append([action, state, next_state, reward, terminated, truncted])
        epi_return += reward

        if terminated or truncted:
            break
        state = next_state
    agent.train(transitions)
    if (i+1) % 10 == 0:
        print("Episode[%d|%d] return:%.4f"%(i+1, NUM_EPISODE, epi_return))
env.close()

#渲染环境，可视化
env = gym.make("CartPole-v1", render_mode="human")

for i in range(10):
    state = env.reset()[0]
    while True:
        env.render()
        action = agent.take_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # add transition to queue.
        if terminated or truncated:
            break
        state = next_state
        

'''
Q: value网络估计V值而非Q值的原因是加快收敛吗？可是policy gradient的公式应该是policy网络的梯度乘Q值？
关于policy gradient的推导还是需要再看一下。搞清楚这里为什么value network是估计V值，而非Q值

Q：为什么不用replay buffer？为什么DQN可以用replay buffer
'''
