import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import gym
import numpy as np
from random import sample

LR = 3e-3
GAMMA = 0.97
NUM_EPISODE = 200
HIDDEN_DIM = 128
EPSILON = 0.01
CAPACITY = 1000
BATCH_SIZE = 64
MINI_SIZE = 500


from collections import deque
class ReplayBuffer:
    def __init__(self, capacity) -> None:
        self.buffer = deque(maxlen=capacity) #设置一个最大长度的队列，存放用于TD训练的每条样本
        return
    
    def add(self, state, action, reward, next_state, terminated, truncted):
        self.buffer.append([state, action, reward, next_state, terminated, truncted])

    def sample(self, batch_size):
        return sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    

class QNet(nn.Module):
    def __init__(self, state_dim, n_action, hidden_dim, gamma=GAMMA, epsilon=EPSILON) -> None:
        super().__init__()
        self.gamma = gamma
        self.eps = epsilon #eps greedy search
        self.n_action = n_action
        self.qnet = nn.Sequential(nn.Linear(state_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, n_action))
        self.state_dim = state_dim
    def forward(self, state):
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state)
        for layer in self.qnet:
            state = layer(state)
        return state

    def epsilon_gs(self, state):
        if np.random.random() < self.eps:
            return np.random.randint(self.n_action)
        else:
            return torch.argmax(self.forward(state)).numpy()
        
    def train(self, transitions, batch_size, optimizer):
        states, actions, rewards, next_states, terminateds, truncteds = torch.zeros((batch_size, self.state_dim)), torch.zeros((batch_size, 1), dtype=torch.int64), \
                                                torch.zeros((batch_size)), torch.zeros((batch_size, self.state_dim)), \
                                                torch.zeros((batch_size)), torch.zeros((batch_size))
        for i in range(batch_size):
            states[i], actions[i], rewards[i], next_states[i], terminateds[i], truncteds[i] = torch.tensor(transitions[i][0]), torch.tensor(transitions[i][1], dtype=torch.int64), \
                                                                torch.tensor(transitions[i][2]), torch.tensor(transitions[i][3]), \
                                                                torch.tensor(transitions[i][4]), torch.tensor(transitions[i][5])
        #计算batch loss
        td_target = rewards + torch.max(self.gamma * self.forward(next_states), axis=1)[0] * (1 - terminateds) * (1 - truncteds)
        loss = F.mse_loss(self.forward(states).gather(1, actions).squeeze(), td_target)
        #更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    n_action, state_dim, hidden_dim, batch_size = env.action_space.n, env.observation_space.shape[0], HIDDEN_DIM, BATCH_SIZE

    agent = QNet(state_dim, n_action, hidden_dim)
    buffer = ReplayBuffer(capacity=CAPACITY)
    optimizer = Adam(agent.qnet.parameters(), lr=LR)

    returns = []

    for i in range(NUM_EPISODE):
        state = env.reset()[0]
        epi_return = 0

        while True:
            #eps_greedy_search
            action = agent.epsilon_gs(state)
            #action和环境交互
            next_state, reward, terminated, truncted, _ = env.step(action)
            buffer.add(state, action, reward, next_state, terminated, truncted)
            epi_return += reward

            if terminated or truncted:
                break
            #更新state
            state = next_state
            #更新return
            epi_return += reward
            
            if len(buffer) > MINI_SIZE:
                transitions = buffer.sample(batch_size)
                agent.train(transitions, batch_size, optimizer)


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
            action = agent(state).argmax().numpy()
            next_state, reward, terminated, truncated, _ = env.step(action)
            # add transition to queue.
            if terminated or truncated:
                break
            state = next_state



'''
batch_size对结果的影响非常大。当batch_size为1时reward并不会呈现明显增长的趋势，而是呈现非常震荡的略微上升。
王树森强化学习经验回放章节指出：SGD的缺点主要在于：
1.每一条transition是一条经验，SGD会导致经验的浪费，一条经验只参与一次BP，用完就丢弃了。
2.SGD的话相邻两次更新的state是非常相似的，实验表明，这样更新agent效果不好。？？不知道是否有理论解释

replay buffer中需要把terminated和truncted的sanple也添加进去，如果没有这一项，agent无法学习。
这是因为需要terminated和truncted提供负反馈？当前state和action会导致游戏失败。
如果没有terminated和truncted的话，只有正反馈。
'''