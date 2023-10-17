'''
基于Q-table,TD算法更新Q-table
只能用于有限状态空间，有限动作空间
'''
NUM_EPISODE = 10000
GAMMA = 0.99
LR = 0.85
import gym
import numpy as np


env = gym.make('FrozenLake-v1')
#获得状态空间和动作空间
n_state, action_card = env.observation_space.n, env.action_space.n
#初始化Q-table
qtable = np.zeros([n_state, action_card])
#初始化状态

rewards = []
for i in range(NUM_EPISODE):
    state = env.reset()[0]
    epi_reward = 0
    for j in range(99):
        #noisy-greedy得到下一个状态
        action = np.argmax(qtable[state, :] + np.random.randn(1, action_card) * (1. / (i + 1)))

        #action和env交互，得到reward和next_state
        next_state, reward, terminated, truncted, _ = env.step(action)

        #qlearning更新
        ##greedy search得到下一个action

        qtable[state, action] = qtable[state, action] + LR * (reward + GAMMA * np.max(qtable[next_state, :]) - qtable[state, action])
        epi_reward += reward

        #更新状态
        state = next_state
        if terminated or truncted:
            break
    print('episode:%d return: %.4f'%(i, epi_reward))

#渲染一下。。。
env = gym.make('FrozenLake-v1', render_mode='human')
for i in range(10):
    
    state = env.reset()[0]
    env.render()
    while True:
        action = np.argmax(qtable[state,:])
        next_state, reward, terminated, truncted, _ = env.step(action)

        if truncted or terminated:
            break
        state = next_state
