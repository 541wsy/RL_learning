import gym
import numpy as np

#定义超参数----
LR = 0.85
GAMMA = 0.99
NUM_episode = 20000


env = gym.make('FrozenLake-v1')
n_state, n_action = env.observation_space.n, env.action_space.n
qtable = np.zeros([n_state, n_action]) #全零矩阵初始化qtable
returns = [] #存放每个episode的累积折扣回报

for i in range(NUM_episode):
    return_epi = 0
    state = env.reset()[0] #初始化state

    for j in range(100):
        #noisy greedy search
        action = np.argmax(qtable[state, :] + np.random.randn(n_action) / (i+1))
        #action和env交互
        next_state, reward, terminated, truncted, _ = env.step(action)
        #noisy greedy search获得next_action
        next_action = np.argmax(qtable[next_state, :] + np.random.randn(n_action) / (i+1))
        #计算TD target
        TD_target = reward + GAMMA * (qtable[next_state, next_action])
        #更新qtable
        qtable[state, action] += LR * (TD_target - qtable[state, action])

        return_epi += reward

        if terminated or truncted:
            break
        #更新状态
        state = next_state
    returns.append(return_epi)
    if (i+1) % 100 == 0:
        print("Episode[%d|%d] return:%.4f succes rate:%.4f"%(i+1, NUM_episode, return_epi, sum(returns) / len(returns)))

#渲染一下，看下效果
env = gym.make('FrozenLake-v1', render_mode='human')

for i in range(10):
    state = env.reset()[0]
    env.render()

    while True:
        action = np.argmax(qtable[state, :])
        next_state, reward, terminated, truncted, _ = env.step(action)
        if terminated or truncted:
            break
        state = next_state

