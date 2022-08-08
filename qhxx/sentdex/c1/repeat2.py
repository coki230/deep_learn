import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('MountainCar-v0')

q_table = np.random.uniform(low=-2, high=0, size=(20, 20, 3))

LEARNING_RATE = 0.1
high = env.observation_space.high
low = env.observation_space.low
window = (high - low) / 20
PRE_DRAW = 50

all_reward = []
reward_info = {'episode': [], 'min': [], 'max': [], 'avg': [], 'all': []}


def get_discrete_value(state):
    return tuple(((state - low) / window).astype(int))


for episode in range(5000):
    init_state = env.reset()
    done = False
    while not done:
        action = np.argmax(q_table[get_discrete_value(init_state)])
        new_state, reward, done, _ = env.step(action)
        all_reward.append(reward)

        new_value = (1 - LEARNING_RATE) * q_table[get_discrete_value(init_state) + (action,)] + LEARNING_RATE * (reward + 0.9 * np.max(q_table[get_discrete_value(new_state)]))
        q_table[get_discrete_value(init_state) + (action,)] = new_value

        if episode % 200 == 0:
            env.render()
        init_state = new_state

        if reward > -1:
            print(episode)
        #
        # if episode % PRE_DRAW == 0:
        #     all_val = np.sum(all_reward)
        #     min_val = np.min(all_reward)
        #     max_val = np.max(all_reward)
        #     avg_val = np.sum(all_reward[-PRE_DRAW:]) / PRE_DRAW
        #     reward_info.get('min').append(min_val)
        #     reward_info.get('max').append(max_val)
        #     reward_info.get('all').append(all_val)
        #     reward_info.get('avg').append(avg_val)
        #     reward_info.get('episode').append(episode)

env.close()
#
# plt.plot(reward_info.get('episode'), reward_info.get('min'), label='min')
# plt.plot(reward_info.get('episode'), reward_info.get('max'), label='max')
# plt.plot(reward_info.get('episode'), reward_info.get('avg'), label='avg')
# # plt.plot(reward_info.get('episode'), reward_info.get('all'), label='all')
# plt.legend(loc=4)
# plt.show()
