import gym
import numpy as np

env = gym.make('MountainCar-v0')
init_state = env.reset()

done = False

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

print(discrete_os_win_size)

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [3]))

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000
SHOW_EVERY = 2000


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


print(q_table[1:3])


for episode in range(EPISODES):
    done = False
    show_flag = False
    if episode % SHOW_EVERY == 0:
        show_flag = True
    else:
        show_flag = False
    discrete_state = get_discrete_state(env.reset())
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if show_flag:
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        elif new_state[0] > env.goal_position:
            print(episode)
            # q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

env.close()
