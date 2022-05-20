import time

import gym

env = gym.make("CartPole-v1")
state = env.reset()

for t in range(10000):
    env.render()

    action = env.action_space.sample()
    state, reward, done, info = env.step(action)

    # time.sleep(1)
    if done:
        break

env.close()
