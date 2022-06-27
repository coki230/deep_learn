import gym

env = gym.make("Taxi-v3")

env.reset()

state = env.encode(3, 1, 2, 0)
print(state)
env.s = state

env.render()

print(env.action_space)
print(env.observation_space)
