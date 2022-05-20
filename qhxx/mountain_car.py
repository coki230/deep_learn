import gym

env = gym.make("MountainCar-v0")
print("state: ", env.observation_space)
print("action: ", env.action_space)
print("action num: ", env.action_space.n)


class BespokeAgent:
    def __int__(self, env_agent):
        pass

    def decide(self, observation):
        position, velocity = observation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action

    def learn(self, *args):
        pass


agent = BespokeAgent()


def play_montecarlo(env_play, agent_play, render=False, train=False):
    episode_reward = 0.
    observation = env_play.reset()
    while True:
        if render:
            env_play.render()
        action = agent_play.decide(observation)
        next_observation, reward, done, _ = env_play.step(action)
        episode_reward += reward
        if train:
            agent_play.learn(observation, action, reward, done)
        if done:
            break
        observation = next_observation
    return episode_reward


env.seed(0)
episode_reward = play_montecarlo(env, agent, render=True)
print("all reward: ", episode_reward)
env.close()
