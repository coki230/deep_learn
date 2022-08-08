import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

e_name = 'CartPole-v0'

# env = gym.make(e_name)
# times = 5
# for num in range(times):
#     state = env.reset()
#     done = False
#     score = 0
#
#     while not done:
#         env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, info = env.step(action)
#         score += reward
#     print('time:{}, score:{}'.format(num, score))
# env.close()

# log_path = os.path.join('Training', 'Logs')
env = gym.make(e_name)
# env = DummyVecEnv([lambda: env])
# model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
# model.learn(total_timesteps=20000)

# save model
model_path = os.path.join('Training', 'saveModel')
# model.save(model_path)

model = PPO.load(model_path)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True)
print(mean_reward, std_reward)
