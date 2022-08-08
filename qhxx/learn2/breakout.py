import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
from atari_py import import_roms

import_roms('C:\\Users\\Coki_Zhao\\Desktop\\learn\\ai\\keras\\qhxx\\ROMS')
environment_name = 'Breakout-v0'
env = gym.make(environment_name)

env.reset()
