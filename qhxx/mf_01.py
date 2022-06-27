import numpy as np
import pandas as pd
import time

np.random.seed(2)

n_states = 6
actions = ["left", "right"]
epsilon = 0.9
alpha = 0.1
lambda_para = 0.9
max_episodes = 13
fresh_time = 0.3


def build_q_table(n_state, action_para):
    table = pd.DataFrame(np.zeros((n_state, len(action_para))), columns=action_para)
    print(table)
    return table

#
# def choose_action(state_para, q_table):
# 
