import gym
import numpy as np

env = gym.make('MountainCar-v0')
init_state = env.reset()

"""
    Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07

"""
ob_high = env.observation_space.high
ob_low = env.observation_space.low
ob_len = ob_high - ob_low
ob_win = [20, 20]
ob_gap = ob_len / ob_win
LEARNING_RATE = 0.1
EPISODES = 15000
DISCOUNT = 0.95

q_table = np.random.uniform(low=-2, high=0, size=(ob_win + [env.action_space.n]))



def get_discrete_value(observation):
    return tuple(((observation - ob_low)/ ob_gap).astype(int))

#
# print(q_table[1:3])
#
# for episode in range(EPISODES):
#     observation_state = env.reset()
#     done = False
#     show_flag = False
#     if (episode + 1) % 2000 == 0:
#         show_flag = True
#     while not done:
#         state_vals = q_table[get_discrete_value(observation_state)]
#         action = np.argmax(state_vals)
#         new_state, reward, done, info = env.step(action)
#         current_q = q_table[get_discrete_value(observation_state) + (action,)]
#         new_state_vals = q_table[get_discrete_value(new_state)]
#         if not done:
#             new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * np.max(new_state_vals))
#             q_table[get_discrete_value(observation_state) + (action,)] = new_q
#         elif new_state[0] > env.goal_position:
#             q_table[get_discrete_value(observation_state) + (action,)] = 0
#
#         observation_state = new_state
#
#         if show_flag:
#             ren = env.render()
#         if done and reward > -1:
#             print(episode)
#
# np.save('q_table.npy', q_table)


# varify the result
test_q_table = np.load('q_table.npy')
for i in range(5):
    test_state = env.reset()
    test_done = False
    while not test_done:
        action = test_q_table[get_discrete_value(test_state)]
        test_new_state, test_reward, test_done, _ = env.step(np.argmax(action))
        env.render()
        test_state = test_new_state

env.close()
