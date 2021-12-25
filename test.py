import time
import os
import gym
import numpy as np
import gym_xarm
# FetchPickAndPlace-v0 XarmPDStackTower-v0 XarmPDPushWithDoor-v0 XarmPDOpenBoxAndPlace-v0 XarmPDHandover-v0
config = {
    'GUI': True,
    'num_obj': 2, 
    'same_side_rate': 0.5, 
    'goal_shape': 'any', 
    'use_stand': False, 
}
env = gym.make('XarmHandover-v0', config = config) 
agent = lambda ob: env.action_space.sample()
ob = env.reset()
for i in range(env._max_episode_steps*100):
    assert env.observation_space.contains(ob)
    a = agent(ob)
    assert env.action_space.contains(a)
    (ob, _reward, done, _info) = env.step(a)
    time.sleep(0.02)
    if i % env._max_episode_steps == 0:
        ob = env.reset()
env.close()