import time
import os
import gym
# import gym_xarm.envs.xarm_reach_env 
import numpy as np
import gym_xarm

# env = gym.make('FetchPickAndPlace-v1') # FetchPickAndPlace-v0
env = gym.make('XarmFetch-v0') # FetchPickAndPlace-v0
agent = lambda ob: env.action_space.sample()
ob = env.reset()
while(1):
    env.render()
    assert env.observation_space.contains(ob)
    a = agent(ob)
    a = [0,0,0,0]
    assert env.action_space.contains(a)
    (ob, _reward, done, _info) = env.step(a)
    time.sleep(1/60)
env.close()