import time
import os
import gym
import numpy as np
import gym_xarm
# FetchPickAndPlace-v0 XarmPDStackTower-v0 XarmPDPushWithDoor-v0 XarmPDOpenBoxAndPlace-v0 XarmPDHandover-v0
env = gym.make('XarmReach-v1') 
agent = lambda ob: env.action_space.sample()
ob = env.reset()
for _ in range(env._max_episode_steps*100):
    env.render()
    assert env.observation_space.contains(ob)
    a = agent(ob)
    assert env.action_space.contains(a)
    (ob, _reward, done, _info) = env.step(a)
    time.sleep(0.02)
env.close()