import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import gym
import gym_xarm

env_id = "XarmPDHandover-v1"
log_dir = 'saved_data/'+env_id
# Load the saved statistics
env = DummyVecEnv([lambda: gym.make(env_id)])
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env = VecNormalize.load(stats_path, env)
env.training = False
env.norm_reward = False
# load agent
model = SAC.load("sac_hanover")

obs = env.reset()
episode_reward = 0
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done or info.get("is_success", False):
        print("Reward:", episode_reward, "Success?", info.get("is_success", False))
        episode_reward = 0.0
        obs = env.reset()