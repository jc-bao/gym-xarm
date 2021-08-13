import os
import numpy as np
import matplotlib.pyplot as plt

import gym
import gym_xarm

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results

class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True

if __name__ == '__main__':
    reward_type = "dense"
    env_id = "XarmPDHandoverNoGoal-v1"
    log_dir = 'saved_data/'+env_id
    num_cpu = 4
    timesteps = 25000
    n_sampled_goal = 4 # artificial/real transition in HER

    # make env
    env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=DummyVecEnv, monitor_dir = log_dir)
    env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_obs=10.)

    # create model
    if reward_type == "dense":
        model = SAC('MultiInputPolicy', env, verbose=1)
    elif reward_type == "sparse":
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=n_sampled_goal,
                goal_selection_strategy="future",
                max_episode_length=100,
                online_sampling=True,
            ),
            verbose=1,
            buffer_size=int(1e6),
            learning_rate=1e-3,
            gamma=0.95,
            batch_size=256,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
        )
    else:
        raise "Reward Type Error"
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    # learn
    model.learn(total_timesteps=timesteps, callback=callback)

    # save 
    # the VecNormalize statistics when saving the agent
    stats_path = os.path.join(log_dir, "vec_normalize.pkl")
    env.save(stats_path)
    # the fig
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Xarm_Handover_Dense")
    plt.savefig("saved_data/"+"Plot.png")