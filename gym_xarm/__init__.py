import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='XarmHandover-v0',
    entry_point='gym_xarm.envs:XarmHandover',
    max_episode_steps = 100,
)

register(
    id='XarmPickAndPlace-v1',
    entry_point='gym_xarm.envs:XarmPickAndPlace',
    max_episode_steps = 50,
)