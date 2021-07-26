import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Xarm-v0',
    entry_point='gym_xarm.envs:XarmEnv',
)