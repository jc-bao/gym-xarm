import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='XarmFetch-v0',
    entry_point='gym_xarm.envs:XarmFetchEnv',
)
register(
    id='XarmPDFetch-v0',
    entry_point='gym_xarm.envs:XarmPDFetchEnv',
)
register(
    id='XarmReach-v0',
    entry_point='gym_xarm.envs:XarmReachEnv',
)