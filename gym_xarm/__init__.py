import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='XarmPickAndPlace-v0',
    entry_point='gym_xarm.envs:XarmPickAndPlaceEnv',
)
register(
    id='XarmPDPickAndPlace-v0',
    entry_point='gym_xarm.envs:XarmPDPickAndPlaceEnv',
)
register(
    id='XarmReach-v0',
    entry_point='gym_xarm.envs:XarmReachEnv',
)
register(
    id='XarmPDRearrange-v0',
    entry_point='gym_xarm.envs:XarmPDRearrangeEnv',
)
register(
    id='XarmPDStackTower-v0',
    entry_point='gym_xarm.envs:XarmPDStackTowerEnv',
)
register(
    id='XarmPDPushWithDoor-v0',
    entry_point='gym_xarm.envs:XarmPDPushWithDoorEnv',
)
register(
    id='XarmPDOpenBoxAndPlace-v0',
    entry_point='gym_xarm.envs:XarmPDOpenBoxAndPlaceEnv',
)
register(
    id='XarmPDHandover-v0',
    entry_point='gym_xarm.envs:XarmPDHandover',
)
register(
    id='XarmPDHandover-v1',
    entry_point='gym_xarm.envs:XarmPDHandoverDense',
)
register(
    id='XarmPDHandoverNoGoal-v1',
    entry_point='gym_xarm.envs:XarmPDHandoverDenseNoGoal',
)
