import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

class XarmReachEnv(gym.GoalEnv):
    _num_client = 0
    @property
    def num_client(self): return type(self)._num_client
    @num_client.setter
    def num_client(self, val): type(self)._num_client = val
    def __init__(self, config):
        self.timeStep=1./240
        self.n_substeps = 20
        self.dt = self.timeStep*self.n_substeps
        # robot parameters
        self.distance_threshold=0.05
        self.num_joints = 17
        self.gripper_driver_index = 10
        self.gripper_base_index = 9
        self.arm_eef_index = 8
        self.reward_type = config['reward_type']
        self.pos_space = spaces.Box(low=np.array([0.2, -0.4 ,0.2]), high=np.array([0.8, 0.4, 0.6]))
        self.goal_space = spaces.Box(low=np.array([0.3, -0.25, 0.3]),high=np.array([0.5, 0.25, 0.4]))
        self.max_vel = 1
        self.max_gripper_vel = 20
        self.height_offset = 0.025
        self.startPos = [0, 0, 0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.joint_init_pos = [0, -0.009068751632859924, -0.08153217279952825, 0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 1.1524205092196147, -0.0004991403332530034] + [0]*9
        # training parameters
        self._max_episode_steps = 25
        
        # connect bullet
        if self.num_client == 1 and config['GUI']:
            p.connect(p.GUI) #or p.DIRECT for non-graphical version
            p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[-0.1,0.1,-0.1])
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        else:
            p.connect(p.DIRECT)
        self.num_client += 1
        self.if_render = False

        # bullet setup
        self.seed()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        # load table
        self.table = p.loadURDF("table/table.urdf", [0,0,-0.625], useFixedBase=True)
        # load arm
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7.urdf')
        self.xarm = p.loadURDF(fullpath, self.startPos, self.startOrientation, useFixedBase=True)
        # jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, self.startGripPos, [1,0,0,0])[:self.arm_eef_index]
        for i in range(self.num_joints):
            p.resetJointState(self.xarm, i, self.joint_init_pos[i])
        # load goal
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_sphere.urdf')
        self.sphere = p.loadURDF(fullpath,useFixedBase=True)
        # load debug setting
        p.setDebugObjectColor(self.xarm, self.arm_eef_index,objectDebugColorRGB=[1, 0, 0])

        # gym setup
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        ))
        self.achieved_goal_index = len(obs['observation'])
        self.desired_goal_index = len(obs['observation']) + len(obs['achieved_goal'])
        p.stepSimulation()

    # basic methods
    # -------------------------
    def step(self, action):
        self.num_steps += 1
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        p.setGravity(0,0,-9.8)
        p.stepSimulation()
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
            'future_length': self._max_episode_steps - self.num_steps
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        done = (self.num_steps == self._max_episode_steps)
        return obs, reward, done, info

    def reset(self):
        super(XarmReachEnv, self).reset()
        self._reset_sim()
        self.goal = self._sample_goal()
        self.d_old = np.linalg.norm(p.getLinkState(self.xarm, self.gripper_base_index)[0] - self.goal, axis=-1)
        self.num_steps = 0
        return self._get_obs()

    # GoalEnv methods
    # -------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'sparse':
            return (d < self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            return -d
        elif self.reward_type == 'dense_diff':
            delta = self.d_old - d
            self.d_old = d
            return delta

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        self.if_render = True

    def close(self):
        p.disconnect()

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        assert action.shape == (4,), 'action shape error'
        cur_pos = np.array(p.getLinkState(self.xarm, self.arm_eef_index)[0])
        new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
        new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
        cur_gripper_pos = p.getJointState(self.xarm, self.gripper_driver_index)[0]
        new_gripper_pos = cur_gripper_pos + action[3]*self.dt * self.max_gripper_vel
        jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, new_pos, [1,0,0,0], maxNumIterations = self.n_substeps)
        for i in range(1, self.arm_eef_index):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1],force=5 * 240.)
        for i in range(self.gripper_driver_index, self.num_joints):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, new_gripper_pos,force=5 * 240.)

    def _get_obs(self):
        # robot state
        robot_state = p.getJointStates(self.xarm, np.arange(0,self.num_joints))
        # gripper state
        gripper_pos = np.array([robot_state[self.gripper_driver_index][0]])
        gripper_vel = np.array([robot_state[self.gripper_driver_index][1]])
        grip_state = p.getLinkState(self.xarm, self.gripper_base_index, computeLinkVelocity=1)
        grip_pos = np.array(grip_state[0])
        grip_velp = np.array(grip_state[6])
        # observation
        obs = np.concatenate((
                    grip_pos, grip_velp, gripper_pos, gripper_vel
        ))
        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(grip_pos.copy()),
            'desired_goal': self.goal.copy()
        }

    def _reset_sim(self):
        # reset arm
        for i in range(self.num_joints):
            p.resetJointState(self.xarm, i, self.joint_init_pos[i])
        p.stepSimulation()
        return True

    def _sample_goal(self):
        goal = np.array(self.goal_space.sample())
        p.resetBasePositionAndOrientation(self.sphere, goal, self.startOrientation)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - self.goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)