import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

'''
Uses Panda Gripper
'''

class XarmPDPickAndPlaceEnv(gym.GoalEnv):
    def __init__(self, config):
        # bullet paramters
        self.timeStep=1./60
        self.n_substeps = 15
        self.dt = self.timeStep*self.n_substeps
        # robot parameters
        self.distance_threshold=0.05
        self.num_joints = 13
        self.arm_eef_index = 8
        self.gripper_base_index = 9
        self.finger1_index = 10
        self.finger2_index = 11
        self.grasp_index = 12
        self.reward_type = 'sparse'
        self.pos_space = spaces.Box(low=np.array([0.3, -0.3 ,0.125]), high=np.array([0.5, 0.3, 0.4]))
        self.goal_space = spaces.Box(low=np.array([0.35, -0.25, 0.025]),high=np.array([0.45, 0.25, 0.27]))
        self.obj_space = spaces.Box(low=np.array([0.35, -0.25]), high=np.array([0.45, 0.25]))
        self.gripper_space = spaces.Box(low=0.021, high=0.04, shape=[1])
        self.max_vel = 0.25
        self.max_gripper_vel = 1
        self.height_offset = 0.025
        self.startPos = [0, 0, 0]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.joint_init_pos = [0, -0.009068751632859924, -0.08153217279952825, 0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 1.1524205092196147, -0.0004991403332530034] + [0]*5
        # training parameters
        self._max_episode_steps = 50
        
        # connect bullet
        p.connect(p.DIRECT) #or p.DIRECT for non-graphical version
        self.if_render = False

        # bullet setup
        self.seed()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        # load table
        self.table = p.loadURDF("table/table.urdf", [0,0,-0.625], useFixedBase=True)
        # load lego
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_cube.urdf')
        lego_pos = np.concatenate((self.obj_space.sample(), [self.height_offset]))
        self.lego = p.loadURDF(fullpath,lego_pos)
        # load arm
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7_pd.urdf')
        self.xarm = p.loadURDF(fullpath, self.startPos, self.startOrientation, useFixedBase=True)
        c = p.createConstraint(self.xarm,self.finger1_index,self.xarm,self.finger2_index,jointType=p.JOINT_GEAR,jointAxis=[1, 0, 0],parentFramePosition=[0, 0, 0],childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
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
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

        p.stepSimulation()
        p.setRealTimeSimulation(True)

    # basic methods
    # -------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        p.setGravity(0,0,-9.8)
        p.stepSimulation()
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        done = False
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
        return obs, reward, done, info

    def reset(self):
        super(XarmPDPickAndPlaceEnv, self).reset()
        self._reset_sim()
        self.goal = self._sample_goal()
        return self._get_obs()

    # GoalEnv methods
    # -------------------------

    def compute_reward(self, achieved_goal, goal, info):
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        # p.connect(p.GUI)
        self.if_render = True

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        assert action.shape == (4,), 'action shape error'
        cur_pos = np.array(p.getLinkState(self.xarm, self.arm_eef_index)[0])
        new_pos = cur_pos + np.array(action[:3]) * self.max_vel * self.dt
        new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
        cur_gripper_pos = p.getJointState(self.xarm, self.finger1_index)[0]
        new_gripper_pos = np.clip(cur_gripper_pos + action[3]*self.dt * self.max_gripper_vel, self.gripper_space.low, self.gripper_space.high)
        jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, new_pos, [1,0,0,0], maxNumIterations = self.n_substeps)
        for i in range(1, self.arm_eef_index):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
        p.setJointMotorControl2(self.xarm, self.finger1_index, p.POSITION_CONTROL, new_gripper_pos)
        p.setJointMotorControl2(self.xarm, self.finger2_index, p.POSITION_CONTROL, new_gripper_pos)

    def _get_obs(self):
        # robot state
        robot_state = p.getJointStates(self.xarm, np.arange(0,self.num_joints))
        # gripper state
        gripper_pos = np.array([robot_state[self.finger1_index][0]])
        gripper_vel = np.array([robot_state[self.finger1_index][1]])
        grip_state = p.getLinkState(self.xarm, self.gripper_base_index, computeLinkVelocity=1)
        grip_pos = np.array(grip_state[0])
        grip_velp = np.array(grip_state[6])
        # object state
        obj_pos = np.array(p.getBasePositionAndOrientation(self.lego)[0])
        obj_rot = np.array(p.getBasePositionAndOrientation(self.lego)[1])
        obj_velp = np.array(p.getBaseVelocity(self.lego)[0]) - grip_velp
        obj_velr = np.array(p.getBaseVelocity(self.lego)[1])
        obj_rel_pos = obj_pos - grip_pos
        # observation
        obs = np.concatenate((
                    obj_pos, obj_rel_pos, obj_rot, obj_velp, obj_velr,
                    grip_pos, grip_velp, gripper_pos, gripper_vel
        ))
        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(obj_pos.copy()),
            'desired_goal': self.goal.copy()
        }

    def _reset_sim(self):
        # reset arm
        for i in range(self.num_joints):
            p.resetJointState(self.xarm, i, self.joint_init_pos[i])
        # randomize position of lego
        lego_pos = np.concatenate((self.obj_space.sample(), [self.height_offset]))
        p.resetBasePositionAndOrientation(self.lego, lego_pos, self.startOrientation)
        p.stepSimulation()
        return True

    def _sample_goal(self):
        goal = np.array(self.goal_space.sample())
        p.resetBasePositionAndOrientation(self.sphere, goal, self.startOrientation)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - self.goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)