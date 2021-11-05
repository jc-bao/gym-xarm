import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd
import time
from gym.wrappers.monitoring import video_recorder

'''
Uses Panda Gripper
[TODO] tune to make it successful at first grasp
'''

class XarmPickAndPlaceNew(gym.GoalEnv):
    def __init__(self, config):
        # env parameter
        self.num_steps = 0
        self.init_grasp_rate = config['init_grasp_rate']
        self.goal_ground_rate = config['goal_ground_rate']
        self.reward_type = config['reward_type']
        self.action_type = config['action_type']
        self.grasp_mode = config['grasp_mode'] # [TODO] add support to continous control, now only support multi discrete
        self.metadata = {
            "render.modes": ["rgb_array"],
            "video.frames_per_second": int(30),
        }
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
        self.pos_space = spaces.Box(low=np.array([0.3, -0.3 ,0.15]), high=np.array([0.5, 0.3, 0.4]))
        self.goal_space = spaces.Box(low=np.array([0.35, -0.25, 0.025]),high=np.array([0.45, 0.25, 0.27]))
        self.obj_space = spaces.Box(low=np.array([0.35, -0.25]), high=np.array([0.45, 0.25]))
        self.gripper_space = spaces.Box(low=0.01, high=0.04, shape=[1])
        self.max_vel = 0.25
        self.max_gripper_vel = 0.08
        self.height_offset = 0.025
        self.startBasePos = [0, 0, 0]
        self.startGripperPos = [0.4, 0., 0.12]
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.eef2grip_offset = [0,0,0.088-0.021]
        # training parameters
        self._max_episode_steps = 50
        
        # connect bullet
        p.connect(p.DIRECT)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])

        # bullet setup
        self.seed()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        #  table
        self.table = p.loadURDF("table/table.urdf", [0,0,-0.625], useFixedBase=True)
        # load lego
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_cube.urdf')
        lego_pos = np.concatenate((self.obj_space.sample(), [self.height_offset]))
        self.lego = p.loadURDF(fullpath,lego_pos)
        # load arm
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7_pd.urdf')
        self.xarm = p.loadURDF(fullpath, self.startBasePos, self.startOrientation, useFixedBase=True)
        c = p.createConstraint(self.xarm,self.finger1_index,self.xarm,self.finger2_index,jointType=p.JOINT_GEAR,jointAxis=[1, 0, 0],parentFramePosition=[0, 0, 0],childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        # load goal
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_sphere.urdf')
        self.sphere = p.loadURDF(fullpath,useFixedBase=True)
        # load debug setting
        p.setDebugObjectColor(self.xarm, self.arm_eef_index,objectDebugColorRGB=[1, 0, 0])

        # gym setup
        self.goal = self._sample_goal()
        obs = self._get_obs()
        if self.action_type == 'continous':
            ''' continous action space
            [0]v_x   [1]v_y   [2]v_z   [3]gripper vel
            '''
            self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        elif self.action_type == 'discrete':
            ''' discrete action space
            [0]v_x=-1 [1]v_x=-0.5 [2]v_x=0 [3]v_x=0.5 [4]v_x=1
            [5]v_y=-1 [6]v_y=-0.5 [7]v_y=0 [8]v_y=0.5 [9]v_y=1
            [10]v_z=-1 [11]v_z=-0.5 [12]v_z=0 [13]v_z=0.5 [14]v_z=1
            [15]v_gripper = -1 [16]v_gripper = 0 [17]v_gripper = 1
            '''
            self.action_table = [
                [-1, 0, 0, 0],
                [-0.5, 0, 0, 0],
                [0, 0, 0, 0],
                [0.5, 0, 0, 0],
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, -0.5, 0, 0],
                [0, 0, 0, 0],
                [0, 0.5, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, -0.5, 0],
                [0, 0, 0, 0],
                [0, 0, 0.5, 0],
                [0, 0, 0, -1],
                [0, 0, 0, 0],
                [0, 0, 0, 1],
            ]
            self.action_space = spaces.Discrete(18)
        elif self.action_type == 'multi_discrete':
            if self.grasp_mode == 'easy':
                self.action_space = spaces.MultiDiscrete([20, 20, 20, 2])
            else: 
                self.action_space = spaces.MultiDiscrete([20, 20, 20, 8])
        else:
            raise NotImplementedError
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
        self.num_steps += 1
        self._set_action(action)
        p.setGravity(0,0,-9.8)
        p.stepSimulation()
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        done = (np.linalg.norm(obs['achieved_goal'] - self.goal, axis=-1) < self.distance_threshold) or self.num_steps == self._max_episode_steps
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, self.if_render) enble if want to control rendering 
        return obs, reward, done, info

    def reset(self):
        super(XarmPickAndPlaceNew, self).reset()
        self.num_steps = 0
        self._reset_sim()
        self.goal = self._sample_goal()
        self.d_old = np.linalg.norm(p.getBasePositionAndOrientation(self.lego)[0] - self.goal, axis=-1)
        return self._get_obs()

    def render(self, mode="rgb_array", width=500, height=500):
        if mode == 'rgb_array':
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=(0.3, 0, 0.2),
                distance=1.2,
                yaw=45,
                pitch=-10,
                roll=0,
                upAxisIndex=2,
            )
            proj_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=1.0, nearVal=0.1, farVal=100.0
            )
            (_, _, px, _, _) = p.getCameraImage(
                width=width, height=height, viewMatrix=view_matrix, projectionMatrix=proj_matrix,
            )
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (height, width, 4))
            
            return rgb_array
        else:
            raise NotImplementedError 

    # GoalEnv methods
    # -------------------------

    def compute_reward(self, achieved_goal, goal, info):
        ''' dense reward
        1. Xarm1 Reaching [0, 0.25]
        2. Xarm1 Grasping {0, 0.5}
        3. Xarm1 Lefting {0, 1.0}
        4. Xarm1 Hovering {0, [1.0, 1.25]}
        '''
        d_og = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'sparse':
            return (d_og < self.distance_threshold).astype(np.float32)
        elif self.reward_type == 'dense':
            if_grasp = len(p.getContactPoints(self.xarm, self.lego, self.finger1_index))!=0 and len(p.getContactPoints(self.xarm, self.lego, self.finger2_index))!=0
            grip_pos = np.array(p.getLinkState(self.xarm, self.gripper_base_index)[0])-self.eef2grip_offset
            d_ao = np.linalg.norm(grip_pos - achieved_goal + [0.06,0,0])
            if not if_grasp:
                return 0.25 * (1 - np.tanh(1.0 * d_ao))
            elif achieved_goal[2] > 0.05:
                return (1.0 + 0.25*(1 - np.tanh(1.0 * d_og)))
            else:
                return 0.5
        elif self.reward_type == 'dense_o2g':
            return -d_og
        elif self.reward_type == 'dense_diff_o2g':
            reward = self.d_old - self.d_og
            self.d_old = self.d_og
            return reward
        else:
            raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        if self.action_type == 'continous':
            assert action.shape == (4,), 'action shape error'
            vel_control = np.clip(action, self.action_space.low, self.action_space.high)
        elif self.action_type == 'discrete':
            vel_control = self.action_table[action]
        elif self.action_type == 'multi_discrete':
            if self.grasp_mode == 'easy':
                vel_control = np.append(action[:3]/10 - 1, action[-1]) # grasp 0:open 1:close
            else:
                vel_control = np.append(action[:3]/10 - 1, action[-1]/4 - 1)
        else: 
            raise NotImplementedError
        cur_pos = np.array(p.getLinkState(self.xarm, self.arm_eef_index)[0])
        new_pos = cur_pos + np.array(vel_control[:3]) * self.max_vel * self.dt
        new_pos = np.clip(new_pos, self.pos_space.low, self.pos_space.high)
        if self.grasp_mode == 'easy':
            new_gripper_pos = 0.024 if vel_control[-1] else 0.04
            friction = 100 if vel_control[-1] else 0
            p.changeDynamics(self.xarm, self.finger1_index, lateralFriction = friction, spinningFriction = friction, rollingFriction = friction)
            p.changeDynamics(self.xarm, self.finger2_index, lateralFriction = friction, spinningFriction = friction, rollingFriction = friction)
        else:
            cur_gripper_pos = p.getJointState(self.xarm, self.finger1_index)[0]
            new_gripper_pos = np.clip(cur_gripper_pos + vel_control[3]*self.dt * self.max_gripper_vel, self.gripper_space.low, self.gripper_space.high)
        jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, new_pos, [1,0,0,0], maxNumIterations = self.n_substeps)
        for i in range(1, self.arm_eef_index):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
        p.setJointMotorControl2(self.xarm, self.finger1_index, p.POSITION_CONTROL, new_gripper_pos, force=1000)
        p.setJointMotorControl2(self.xarm, self.finger2_index, p.POSITION_CONTROL, new_gripper_pos, force=1000)
        if len(p.getContactPoints(self.xarm, self.lego, self.finger1_index))!=0 and len(p.getContactPoints(self.xarm, self.lego, self.finger2_index))!=0: # grasp success -> change friction
            p.changeDynamics(self.xarm, self.finger1_index, lateralFriction = 100)
            p.changeDynamics(self.xarm, self.finger2_index, lateralFriction = 100)
        else:
            p.changeDynamics(self.xarm, self.finger1_index, lateralFriction = 1)
            p.changeDynamics(self.xarm, self.finger2_index, lateralFriction = 1)

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
        for _ in range(5): 
            jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, self.startGripperPos, [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
            p.setJointMotorControl2(self.xarm, self.finger1_index, p.POSITION_CONTROL, 0.02, force=1000)
            p.setJointMotorControl2(self.xarm, self.finger2_index, p.POSITION_CONTROL, 0.02, force=1000)
            p.stepSimulation()
        # randomize position of lego
        if np.random.random() < self.init_grasp_rate: # init in hand
            lego_pos = np.concatenate((self.startGripperPos[:2], [self.height_offset]))
        else: 
            lego_pos = np.concatenate((self.obj_space.sample(), [self.height_offset]))
        p.resetBasePositionAndOrientation(self.lego, lego_pos, self.startOrientation)
        p.stepSimulation()
        return True

    def _sample_goal(self):
        goal = np.array(self.goal_space.sample())
        if np.random.random() < self.goal_ground_rate:
            goal[-1] = self.goal_space.low[-1]
        p.resetBasePositionAndOrientation(self.sphere, goal, self.startOrientation)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - self.goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _run_demo(self, recorder):
        self.reset()
        p.resetBasePositionAndOrientation(self.lego, [0.4,0,0.025], self.startOrientation)
        p.setGravity(0,0,-9.8)
        # push
        # for i in range(30):
        #     p.setJointMotorControl2(self.xarm, self.finger1_index, p.POSITION_CONTROL, 0.01)
        #     p.setJointMotorControl2(self.xarm, self.finger2_index, p.POSITION_CONTROL, 0.01)
        #     jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, [i*0.025, 0, 0.125], [1,0,0,0], maxNumIterations = self.n_substeps)
        #     for i in range(1, self.arm_eef_index):
        #         p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
        #     p.stepSimulation()
        #     obs = self._get_obs()
        #     info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
        #     recorder.capture_frame()
        for _ in range(10):
            jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, [0.4, 0, 0.125], [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            recorder.capture_frame()
        for _ in range(3):
            p.setJointMotorControl2(self.xarm, self.finger1_index, p.POSITION_CONTROL, 0.02)
            p.setJointMotorControl2(self.xarm, self.finger2_index, p.POSITION_CONTROL, 0.02)
            if len(p.getContactPoints(self.xarm, self.lego, self.finger1_index))!=0 and len(p.getContactPoints(self.xarm, self.lego, self.finger2_index))!=0: # grasp success -> change friction
                p.changeDynamics(self.xarm, self.finger1_index, lateralFriction = 100)
                p.changeDynamics(self.xarm, self.finger2_index, lateralFriction = 100)
            else:
                p.changeDynamics(self.xarm, self.finger1_index, lateralFriction = 1)
                p.changeDynamics(self.xarm, self.finger2_index, lateralFriction = 1)
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            recorder.capture_frame()
        # move both
        for i in range(10):
            jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, [0.3, 0, 0.3], [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
            if len(p.getContactPoints(self.xarm, self.lego, self.finger1_index))!=0 and len(p.getContactPoints(self.xarm, self.lego, self.finger2_index))!=0: # grasp success -> change friction
                p.changeDynamics(self.xarm, self.finger1_index, lateralFriction = 100)
                p.changeDynamics(self.xarm, self.finger2_index, lateralFriction = 100)
            else:
                p.changeDynamics(self.xarm, self.finger1_index, lateralFriction = 1)
                p.changeDynamics(self.xarm, self.finger2_index, lateralFriction = 1)
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            recorder.capture_frame()

if __name__ == '__main__':
    gym.logger.set_level(10)
    config = {
        'init_grasp_rate': 0.5,
        'goal_ground_rate': 0.5,
        'reward_type': 'sparse', 
        'action_type': 'continous', 
        'grasp_mode': 'hard'
    }
    env=XarmPickAndPlaceNew(config)
    video_recorder = video_recorder.VideoRecorder(
            env = env, path='/Users/reedpan/Desktop/Research/gym-xarm/gym_xarm/envs/video/pac_demo.mp4'
    )
    # env._run_demo(recorder = video_recorder)
    for _ in range(10):
        env.reset()
        for _ in range(50):
            act = env.action_space.sample()
            # act[-1] = 0
            env.step(act)
            env.render()
            video_recorder.capture_frame()
    video_recorder.close()