import time
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from pybullet_utils import bullet_client
import pybullet_data as pd
import pybullet
try:
  if os.environ["PYBULLET_EGL"]:
    import pkgutil
except:
  pass
'''
Uses Panda Gripper to handoover
'''

class XarmHandover(gym.GoalEnv):
    _num_client = 0
    @property
    def num_client(self): return type(self)._num_client
    @num_client.setter
    def num_client(self, val): type(self)._num_client = val
    def __init__(self, config):
        # bullet paramters
        self.config = config
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
        self.pos_space_1 = spaces.Box(low=np.array([-0.35, -0.2 ,0.1]), high=np.array([0.0, 0.2, 0.2]), dtype=np.float32)
        self.pos_space_2 = spaces.Box(low=np.array([0.0, -0.2 ,0.1]), high=np.array([0.35, 0.2, 0.2]), dtype=np.float32)
        self.goal_space = spaces.Box(low=np.array([0.1, -0.2, 0.025]),high=np.array([0.31, 0.2, 0.25]), dtype=np.float32) 
        self.obj_space = spaces.Box(low=np.array([0.12, -0.2]), high=np.array([0.35, 0.2]), dtype=np.float32) 
        self.gripper_space = spaces.Box(low=0.020, high=0.04, shape=[1], dtype=np.float32)
        self.max_vel = 0.25
        self.max_gripper_vel = 1
        self.height_offset = 0.025
        self.eef2grip_offset = [0,0,0.088-0.021]
        self.startPos_1 = [-0.65, 0, 0]
        self.startPos_2 = [0.65, 0, 0]
        self.startOrientation_1 = pybullet.getQuaternionFromEuler([0,0,0])
        self.startOrientation_2 = pybullet.getQuaternionFromEuler([0,0,np.pi])
        self.joint_init_pos = [0, -0.009068751632859924, -0.08153217279952825, 0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 1.1524205092196147, -0.0004991403332530034] + [0]*2 + [0.04]*2 + [0]
        self.eff_init_pos_1 = [-0.2, 0.0, 0.2]
        self.eff_init_pos_2 = [0.2, 0.0, 0.2]
        self.lego_length = 0.15

        # connect bullet
        if self.config['GUI']:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, False)
        self._p.setRealTimeSimulation(True)
        self._p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0,0,0.2])

        # training parameters
        self._max_episode_steps = 100
        self.num_steps = 0

        # bullet setup
        self.seed()
        self._p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        self._p.setAdditionalSearchPath(pd.getDataPath())
        self._p.setTimeStep(self.timeStep)
        self._p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        self._p.setGravity(0,0,-9.8)
        # load ground
        self.ground = self._p.loadURDF("plane.urdf", [0, 0, -0.625])
        # load table
        self.table_1= self._p.loadURDF("table/table.urdf", [-0.85,0,-0.625], useFixedBase=True)
        # self.table_2 = self._p.loadURDF("table/table.urdf", [0,0,-0.625], useFixedBase=True)
        self.table_3 = self._p.loadURDF("table/table.urdf", [0.85,0,-0.625], useFixedBase=True)
        # load lego
        self.colors = [np.random.sample(size = 3).tolist() + [1] for _ in range(self.config['num_obj'])]
        self.legos = [None] * self.config['num_obj']
        for i in range(self.config['num_obj']):
            lg_v = self._p.createVisualShape(shapeType=self._p.GEOM_BOX, halfExtents = [self.lego_length/2, 0.025, 0.025], rgbaColor = self.colors[i])
            lg_c = self._p.createCollisionShape(shapeType=self._p.GEOM_BOX, halfExtents = [self.lego_length/2, 0.025, 0.025])
            lego_pos = np.concatenate((self.obj_space.sample(), [self.height_offset]))
            self.legos[i] = self._p.createMultiBody(baseVisualShapeIndex=lg_v, baseCollisionShapeIndex = lg_c, baseMass = 0.5, basePosition=lego_pos, baseOrientation = self.startOrientation_1)
        # load arm
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7_pd.urdf')
        self.xarm_1 = self._p.loadURDF(fullpath, self.startPos_1, self.startOrientation_1, useFixedBase=True)
        self.xarm_2 = self._p.loadURDF(fullpath, self.startPos_2, self.startOrientation_2, useFixedBase=True)
        c = self._p.createConstraint(self.xarm_1,self.finger1_index,self.xarm_1,self.finger2_index,jointType=self._p.JOINT_GEAR,jointAxis=[1, 0, 0],parentFramePosition=[0, 0, 0],childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        c = self._p.createConstraint(self.xarm_2,self.finger1_index,self.xarm_2,self.finger2_index,jointType=self._p.JOINT_GEAR,jointAxis=[1, 0, 0],parentFramePosition=[0, 0, 0],childFramePosition=[0, 0, 0])
        self._p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        for i in range(self.num_joints):
            self._p.resetJointState(self.xarm_1, i, self.joint_init_pos[i])
            self._p.resetJointState(self.xarm_2, i, self.joint_init_pos[i])
        # load goal
        self.spheres = [None] * self.config['num_obj']
        for i in range(self.config['num_obj']):
            sp = self._p.createVisualShape(shapeType=self._p.GEOM_SPHERE, radius = 0.03, rgbaColor = self.colors[i])
            self.spheres[i] = self._p.createMultiBody(baseVisualShapeIndex=sp)
        # load debug setting
        self._p.setDebugObjectColor(self.xarm_1, self.arm_eef_index,objectDebugColorRGB=[1, 0, 0])
        self._p.setDebugObjectColor(self.xarm_2, self.arm_eef_index,objectDebugColorRGB=[1, 0, 0])
        # gym setup
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(8,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self._p.stepSimulation()

    # basic methods
    # -------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self._p.stepSimulation()
        obs = self._get_obs()
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        done = (self.num_steps == self._max_episode_steps) or info['is_success']
        return obs, reward, done, info

    def reset(self):
        super(XarmHandover, self).reset()
        self._reset_sim()
        self.goal = self._sample_goal()
        return self._get_obs()
    
    def close(self):
        pybullet.disconnect(self._p._client)

    # GoalEnv methods
    # -------------------------

    def compute_reward(self, achieved_goal, goal, info):
        """
        dense reward:
            1. Xarm1 Reaching [0, 0.25]
            2. Xarm1 Grasping {0, 0.5}
            3. Xarm1 Lefting {0, 1.0}
            4. Xarm1 Hovering {0, [1.0, 1.25]}
            5. Mutual Grasping {0, 1.5}
            6. Handover {0, 2.0}
            7. Xarm2 Reaching {0, [2.0, 2.25]}
        """
        if self.reward_type == 'sparse':
            reward = - self.config['num_obj']
            grip_pos_1 = achieved_goal[-6:-3]
            grip_pos_2 = achieved_goal[-3:]
            for i in range(self.config['num_obj']):
                d = np.linalg.norm(achieved_goal[i*3:i*3+2] - goal[i*3:i*3+2], axis=-1)
                if d < self.distance_threshold:
                    reward += 1
                    # if the object is set, encourage the gripper leave
                    # if np.linalg.norm(achieved_goal[i*3:i*3+3]-grip_pos_1)>self.distance_threshold*2 and \
                    #     np.linalg.norm(achieved_goal[i*3:i*3+3]-grip_pos_2)>self.distance_threshold*2:
                    #     reward += 1
            return reward
        else:
            grip_pos_1 = np.array(self._p.getLinkState(self.xarm_1, self.gripper_base_index)[0])-self.eef2grip_offset
            grip_pos_2 = np.array(self._p.getLinkState(self.xarm_2, self.gripper_base_index)[0])-self.eef2grip_offset
            dist_12lego = np.linalg.norm(grip_pos_1 - achieved_goal + [0.06,0,0])
            dist_22lego = np.linalg.norm(grip_pos_2 - achieved_goal + [-0.06,0,0])
            if not self.if_xarm1_grasp and not self.if_xarm2_grasp:
                return 0.25 * (1 - np.tanh(1.0 * dist_12lego)) / 2.25
            elif self.if_xarm1_grasp and not self.if_xarm2_grasp:
                if achieved_goal[2] > 0.05: 
                    return (1.0 + 0.25*(1 - np.tanh(1.0 * dist_22lego))) / 2.25
                else: 
                    return 0.5 / 2.25
            elif self.if_xarm1_grasp and self.if_xarm2_grasp:
                return 1.5 / 2.25
            else:
                return (2.0 + 0.25*(1 - np.tanh(1.0 * d)))/2.25

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        raise NotImplementedError

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        assert action.shape == (8,), 'action shape error'
        # move joint
        cur_pos_1 = np.array(self._p.getLinkState(self.xarm_1, self.arm_eef_index)[0])
        cur_pos_2 = np.array(self._p.getLinkState(self.xarm_2, self.arm_eef_index)[0])
        new_pos_1 = cur_pos_1 + np.array(action[0:3]) * self.max_vel * self.dt
        new_pos_1 = np.clip(new_pos_1, self.pos_space_1.low, self.pos_space_1.high)
        new_pos_2 = cur_pos_2 + np.array(action[4:7]) * self.max_vel * self.dt
        new_pos_2 = np.clip(new_pos_2, self.pos_space_2.low, self.pos_space_2.high)
        cur_gripper_pos_1 = self._p.getJointState(self.xarm_1, self.finger1_index)[0]
        new_gripper_pos_1 = np.clip(cur_gripper_pos_1 + action[3]*self.dt * self.max_gripper_vel, self.gripper_space.low, self.gripper_space.high)
        cur_gripper_pos_2 = self._p.getJointState(self.xarm_2, self.finger1_index)[0]
        new_gripper_pos_2 = np.clip(cur_gripper_pos_2 + action[7]*self.dt * self.max_gripper_vel, self.gripper_space.low, self.gripper_space.high)
        jointPoses_1 = self._p.calculateInverseKinematics(self.xarm_1, self.arm_eef_index, new_pos_1, [1,0,0,0], maxNumIterations = self.n_substeps)
        jointPoses_2 = self._p.calculateInverseKinematics(self.xarm_2, self.arm_eef_index, new_pos_2, [1,0,0,0], maxNumIterations = self.n_substeps)
        for i in range(1, self.arm_eef_index):
            self._p.setJointMotorControl2(self.xarm_1, i, self._p.POSITION_CONTROL, jointPoses_1[i-1]) # max=1200
            self._p.setJointMotorControl2(self.xarm_2, i, self._p.POSITION_CONTROL, jointPoses_2[i-1]) # max=1200
        # move gripper
        self.if_xarm1_grasp = len(self._p.getContactPoints(self.xarm_1, self.legos[0], self.finger1_index))!=0 and len(self._p.getContactPoints(self.xarm_1, self.legos[0], self.finger2_index))!=0
        self.if_xarm2_grasp = len(self._p.getContactPoints(self.xarm_2, self.legos[0], self.finger1_index))!=0 and len(self._p.getContactPoints(self.xarm_2, self.legos[0], self.finger2_index))!=0
        self._p.setJointMotorControl2(self.xarm_1, self.finger1_index, self._p.POSITION_CONTROL, new_gripper_pos_1)
        self._p.setJointMotorControl2(self.xarm_1, self.finger2_index, self._p.POSITION_CONTROL, new_gripper_pos_1)
        self._p.setJointMotorControl2(self.xarm_2, self.finger1_index, self._p.POSITION_CONTROL, new_gripper_pos_2)
        self._p.setJointMotorControl2(self.xarm_2, self.finger2_index, self._p.POSITION_CONTROL, new_gripper_pos_2)
        if len(self._p.getContactPoints(self.xarm_1, self.legos[0], self.finger1_index))!=0 and len(self._p.getContactPoints(self.xarm_1, self.legos[0], self.finger2_index))!=0: # grasp success -> change friction
            self._p.changeDynamics(self.xarm_1, self.finger1_index, lateralFriction = 100)
            self._p.changeDynamics(self.xarm_1, self.finger2_index, lateralFriction = 100)
        else:
            self._p.changeDynamics(self.xarm_1, self.finger1_index, lateralFriction = 1)
            self._p.changeDynamics(self.xarm_1, self.finger2_index, lateralFriction = 1)
        if len(self._p.getContactPoints(self.xarm_2, self.legos[0], self.finger1_index))!=0 and len(self._p.getContactPoints(self.xarm_2, self.legos[0], self.finger2_index))!=0: # grasp success -> change friction
            self._p.changeDynamics(self.xarm_2, self.finger1_index, lateralFriction = 100)
            self._p.changeDynamics(self.xarm_2, self.finger2_index, lateralFriction = 100)
        else:
            self._p.changeDynamics(self.xarm_2, self.finger1_index, lateralFriction = 1)
            self._p.changeDynamics(self.xarm_2, self.finger2_index, lateralFriction = 1)
        # reset lego pos to aviod fly away and aviod change direction
        for i in range(self.config['num_obj']):
            lego_pos = self._p.getBasePositionAndOrientation(self.legos[i])[0]
            lego_ori = np.array(self._p.getBasePositionAndOrientation(self.legos[i])[1])
            lego_ori = [0, self._p.getEulerFromQuaternion(lego_ori)[1], 0]
            lego_ori = self._p.getQuaternionFromEuler(lego_ori)
            self._p.resetBasePositionAndOrientation(self.legos[i], lego_pos, lego_ori)

    def _get_obs(self):
        # robot state
        robot_state_1 = self._p.getJointStates(self.xarm_1, np.arange(0,self.num_joints))
        robot_state_2 = self._p.getJointStates(self.xarm_2, np.arange(0,self.num_joints))
        # gripper state
        gripper_pos_1 = np.array([robot_state_1[self.finger1_index][0]])
        gripper_vel_1 = np.array([robot_state_1[self.finger1_index][1]])
        gripper_pos_2 = np.array([robot_state_2[self.finger1_index][0]])
        gripper_vel_2 = np.array([robot_state_2[self.finger1_index][1]])
        grip_state_1 = self._p.getLinkState(self.xarm_1, self.gripper_base_index, computeLinkVelocity=1)
        grip_state_2 = self._p.getLinkState(self.xarm_2, self.gripper_base_index, computeLinkVelocity=1)
        grip_pos_1 = np.array(grip_state_1[0])-self.eef2grip_offset
        grip_pos_2 = np.array(grip_state_2[0])-self.eef2grip_offset
        grip_velp_1 = np.array(grip_state_1[6])
        grip_velp_2 = np.array(grip_state_2[6])
        # object state
        obj_pos = np.array(self._p.getBasePositionAndOrientation(self.legos[0])[0])
        obj_rot = np.array(self._p.getBasePositionAndOrientation(self.legos[0])[1])
        obj_velp = np.array(self._p.getBaseVelocity(self.legos[0])[0])
        obj_velr = np.array(self._p.getBaseVelocity(self.legos[0])[1])
        for i in range(1, self.config['num_obj']):
            obj_pos = np.concatenate((obj_pos, self._p.getBasePositionAndOrientation(self.legos[i])[0]))
            obj_rot = np.concatenate((obj_rot, self._p.getBasePositionAndOrientation(self.legos[i])[1]))
            obj_velp = np.concatenate((obj_velp, self._p.getBaseVelocity(self.legos[i])[0]))
            obj_velr = np.concatenate((obj_velr, self._p.getBaseVelocity(self.legos[i])[1]))
        # final obs
        obs = np.concatenate((
            obj_pos, obj_rot, obj_velp, obj_velr,
            grip_pos_1, grip_velp_1, gripper_pos_1, gripper_vel_1,
            grip_pos_2, grip_velp_2, gripper_pos_2, gripper_vel_2
        ))
        # achieved_goal = np.squeeze(np.concatenate([obj_pos.copy(), grip_pos_1.copy(), grip_pos_2.copy()])) Q: if add gripper pos
        achieved_goal = np.squeeze(np.concatenate([obj_pos.copy()]))
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal,
            'desired_goal': self.goal.copy()
        }

    def _reset_sim(self):
        self.num_steps = 0
        # reset arm
        '''
        This way not work, reason still investigating...
        for i in range(self.num_joints): 
            self._p.resetJointState(self.xarm_1, i, self.joint_init_pos[i])
            self._p.resetJointState(self.xarm_2, i, self.joint_init_pos[i])
        '''
        for _ in range(5): 
            jointPoses_1 = self._p.calculateInverseKinematics(self.xarm_1, self.arm_eef_index, self.eff_init_pos_1, [1,0,0,0], maxNumIterations = self.n_substeps)
            jointPoses_2 = self._p.calculateInverseKinematics(self.xarm_2, self.arm_eef_index, self.eff_init_pos_2, [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                self._p.setJointMotorControl2(self.xarm_1, i, self._p.POSITION_CONTROL, jointPoses_1[i-1]) # max=1200
                self._p.setJointMotorControl2(self.xarm_2, i, self._p.POSITION_CONTROL, jointPoses_2[i-1]) # max=1200
            self._p.stepSimulation()
        # randomize position of lego
        pos = []
        for i in range(self.config['num_obj']):
            pos.append(self.obj_space.sample())
            if i > 0:
                while min([np.linalg.norm(pos[i][1]-pos[j][1]) for j in range(i)]) < 0.05:
                    pos[i] = self.obj_space.sample()
            if np.random.uniform() < 0.5:
                pos[i][0] = - pos[i][0]
            lego_pos = np.concatenate((pos[i], [self.height_offset]))
            self._p.resetBasePositionAndOrientation(self.legos[i], lego_pos, self.startOrientation_1)
        self._p.stepSimulation()
        self.if_xarm1_grasp = len(self._p.getContactPoints(self.xarm_1, self.legos[0], self.finger1_index))!=0 and len(self._p.getContactPoints(self.xarm_1, self.legos[0], self.finger2_index))!=0
        self.if_xarm2_grasp = len(self._p.getContactPoints(self.xarm_2, self.legos[0], self.finger1_index))!=0 and len(self._p.getContactPoints(self.xarm_2, self.legos[0], self.finger2_index))!=0
        return True

    def _sample_goal(self):
        goal = [None]*self.config['num_obj']
        for i in range(self.config['num_obj']):
            obj_pos = [self._p.getBasePositionAndOrientation(lego)[0] for lego in self.legos]
            goal[i] = self.goal_space.sample()
            if i > 0:
                min_dis2obj = min([np.linalg.norm(goal[i][:2] - obj_pos[k][:2]) for k in range(self.config['num_obj'])])
                while min([np.linalg.norm(goal[i][1]-goal[j][1]) for j in range(i)]) < 0.08 or min_dis2obj < 0.08:
                    goal[i] = self.goal_space.sample()
                    min_dis2obj = min([np.linalg.norm(goal[i][:2] - obj_pos[k][:2]) for k in range(self.config['num_obj'])])
            if_same_side = (np.random.uniform() < self.config['same_side_rate']) # 0.5: same side rate
            if (np.array(obj_pos[i][0] > 0) ^ if_same_side):
                goal[i][0] = -goal[i][0]
            if self.config['goal_shape'] == 'ground':
                goal[i][2] = self.height_offset
            self._p.resetBasePositionAndOrientation(self.spheres[i], goal[i], self.startOrientation_1)
        return np.array(goal).flatten()

    def _is_success(self, achieved_goal, desired_goal):
        state = True
        for i in range(self.config['num_obj']):
            d = np.linalg.norm(achieved_goal[i*3:i*3+3] - desired_goal[i*3:i*3+3], axis=-1)
            state = (d < self.distance_threshold) and state
        return float(state)

    def ezpolicy(self, obs):
        # observation
        obs = obs['observation']
        obj_pos = obs[0:3]
        obj_rot = obs[3:7] 
        obj_velp = obs[7:10] 
        obj_velr = obs[10:13]
        grip_pos_1 = obs[13:16] 
        grip_velp_1 = obs[16:19] 
        gripper_pos_1 = obs[19] 
        gripper_vel_1 = obs[20]
        grip_pos_2 = obs[21:24] 
        grip_velp_2 = obs[24:27] 
        gripper_pos_2 = obs[27] 
        gripper_vel_2 = obs[28]
        # state
        if_grasp_1 = gripper_pos_1 < 0.25 and np.linalg.norm(obj_pos - grip_pos_1)<0.05
        if_grasp_2 = gripper_pos_2 < 0.25 and np.linalg.norm(obj_pos - grip_pos_2)<0.05
        delta_1 = obj_pos - grip_pos_1 + [-0.07, 0, 0]
        dis_1 = np.linalg.norm(delta_1)
        delta_2 = obj_pos - grip_pos_2 + [0.07, 0, 0]
        dis_2 = np.linalg.norm(delta_2)
        delta_3 = grip_pos_2 - grip_pos_1 + [0, 0, 0]
        dis_3 = np.linalg.norm(delta_3)
        # action
        action = [0]*8
        if np.linalg.norm(obj_pos - grip_pos_1)<0.1:
            action[3] = -0.5
        else:
            action[3] = 0.5
        if np.linalg.norm(obj_pos - grip_pos_2)<0.1:
            action[7] = -0.5
        else:
            action[7] = 0.5
        if not if_grasp_1:
            action[0:3] = delta_1 / dis_1
        else:
            if not if_grasp_2:
                action[0:3] = [0.5, 0, 0.5]
                action[4:7] = delta_2 / dis_2
            else:
                action[4] = -0.5
        return action
    
if __name__ == '__main__':
    config = {
        'goal_shape': 'ground',
        'num_obj': 2,
        'GUI': True, 
        'same_side_rate': 0.5
    }
    env = XarmHandover(config)
    obs = env.reset()
    for i in range(10000):
        # action = env.ezpolicy(obs)
        action = env.action_space.sample()
        obs, *_= env.step(action)
        time.sleep(0.03)
        if i % 50 == 0:
            env.reset()