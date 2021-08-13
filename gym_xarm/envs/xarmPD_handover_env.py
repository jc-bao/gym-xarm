import time
import os
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import pybullet as p
import pybullet_data as pd

'''
Uses Panda Gripper to handoover
'''

class XarmPDHandover(gym.GoalEnv):
    def __init__(self):
        # bullet paramters
        self.timeStep=1./60
        self.n_substeps = 1
        self.dt = self.timeStep*self.n_substeps
        # robot parameters
        self.num_obj = 1
        self.distance_threshold=0.03 * self.num_obj
        self.num_joints = 13
        self.arm_eef_index = 8
        self.gripper_base_index = 9
        self.finger1_index = 10
        self.finger2_index = 11
        self.grasp_index = 12
        self.reward_type = 'sparse'
        self.pos_space_1 = spaces.Box(low=np.array([-0.4, 0.3 ,0.125]), high=np.array([0.1, 0.3, 0.4]), dtype=np.float32)
        self.pos_space_2 = spaces.Box(low=np.array([-0.1, -0.3 ,0.125]), high=np.array([0.4, 0.3, 0.4]), dtype=np.float32)
        self.goal_space = spaces.Box(low=np.array([0.3, -0.3, 0.025]),high=np.array([0.4, 0.3, 0.27]), dtype=np.float32)
        self.obj_space = spaces.Box(low=np.array([-0.35, -0.3]), high=np.array([-0.25, 0.3]), dtype=np.float32)
        self.gripper_space = spaces.Box(low=0.020, high=0.04, shape=[1], dtype=np.float32)
        self.max_vel = 0.25
        self.max_gripper_vel = 1
        self.height_offset = 0.025
        self.eef2grip_offset = [0,0,0.088-0.021]
        self.startPos_1 = [-0.6, 0, 0]
        self.startPos_2 = [0.6, 0, 0]
        self.startOrientation_1 = p.getQuaternionFromEuler([0,0,0])
        self.startOrientation_2 = p.getQuaternionFromEuler([0,0,np.pi])
        self.joint_init_pos = [0, -0.009068751632859924, -0.08153217279952825, 0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 1.1524205092196147, -0.0004991403332530034] + [0]*2 + [0.04]*2 + [0]
        self.lego_length = 0.2
        # training parameters
        self._max_episode_steps = 50
        
        # connect bullet
        p.connect(p.GUI) #or p.DIRECT for non-graphical version
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
        self.if_render = False

        # bullet setup
        self.seed()
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setTimeStep(self.timeStep)
        p.setPhysicsEngineParameter(numSubSteps = self.n_substeps)
        p.setGravity(0,0,-9.8)
        # load table
        # self.table_1= p.loadURDF("table/table.urdf", [-1.5,0,-0.625], useFixedBase=True)
        self.table_2 = p.loadURDF("table/table.urdf", [0,0,-0.625], useFixedBase=True)
        # self.table_3 = p.loadURDF("table/table.urdf", [1.5,0,-0.625], useFixedBase=True)
        # load lego
        self.colors = [np.random.sample(size = 3).tolist() + [1] for _ in range(self.num_obj)]
        self.legos = [None] * self.num_obj
        for i in range(self.num_obj):
            lg_v = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents = [self.lego_length/2, 0.025, 0.025], rgbaColor = self.colors[i])
            lg_c = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents = [self.lego_length/2, 0.025, 0.025])
            lego_pos = np.concatenate((self.obj_space.sample(), [self.height_offset]))
            self.legos[i] = p.createMultiBody(baseVisualShapeIndex=lg_v, baseCollisionShapeIndex = lg_c, baseMass = 0.1, basePosition=lego_pos, baseOrientation = self.startOrientation_1)
        # load arm
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7_pd.urdf')
        self.xarm_1 = p.loadURDF(fullpath, self.startPos_1, self.startOrientation_1, useFixedBase=True)
        self.xarm_2 = p.loadURDF(fullpath, self.startPos_2, self.startOrientation_2, useFixedBase=True)
        c = p.createConstraint(self.xarm_1,self.finger1_index,self.xarm_1,self.finger2_index,jointType=p.JOINT_GEAR,jointAxis=[1, 0, 0],parentFramePosition=[0, 0, 0],childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        c = p.createConstraint(self.xarm_2,self.finger1_index,self.xarm_2,self.finger2_index,jointType=p.JOINT_GEAR,jointAxis=[1, 0, 0],parentFramePosition=[0, 0, 0],childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
        for i in range(self.num_joints):
            p.resetJointState(self.xarm_1, i, self.joint_init_pos[i])
            p.resetJointState(self.xarm_2, i, self.joint_init_pos[i])
        # load goal
        self.spheres = [None] * self.num_obj
        for i in range(self.num_obj):
            sp = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius = 0.02, rgbaColor = self.colors[i])
            self.spheres[i] = p.createMultiBody(baseVisualShapeIndex=sp)
        # load debug setting
        p.setDebugObjectColor(self.xarm_1, self.arm_eef_index,objectDebugColorRGB=[1, 0, 0])
        p.setDebugObjectColor(self.xarm_2, self.arm_eef_index,objectDebugColorRGB=[1, 0, 0])
        # gym setup
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(8,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        p.stepSimulation()
        p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[-0.1,0.1,-0.1])
        p.setRealTimeSimulation(True)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)

    # basic methods
    # -------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
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
        super(XarmPDHandover, self).reset()
        self._reset_sim()
        self.goal = self._sample_goal()
        return self._get_obs()

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
        d = np.linalg.norm(achieved_goal - goal, axis=-1)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            if_xarm1_grasp = len(p.getContactPoints(self.xarm_1, self.legos[0], self.finger1_index))!=0 and len(p.getContactPoints(self.xarm_1, self.legos[0], self.finger2_index))!=0
            if_xarm2_grasp = len(p.getContactPoints(self.xarm_2, self.legos[0], self.finger1_index))!=0 and len(p.getContactPoints(self.xarm_2, self.legos[0], self.finger2_index))!=0
            grip_pos_1 = np.array(p.getLinkState(self.xarm_1, self.gripper_base_index)[0])-self.eef2grip_offset
            grip_pos_2 = np.array(p.getLinkState(self.xarm_2, self.gripper_base_index)[0])-self.eef2grip_offset
            dist_12lego = np.linalg.norm(grip_pos_1 - achieved_goal + [0.06,0,0])
            dist_22lego = np.linalg.norm(grip_pos_2 - achieved_goal + [-0.06,0,0])
            if not if_xarm1_grasp and not if_xarm2_grasp:
                return 0.25 * (1 - np.tanh(1.0 * dist_12lego)) / 2.25
            elif if_xarm1_grasp and not if_xarm2_grasp:
                if achieved_goal[2] > 0.05: 
                    return (1.0 + 0.25*(1 - np.tanh(1.0 * dist_22lego))) / 2.25
                else: 
                    return 0.5 / 2.25
            elif if_xarm1_grasp and if_xarm2_grasp:
                return 1.5 / 2.25
            else:
                return (2.0 + 0.25*(1 - np.tanh(1.0 * d)))/2.25

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        # p.connect(p.GUI)
        self.if_render = True

    # RobotEnv method
    # -------------------------

    def _set_action(self, action):
        assert action.shape == (8,), 'action shape error'
        cur_pos_1 = np.array(p.getLinkState(self.xarm_1, self.arm_eef_index)[0])
        cur_pos_2 = np.array(p.getLinkState(self.xarm_2, self.arm_eef_index)[0])
        new_pos_1 = cur_pos_1 + np.array(action[0:3]) * self.max_vel * self.dt
        new_pos_1 = np.clip(new_pos_1, self.pos_space_1.low, self.pos_space_1.high)
        new_pos_2 = cur_pos_2 + np.array(action[4:7]) * self.max_vel * self.dt
        new_pos_2 = np.clip(new_pos_2, self.pos_space_2.low, self.pos_space_2.high)
        cur_gripper_pos_1 = p.getJointState(self.xarm_1, self.finger1_index)[0]
        new_gripper_pos_1 = np.clip(cur_gripper_pos_1 + action[3]*self.dt * self.max_gripper_vel, self.gripper_space.low, self.gripper_space.high)
        cur_gripper_pos_2 = p.getJointState(self.xarm_2, self.finger1_index)[0]
        new_gripper_pos_2 = np.clip(cur_gripper_pos_2 + action[7]*self.dt * self.max_gripper_vel, self.gripper_space.low, self.gripper_space.high)
        jointPoses_1 = p.calculateInverseKinematics(self.xarm_1, self.arm_eef_index, new_pos_1, [1,0,0,0], maxNumIterations = self.n_substeps)
        jointPoses_2 = p.calculateInverseKinematics(self.xarm_2, self.arm_eef_index, new_pos_2, [1,0,0,0], maxNumIterations = self.n_substeps)
        for i in range(1, self.arm_eef_index):
            p.setJointMotorControl2(self.xarm_1, i, p.POSITION_CONTROL, jointPoses_1[i-1]) # max=1200
            p.setJointMotorControl2(self.xarm_2, i, p.POSITION_CONTROL, jointPoses_2[i-1]) # max=1200
        p.setJointMotorControl2(self.xarm_1, self.finger1_index, p.POSITION_CONTROL, new_gripper_pos_1)
        p.setJointMotorControl2(self.xarm_1, self.finger2_index, p.POSITION_CONTROL, new_gripper_pos_1)
        p.setJointMotorControl2(self.xarm_2, self.finger1_index, p.POSITION_CONTROL, new_gripper_pos_2)
        p.setJointMotorControl2(self.xarm_2, self.finger2_index, p.POSITION_CONTROL, new_gripper_pos_2)

    def _get_obs(self):
        # robot state
        robot_state_1 = p.getJointStates(self.xarm_1, np.arange(0,self.num_joints))
        robot_state_2 = p.getJointStates(self.xarm_2, np.arange(0,self.num_joints))
        # gripper state
        gripper_pos_1 = np.array([robot_state_1[self.finger1_index][0]])
        gripper_vel_1 = np.array([robot_state_1[self.finger1_index][1]])
        gripper_pos_2 = np.array([robot_state_2[self.finger1_index][0]])
        gripper_vel_2 = np.array([robot_state_2[self.finger1_index][1]])
        grip_state_1 = p.getLinkState(self.xarm_1, self.gripper_base_index, computeLinkVelocity=1)
        grip_state_2 = p.getLinkState(self.xarm_2, self.gripper_base_index, computeLinkVelocity=1)
        grip_pos_1 = np.array(grip_state_1[0])-self.eef2grip_offset
        grip_pos_2 = np.array(grip_state_2[0])-self.eef2grip_offset
        grip_velp_1 = np.array(grip_state_1[6])
        grip_velp_2 = np.array(grip_state_2[6])
        # object state
        obj_pos = np.array(p.getBasePositionAndOrientation(self.legos[0])[0])
        obj_rot = np.array(p.getBasePositionAndOrientation(self.legos[0])[1])
        obj_velp = np.array(p.getBaseVelocity(self.legos[0])[0])
        obj_velr = np.array(p.getBaseVelocity(self.legos[0])[1])
        for i in range(1, self.num_obj):
            obj_pos = np.concatenate((obj_pos, p.getBasePositionAndOrientation(self.legos[i])[0]))
            obj_rot = np.concatenate((obj_rot, p.getBasePositionAndOrientation(self.legos[i])[1]))
            obj_velp = np.concatenate((obj_velp, p.getBaseVelocity(self.legos[i])[0]))
            obj_velr = np.concatenate((obj_velr, p.getBaseVelocity(self.legos[i])[1]))
        # final obs
        obs = np.concatenate((
            obj_pos, obj_rot, obj_velp, obj_velr,
            grip_pos_1, grip_velp_1, gripper_pos_1, gripper_vel_1,
            grip_pos_2, grip_velp_2, gripper_pos_2, gripper_vel_2
        ))
        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(obj_pos.copy()),
            'desired_goal': self.goal.copy()
        }

    def _reset_sim(self):
        # reset arm
        for i in range(self.num_joints):
            p.resetJointState(self.xarm_1, i, self.joint_init_pos[i])
            p.resetJointState(self.xarm_2, i, self.joint_init_pos[i])
        # randomize position of lego
        for i in range(self.num_obj):
            lego_pos = np.concatenate((self.obj_space.sample(), [self.height_offset]))
            p.resetBasePositionAndOrientation(self.legos[i], lego_pos, self.startOrientation_1)
        p.stepSimulation()
        return True

    def _sample_goal(self):
        goal = [None]*self.num_obj
        for i in range(self.num_obj):
            goal[i] = self.goal_space.sample()
            p.resetBasePositionAndOrientation(self.spheres[i], goal[i], self.startOrientation_1)
        return np.array(goal).flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = np.linalg.norm(achieved_goal - self.goal, axis=-1)
        return (d < self.distance_threshold).astype(np.float32)

    def _run_demo(self):
        p.resetBasePositionAndOrientation(self.legos[0], [-0.3,0,0.025], self.startOrientation_1)
        # move-1
        for _ in range(200):
            jointPoses_1 = p.calculateInverseKinematics(self.xarm_1, self.arm_eef_index, [-0.36, 0, 0.125], [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm_1, i, p.POSITION_CONTROL, jointPoses_1[i-1]) # max=1200
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            print('[Move-1]',self.compute_reward(obs['achieved_goal'], self.goal, info)*2.25)
            time.sleep(0.02)
        for _ in range(20):
            p.setJointMotorControl2(self.xarm_1, self.finger1_index, p.POSITION_CONTROL, 0.023)
            p.setJointMotorControl2(self.xarm_1, self.finger2_index, p.POSITION_CONTROL, 0.023)
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            print('[Grasp-1]',self.compute_reward(obs['achieved_goal'], self.goal, info)*2.25)
            time.sleep(0.02)
        # move both
        for _ in range(200):
            jointPoses_1 = p.calculateInverseKinematics(self.xarm_1, self.arm_eef_index, [-0.05, 0, 0.4], [1,0,0,0], maxNumIterations = self.n_substeps)
            jointPoses_2 = p.calculateInverseKinematics(self.xarm_2, self.arm_eef_index, [0.05, 0, 0.4], [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm_1, i, p.POSITION_CONTROL, jointPoses_1[i-1]) # max=1200
                p.setJointMotorControl2(self.xarm_2, i, p.POSITION_CONTROL, jointPoses_2[i-1]) # max=1200
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            print('[Move Both]',self.compute_reward(obs['achieved_goal'], self.goal, info)*2.25)
            time.sleep(0.02)
        # grasp
        for _ in range(20):
            p.setJointMotorControl2(self.xarm_2, self.finger1_index, p.POSITION_CONTROL, 0.024)
            p.setJointMotorControl2(self.xarm_2, self.finger2_index, p.POSITION_CONTROL, 0.024)
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            print('[Grasp-2]',self.compute_reward(obs['achieved_goal'], self.goal, info)*2.25)
            time.sleep(0.02)
        # realse
        for _ in range(20):
            jointPoses_1 = p.calculateInverseKinematics(self.xarm_1, self.arm_eef_index, [-0.01, 0, 0.5], [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm_1, i, p.POSITION_CONTROL, jointPoses_1[i-1]) # max=1200
            p.setJointMotorControl2(self.xarm_1, self.finger1_index, p.POSITION_CONTROL, 0.04)
            p.setJointMotorControl2(self.xarm_1, self.finger2_index, p.POSITION_CONTROL, 0.04)
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            print('[Release]',self.compute_reward(obs['achieved_goal'], self.goal, info)*2.25)
            # p.addUserDebugText(str(reward),[0,0,0])
            time.sleep(0.02)
        # move-2
        for _ in range(100):
            jointPoses_2 = p.calculateInverseKinematics(self.xarm_2, self.arm_eef_index, [0.4, 0.2, 0.3] , [1,0,0,0], maxNumIterations = self.n_substeps)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm_2, i, p.POSITION_CONTROL, jointPoses_2[i-1]) # max=1200
            p.stepSimulation()
            obs = self._get_obs()
            info = {'is_success': self._is_success(obs['achieved_goal'], self.goal),}
            print('[move-2]',self.compute_reward(obs['achieved_goal'], self.goal, info)*2.25)
            # p.addUserDebugText(str(reward),[0,0,0])
            time.sleep(0.02)
    
if __name__ == '__main__':
    env = XarmPDHandoverDense()
    env._run_demo()