import pybullet as p
import pybullet_data
import numpy as np
import time

from utils.control import get_movej_trajectory

class RobotArmEnvironment:
  def __init__(self, gui=True, urdf="low-cost-arm.urdf"):
    # 0 load environment
    if gui:
      p.connect(p.GUI)
    else:
      p.connect(p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(1.5,45,-45,[0,0,0])
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    self._plane_id = p.loadURDF("plane.urdf")
    p.setGravity(0, 0, -9.8)

    # 1 load UR5 robot
    self.robot_body_id = p.loadURDF(urdf, [0, 0, 0], p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)

    # Get revolute joint indices of robot (skip fixed joints)
    robot_joint_info = [p.getJointInfo(self.robot_body_id, i) for i in range(p.getNumJoints(self.robot_body_id))]
    self._robot_joint_indices = [
      x[0] for x in robot_joint_info if x[2] == p.JOINT_REVOLUTE]
    print(robot_joint_info)

    # joint position threshold in radians (i.e. move until joint difference < epsilon)
    self._joint_epsilon = 1e-3

    # # Robot home joint configuration (over tote 1)
    # self.robot_home_joint_config = [
    #     -np.pi, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]
    # # Robot goal joint configuration (over tote 1)
    # self.robot_goal_joint_config = [
    #     0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0]

    # # 2 load tote
    # # 3D workspace for tote 1
    # self._workspace1_bounds = np.array([
    #     [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
    #     [-0.22, 0.22],
    #     [0.00, 0.5]
    # ])
    # # 3D workspace for tote 2
    # self._workspace2_bounds = np.copy(self._workspace1_bounds)
    # self._workspace2_bounds[0, :] = - self._workspace2_bounds[0, ::-1]        # Load totes and fix them to their position
    # # Load totes and fix them to their position
    # self._tote1_position = (
    #     self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
    # self._tote1_position[2] = 0.01
    # self._tote1_body_id = p.loadURDF(
    #     "assets/tote/toteA_large.urdf", self._tote1_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

    # self._tote2_position = (
    #     self._workspace2_bounds[:, 0] + self._workspace2_bounds[:, 1]) / 2
    # self._tote2_position[2] = 0.01
    # self._tote2_body_id = p.loadURDF(
    #     "assets/tote/toteA_large.urdf", self._tote2_position, p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)

    # 3. load gripper
    self.robot_end_effector_link_index = 4
    # self._robot_tool_offset = [0, 0, -0.05]
    # # Distance between tool tip and end-effector joint
    # self._tool_tip_to_ee_joint = np.array([0, 0, 0.15])

    # # Attach robotiq gripper to UR5 robot
    # # - We use createConstraint to add a fixed constraint between the ur5 robot and gripper.
    # self._gripper_body_id = p.loadURDF("assets/gripper/robotiq_2f_85.urdf")
    # p.resetBasePositionAndOrientation(self._gripper_body_id, [
    #                                   0.5, 0.1, 0.2], p.getQuaternionFromEuler([np.pi, 0, 0]))

    # p.createConstraint(self.robot_body_id, self.robot_end_effector_link_index, self._gripper_body_id, 0, jointType=p.JOINT_FIXED, jointAxis=[
    #                     0, 0, 0], parentFramePosition=[0, 0, 0], childFramePosition=self._robot_tool_offset, childFrameOrientation=p.getQuaternionFromEuler([0, 0, np.pi/2]))

    # # Set friction coefficients for gripper fingers
    # for i in range(p.getNumJoints(self._gripper_body_id)):
    #     p.changeDynamics(self._gripper_body_id, i, lateralFriction=1.0, spinningFriction=1.0,
    #                       rollingFriction=0.0001, frictionAnchor=True)
    
    # self.set_joints(self.robot_home_joint_config)

    # # 4. load camera
    # self.camera = camera.Camera(
    #     image_size=(128, 128),
    #     near=0.01,
    #     far=10.0,
    #     fov_w=80
    # )
    # camera_target_position = (self._workspace1_bounds[:, 0] + self._workspace1_bounds[:, 1]) / 2
    # camera_target_position[2] = 0
    # camera_distance = np.sqrt(((np.array([0.5, -0.5, 0.5]) - camera_target_position)**2).sum())
    # self.view_matrix = p.computeViewMatrixFromYawPitchRoll(
    #     cameraTargetPosition=camera_target_position,
    #     distance=camera_distance,
    #     yaw=90,
    #     pitch=-90,
    #     roll=0,
    #     upAxisIndex=2,
    # )

    # # 5. prepare loading objects
    # self.object_ids = list()
  
  def get_joints(self, indices=None):
    """
    Get array of current joint states (radians)
    """
    if indices is None:
      indices = self._robot_joint_indices
    return np.array([x[0] for x in p.getJointStates(self.robot_body_id, indices)]) # radians, velocity, force, motor torque
  
  def set_joints(self, target_joint_state, steps=1e2): # target joint state in radians
    """
    Teleports to desired joint state (radians)
    """
    assert len(self._robot_joint_indices) == len(target_joint_state)
    for joint, value in zip(self._robot_joint_indices, target_joint_state):
      p.resetJointState(self.robot_body_id, joint, value)
    if steps > 0:
      self.step_simulation(steps)
  
  def move_joints(self, target_joint_state, acceleration=10, speed=3.0):
    """
    Move robot arm to specified joint configuration by appropriate motor control
    """
    assert len(self._robot_joint_indices) == len(target_joint_state)
    
    dt = 1./240
    q_current = np.array([x[0] for x in p.getJointStates(self.robot_body_id, self._robot_joint_indices)])
    q_target = np.array(target_joint_state)
    q_traj = get_movej_trajectory(q_current, q_target, 
        acceleration=acceleration, speed=speed)
    qdot_traj = np.gradient(q_traj, dt, axis=0)
    p_gain = 1 * np.ones(len(self._robot_joint_indices))
    d_gain = 1 * np.ones(len(self._robot_joint_indices))

    for i in range(len(q_traj)):
      p.setJointMotorControlArray(
        bodyUniqueId=self.robot_body_id, 
        jointIndices=self._robot_joint_indices,
        controlMode=p.POSITION_CONTROL, 
        targetPositions=q_traj[i],
        targetVelocities=qdot_traj[i],
        positionGains=p_gain,
        velocityGains=d_gain
      )
      self.step_simulation(1)
  
  def get_end_effector_position(self):
    """
    Get current end-effector position (meters)
    """
    return p.getLinkState(self.robot_body_id, self.robot_end_effector_link_index)[0]
  
  def move_end_effector(self, position, orientation, acceleration=10, speed=3.0):
    """
    Move robot end-effector to a specified pose
      @param position: Target position of the end-effector link
      @param orientation: Target orientation of the end-effector link
    """
    target_joint_state = p.calculateInverseKinematics(self.robot_body_id, self.robot_end_effector_link_index, position, orientation)
    self.move_joints(target_joint_state, acceleration=acceleration, speed=speed)
    
  def step_simulation(self, num_steps):
    for i in range(int(num_steps)):
      p.stepSimulation()

if __name__ == "__main__":
  env = RobotArmEnvironment(gui=True)
  
  from utils.env_test import *
  # test_get_joints(env)
  test_joint_movement(env)
  test_move_end_effector(env)
  
  while True:
    env.step_simulation(1)
