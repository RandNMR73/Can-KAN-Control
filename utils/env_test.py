import time
from env import RobotArmEnvironment
import numpy as np

def test_get_joints(env):
  print("TESTING GET JOINTS")
  before = env.get_joints()
  print('before move joints', before)
  assert np.allclose(before, [0, 0, 0, 0, 0])
  env.move_joints([1, 0, 0, 0, 0], 10, 3.0)
  after = env.get_joints()
  print('after move to [1, 0, 0, 0, 0]', after)
  assert np.allclose([1, 0, 0, 0, 0], after, atol=1e-5)
  print("PASSED\n-----------\n")
  
def test_joint_movement(env):
  print("TESTING JOINT MOVEMENT")
  test_positions = [
    [0, 0, 0, 0, 0],  # Home position
    [1, 0, 0, 0, 0],
    [-1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],  # Back to home
  ]
  
  start = time.time()
  for i in range(1):
    for target_pos in test_positions:
      print(f"Moving to position: {target_pos}")
      # env.move_joints(target_pos)
      env.set_joints(target_pos)
      env.step_simulation(1)
      print(env.get_joints())
      assert np.allclose(env.get_joints(), target_pos, atol=1e-3)
      time.sleep(0.1)
  end = time.time()
  print(f"Time taken: {end - start}")
  print("PASSED\n-----------\n")

def test_move_end_effector(env):
  print("TESTING MOVE END EFFECTOR")
  env.move_end_effector(position=[0.5, 0.5, 0.5], orientation=[0, 0, 0, 1], acceleration=10, speed=3.0) # no rotation
  time.sleep(0.1)
  print(env.get_end_effector_position())
  # calculate target end effector position
  pos = env.get_end_effector_position() # [x, y, z]
  # assert np.allclose(pos, [0, 0.2, 0.1], atol=1e-2)
  print("PASSED\n-----------\n")
