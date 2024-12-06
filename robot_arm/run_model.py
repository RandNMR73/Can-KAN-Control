from env import RobotArmEnvironment
import torch

def main(env, model):
  """
  Use model as a controller to get to a desired end-effector location.
  
  Args:
    env: RobotArmEnvironment with move_tool() method.
    model: Input [theta_1_prev, theta_2_prev, x_target, y_target]. Outputs [theta_1, theta_2]
  """
  # get initial state theta1, theta2
  current_joint_state = env.get_joints()
  theta1_prev, theta2_prev = current_joint_state[1], current_joint_state[2]
  print(theta1_prev, theta2_prev)

  # get desired end effector positoin
  x_target, y_target = 0.0, 0.1
  
  # run control loop
  while True:
    # 1. pass in input to model
    state = torch.FloatTensor([theta1_prev, theta2_prev, x_target, y_target])
    print(model(state))
    theta1, theta2 = model(state)[0]

    # 2. move joints to new state
    # get current joint state
    target_joint_state = current_joint_state.copy()
    target_joint_state[1], target_joint_state[2] = theta1, theta2
    env.move_joints(target_joint_state=target_joint_state, acceleration=10, speed=3.0)
    theta1_prev, theta2_prev = theta1, theta2

    # 3. print new x_target, y_target
    x_target, y_target, _ = env.get_end_effector_position()
    print(x_target, y_target)

    if abs(x_target - 0.5) < 0.01 and abs(y_target - 0.5) < 0.01:
      break

    env.step_simulation(1)
  

if __name__ == "__main__":
  env = RobotArmEnvironment(gui=True)
  model = torch.jit.load("best_jit.pt")
  main(env, model)

  while True:
    env.step_simulation(1)
