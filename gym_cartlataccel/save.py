import pygame
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import interp1d
from gym_cartlataccel.noise import SimNoise
from gym_cartlataccel.feynman import get_feynman_dataset

class BatchedCartLatAccelEnv(gym.Env):
  """
  Batched CartLatAccel env

  Action space: ndarray shape (bs,) representing accel applied to cart
  Observation space: ndarray shape (bs, 3) with cart state and target, [pos, velocity, target_pos]
  Rewards: r = -error/500, where error is abs x-x_target. Scaled to (-10,0)

  Starting state: random state in obs space
  Episode truncation: 500 timesteps
  """

  metadata = {
    "render_modes": ["human", "rgb_array"],
    "render_fps": 50,
  }

  def __init__(self, render_mode: str = None, noise_mode: str = None, moving_target: bool = True, env_bs: int = 1, eq=-2):
    self.tau = 0.02  # Time step
    self.max_x_frame = 0.25 # size of render frame
    self.max_y_frame = 0.25

    _, _, self.fx, self.ranges_x = get_feynman_dataset(-2)
    _, _, self.fy, self.ranges_y = get_feynman_dataset(-3)
    self.action_dim = len(self.ranges_x)

    self.bs = env_bs

    self.low = [x[0] for x in self.ranges_x]
    self.high = [x[1] for x in self.ranges_x]

    self.min_x, self.max_x = self.find_minmax()
    self.min_x = max(-2, self.min_x)
    self.max_x = min(2, self.max_x)

    # Action space is theta
    action_low = np.stack([np.array(self.low) for _ in range(self.bs)])
    action_high = np.stack([np.array(self.high) for _ in range(self.bs)])
    self.action_space = spaces.Box(
      low=action_low, high=action_high, shape=(self.bs, self.action_dim), dtype=np.float32
    )

    # Obs space is [theta_prev, target]
    self.obs_low = np.stack([np.array(self.low + [self.min_x]) for _ in range(self.bs)])
    self.obs_high = np.stack([np.array(self.high + [self.max_x]) for _ in range(self.bs)])

    self.observation_space = spaces.Box(
      low=self.obs_low,
      high=self.obs_high,
      shape=(self.bs, self.action_dim+1),
      dtype=np.float32
    )

    self.render_mode = render_mode
    self.screen = None
    self.clock = None

    self.max_episode_steps = 500
    self.curr_step = 0
    self.noise_mode = noise_mode
    self.moving_target = moving_target

  def find_minmax(self, num_samples = 10000):
    samples = np.zeros((num_samples, self.action_dim))
    for i in range(self.action_dim):
      samples[:, i] = np.random.uniform(low=self.low[i], high=self.high[i], size=num_samples)
    
    out = self.fx(torch.tensor(samples))
    return min(out).item(), max(out).item()

  # def generate_traj(self, n_traj=1, n_points=10, n_outputs=2, initial_points=np.array([])):
  #   # generates smooth curve using cubic interpolation
  #   t_control = np.linspace(0, self.max_episode_steps - 1, n_points)

  #   control_points = self.np_random.uniform(-2, 2, (n_traj, n_points, n_outputs))

  #   # Initialize the trajectory array
  #   traj = np.zeros((n_traj, self.max_episode_steps, n_outputs))
    
  #   # Interpolate for each output dimension
  #   for output_idx in range(n_outputs):
  #       f = interp1d(t_control, control_points[:, :, output_idx], kind='cubic', axis=1)
  #       t = np.arange(self.max_episode_steps)
  #       traj[:, :, output_idx] = f(t)

  #   row_min = traj.min(axis=1, keepdims=True)
  #   row_max = traj.max(axis=1, keepdims=True)

  #   # print(row_min.shape, row_max.shape, traj.shape, self.min_x.shape, self.max_x.shape)
  #   scaled_traj = self.min_x + (traj - row_min) * (self.max_x - self.min_x) / (row_max - row_min)

  #   if initial_points.size != 0:
  #     # cheating more!
  #     num_points = 100
  #     first_points = scaled_traj[:, 0, :]
  #     # interpolated_points = np.zeros((n_traj, 25, n_outputs))
  #     weights = np.linspace(0, 1, num_points).reshape(1, num_points, 1)
  #     interpolated_points = (1 - weights) * initial_points[:, np.newaxis, :] + weights * first_points[:, np.newaxis, :]

  #     # scaled_traj = np.concatenate((interpolated_points, scaled_traj), axis=1)

  #   # print("==")
  #   # print(scaled_traj.shape)

  #   return scaled_traj

  def generate_traj(self, n_traj=1, n_points=10):
    # generates smooth curve using cubic interpolation
    t_control = np.linspace(0, self.max_episode_steps - 1, n_points)
    control_points = self.np_random.uniform(-2, 2, (n_traj, n_points)) # slightly less than max x
    f = interp1d(t_control, control_points, kind='cubic')
    t = np.arange(self.max_episode_steps)

    traj = f(t)

    row_min = traj.min(axis=1, keepdims=True)
    row_max = traj.max(axis=1, keepdims=True)

    scaled_traj = self.min_x + (traj - row_min) * (self.max_x - self.min_x) / (row_max - row_min)

    return scaled_traj

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.state = self.np_random.uniform(
      low=self.obs_low,
      high=self.obs_high,
      size=(self.bs, self.action_dim+1)
    )
    self.obs = self.state

    if self.moving_target:
      self.x_targets = self.generate_traj(self.bs)
    else:
      self.x_targets = np.full((self.bs, self.max_episode_steps), self.state[-1]) # fixed target
    self.noise_model = SimNoise(self.max_episode_steps, 1/self.tau, self.noise_mode, seed=seed)

    self.curr_step = 0
    if self.render_mode == "human":
      self.render()
    return np.array(self.state, dtype=np.float32), {}

  def step(self, action):
    theta_prev = np.transpose(self.state[:,:-1])
    target = self.state[:,-1]

    scaled_action = action * (np.array(self.high) - np.array(self.low)) + np.array(self.low)

    theta = scaled_action
    # noisy_theta = self.noise_model.add_lat_noise(self.curr_step, scaled_action)

    x = self.fx(torch.tensor(theta)).detach().cpu().numpy()
    x, theta = np.transpose(x), np.transpose(theta)

    new_target = self.x_targets[:, self.curr_step]
    noisy_target = self.noise_model.add_lat_noise(self.curr_step, new_target)

    self.state = np.stack(np.concatenate((theta, new_target.reshape(1, -1)), axis=0), axis=1)
    self.obs = np.stack(np.concatenate((theta, noisy_target.reshape(1, -1)), axis=0), axis=1)

    alpha = 0.1

    step_weight = 1 - (self.curr_step / self.max_episode_steps)
    dist = abs(x - target)
    jerk = abs(theta - theta_prev)
    # jerk = np.clip(jerk - 0.05, 0, None)

    error = np.sum(dist + alpha * jerk, axis=0)
    reward = -error * step_weight / (self.max_x - self.min_x)

    if self.render_mode == "human":
      self.render()

    self.curr_step += 1
    truncated = self.curr_step >= self.max_episode_steps
    info = {"action": action, "noisy_action": theta, "x": x, "x_target": new_target}
    return np.array(self.obs, dtype=np.float32), reward, False, truncated, info

  def render(self):
    if self.screen is None:
        pygame.init()
        if self.render_mode == "human":
            pygame.display.init()
            self.screen = pygame.display.set_mode((600, 400))
        else:  # rgb_array
            self.screen = pygame.Surface((600, 400))
    if self.clock is None:
        self.clock = pygame.time.Clock()

    self.surf = pygame.Surface((600, 400))
    self.surf.fill((255, 255, 255))

    # Extract coordinates for 2D visualization
    theta = self.state[0, :-1].reshape(1, -1)
    print(theta.shape)
    print(f"Theta: {theta}")
    # Get x and y positions for the cart
    cart_x = self.fx(torch.tensor(theta)).detach().cpu().numpy()[0]
    cart_y = self.fy(torch.tensor(theta)).detach().cpu().numpy()[0]
    print(f"x: {cart_x}, y: {cart_y}")

    # Get x and y positions for the target
    target_x = self.x_targets[0, self.curr_step]  # coord=0
    target_y = self.curr_step / self.max_episode_steps / 4  # coord=1

    # Scale positions to fit within the display
    first_cart_x = int((cart_x / self.max_x_frame) * 300 + 300)  # Center is 300
    first_cart_y = int((cart_y / self.max_y_frame) * 200 + 200)  # Center is 200
    first_target_x = int((target_x / self.max_x_frame) * 300 + 300)
    first_target_y = int((target_y / self.max_y_frame) * 200 + 200)

    # Draw the cart as a rectangle and the target as a circle
    pygame.draw.rect(
        self.surf,
        (0, 0, 0),
        pygame.Rect(first_cart_x - 10, first_cart_y - 20, 20, 40)  # Cart dimensions
    )
    pygame.draw.circle(
        self.surf,
        (255, 0, 0),
        (first_target_x, first_target_y),
        5  # Target radius
    )

    # Display the surface
    self.screen.blit(self.surf, (0, 0))
    if self.render_mode == "human":
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
    elif self.render_mode == "rgb_array":
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )

  def close(self):
    if self.screen is not None:
      import pygame
      pygame.display.quit()
      pygame.quit()

# if __name__ == "__main__":
#   from stable_baselines3.common.env_checker import check_env
#   env = CartLatAccelEnv()
#   check_env(env)
#   print(env.observation_space, env.action_space)