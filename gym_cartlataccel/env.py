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

  def __init__(self, render_mode: str = None, noise_mode: str = None, moving_target: bool = True, env_bs: int = 1, eq = 12):
    print("eq", eq)
    self.tau = 0.02  # Time step
    self.max_x_frame = 2.2 # size of render frame

    _, _, self.f, self.ranges = get_feynman_dataset(eq)
    self.action_dim = len(self.ranges)

    self.bs = env_bs

    self.low = [x[0] for x in self.ranges]
    self.high = [x[1] for x in self.ranges]

    self.min_x, self.max_x = self.find_minmax(self.bs)
    # self.min_x = np.clip(self.min_x, 0, None)
    # self.max_x = np.clip(self.max_x, None, 2)
    self.min_x = np.clip(self.min_x, -2, 2)
    self.max_x = np.clip(self.max_x, -2, 2)
    # print(self.min_x.shape, self.max_x.shape)

    self.obs_dim = len(self.min_x) + self.action_dim

    # Action space is theta
    action_low = np.stack([np.array(self.low) for _ in range(self.bs)])
    action_high = np.stack([np.array(self.high) for _ in range(self.bs)])
    self.action_space = spaces.Box(
      low=action_low, high=action_high, shape=(self.bs, self.action_dim), dtype=np.float32
    )

    # Obs space is [theta_prev, target]
    self.obs_low = np.stack([np.array(self.low + list(self.min_x)) for _ in range(self.bs)])
    self.obs_high = np.stack([np.array(self.high + list(self.max_x)) for _ in range(self.bs)])

    self.observation_space = spaces.Box(
      low=self.obs_low,
      high=self.obs_high,
      shape=(self.bs, self.obs_dim),
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
    # print("action dim", self.action_dim)
    samples = np.zeros((num_samples, self.action_dim))
    for i in range(self.action_dim):
      samples[:, i] = np.random.uniform(low=self.low[i], high=self.high[i], size=num_samples)
    
    out = self.f(torch.tensor(samples)).cpu().detach().numpy()
    # print(out.shape)
    return np.min(out, axis=1), np.max(out, axis=1)

  def generate_traj(self, n_traj=1, n_points=10, n_outputs=2):
    # generates smooth curve using cubic interpolation
    t_control = np.linspace(0, self.max_episode_steps - 1, n_points)

    control_points = self.np_random.uniform(-2, 2, (n_traj, n_points, n_outputs))

    # Initialize the trajectory array
    traj = np.zeros((n_traj, self.max_episode_steps, n_outputs))
    
    # Interpolate for each output dimension
    for output_idx in range(n_outputs):
        f = interp1d(t_control, control_points[:, :, output_idx], kind='cubic', axis=1)
        t = np.arange(self.max_episode_steps)
        traj[:, :, output_idx] = f(t)

    row_min = traj.min(axis=1, keepdims=True)
    row_max = traj.max(axis=1, keepdims=True)

    # print(row_min.shape, row_max.shape, traj.shape, self.min_x.shape, self.max_x.shape)
    scaled_traj = self.min_x + (traj - row_min) * (self.max_x - self.min_x) / (row_max - row_min)

    return scaled_traj

  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.state = self.np_random.uniform(
      low=self.obs_low,
      high=self.obs_high,
      size=(self.bs, self.obs_dim)
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
    theta_prev = np.transpose(self.state[:,:-2])
    target = self.state[:,-2:]

    scaled_action = action * (np.array(self.high) - np.array(self.low)) + np.array(self.low)

    theta = scaled_action
    # noisy_theta = self.noise_model.add_lat_noise(self.curr_step, scaled_action)

    x = self.f(torch.tensor(theta)).detach().cpu().numpy()
    x, theta = np.transpose(x), np.transpose(theta)

    new_target = self.x_targets[:, self.curr_step]
    noisy_target = self.noise_model.add_lat_noise(self.curr_step, new_target)

    self.state = np.stack(np.concatenate((theta, np.transpose(new_target)), axis=0), axis=1)
    self.obs = np.stack(np.concatenate((theta, np.transpose(noisy_target)), axis=0), axis=1)

    alpha = 0.1

    step_weight = 1 - (self.curr_step / self.max_episode_steps)
    dist = abs(x - target)
    jerk = abs(theta - theta_prev)
    # jerk = np.clip(jerk - 0.05, 0, None)

    error = np.sum(np.transpose(dist) + alpha * jerk, axis=0)
    reward = -error * step_weight / np.sum(self.max_x - self.min_x)

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

    # Only render the first episode in the batch
    theta = self.state[0, :-2].reshape(1, -1)
    print(theta)
    print(theta.shape)
    cart_x = self.f(torch.tensor(theta)).detach().cpu().numpy()[1][0] # second to last one is index
    print(cart_x)

    first_cart_x = int((cart_x / self.max_x_frame) * 300 + 300)  # center is 300
    first_target_x = int((self.x_targets[0, self.curr_step, 1] / self.max_x_frame) * 300 + 300)

    ma_sines = "i kan't stay awake"

    pygame.draw.rect(self.surf, (0, 0, 0), pygame.Rect(first_cart_x - 10, 180, 20, 40))  # cart
    pygame.draw.circle(self.surf, (255, 0, 0), (first_target_x, 200), 5)  # target
    pygame.draw.line(self.surf, (0, 0, 0), (0, 220), (600, 220))  # line

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