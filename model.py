import torch
from torch import nn
torch.manual_seed(42)

def mlp(hidden_sizes, activation=nn.Tanh, output_activation=nn.Identity):
  layers = []
  for j in range(len(hidden_sizes)-1):
    act = activation if j < len(hidden_sizes)-2 else output_activation
    layers += [nn.Linear(hidden_sizes[j], hidden_sizes[j+1]), act()]
  return nn.Sequential(*layers)

class MLPGaussian(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh, log_std=3., seed=42):
    super(MLPGaussian, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))
    self.register_buffer('std', self.log_std.exp())

  def forward(self, x: torch.Tensor):
    x = x.unsqueeze(0)
    print(x.shape)
    return self.mlp(x)

  def get_action(self, obs: torch.Tensor, deterministic=False):
    mean = self.forward(obs)
    action = mean[0] if deterministic else torch.normal(mean, self.std)[0]
    return action.detach().cpu().numpy()

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
    mean = self.forward(obs)
    logprob = -0.5 * (((act - mean)**2) / self.std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
    return logprob.sum(dim=-1)

class MLPCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh, seed=42):
    super(MLPCritic, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

  def forward(self, x: torch.Tensor):
    return self.mlp(x)

class ActorCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, discrete=False):
    super(ActorCritic, self).__init__()
    self.discrete = discrete
    self.actor = MLPGaussian(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = MLPCritic(obs_dim, hidden_sizes["vf"])

  def forward(self, x: torch.Tensor):
    actor_out, _ = self.actor(x) # mean
    critic_out = self.critic(x)
    return actor_out, critic_out

#----------------------------------------------------------------------------#

from efficient_kan import KAN

class KANGaussian(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, grid_size=5, spline_order=3, log_std=3., seed=42):
    super(KANGaussian, self).__init__()
    layers = [obs_dim] + list(hidden_sizes) + [act_dim]
    self.kan = KAN(
      layers, 
      grid_size=grid_size, 
      spline_order=spline_order,
      scale_noise=0.01,
      scale_base=1, 
      scale_spline=1,
      base_activation=torch.nn.SiLU,
      grid_eps=0.02,
      grid_range=[-1, 1]
    )
    self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))
    self.register_buffer('std', self.log_std.exp())

  def forward(self, x: torch.Tensor):
    x = x.unsqueeze(0)
    return self.kan(x)

  def get_action(self, obs: torch.Tensor, deterministic=False):
    mean = self.forward(obs)
    action = mean[0] if deterministic else torch.normal(mean, self.std)[0]
    return action.detach().cpu().numpy()

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
    mean = self.forward(obs)
    logprob = -0.5 * (((act - mean)**2) / self.std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
    return logprob.sum(dim=-1)

# class KANCritic(nn.Module):
#   def __init__(self, obs_dim, hidden_sizes, grid_size=5, spline_order=3, seed=42):
#     super(KANCritic, self).__init__()
#     layers = [obs_dim] + list(hidden_sizes) + [1]
#     self.kan = KAN(layers, grid_size=grid_size, spline_order=spline_order)

#   def forward(self, x: torch.Tensor):
#     return self.kan(x)

class KANActorCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim):
    super(KANActorCritic, self).__init__()
    self.actor = KANGaussian(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = MLPCritic(obs_dim, hidden_sizes["vf"])
    # self.critic = KANCritic(obs_dim, hidden_sizes["vf"], grid_size, spline_order)

  def forward(self, x: torch.Tensor):
    actor_out = self.actor(x)
    critic_out = self.critic(x)
    return actor_out, critic_out
