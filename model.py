import torch
from torch import nn
import torch.nn.functional as F
torch.manual_seed(42)

def mlp(hidden_sizes, activation=nn.Tanh, output_activation=nn.Identity):
  layers = []
  for j in range(len(hidden_sizes)-1):
    act = activation if j < len(hidden_sizes)-2 else output_activation
    layers += [nn.Linear(hidden_sizes[j], hidden_sizes[j+1]), act()]
  return nn.Sequential(*layers)

# class MLPGaussian(nn.Module):
#   def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh, log_std=3., seed=42):
#     super(MLPGaussian, self).__init__()
#     self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
#     self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))
#     self.register_buffer('std', self.log_std.exp())

#   def forward(self, x: torch.Tensor):
#     x = x.unsqueeze(0)
#     return self.mlp(x)

#   def get_action(self, obs: torch.Tensor, deterministic=False):
#     mean = self.forward(obs)
#     action = mean[0] if deterministic else torch.normal(mean, self.std)[0]
#     return action.detach().cpu().numpy()

#   def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
#     mean = self.forward(obs)
#     logprob = -0.5 * (((act - mean)**2) / self.std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
#     return logprob.sum(dim=-1)

class MLPGaussian(nn.Module):
  def __init__(self, obs_dim: int, hidden_sizes: list[int], act_dim: int, activation: nn.Module = nn.Tanh, log_std: float = 0.) -> None:
    super(MLPGaussian, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
    self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.mlp(x)
  
  def get_policy(self, obs: torch.Tensor) -> torch.Tensor:
    mean = self.forward(obs)
    std = self.log_std.exp()
    return mean, std

  def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
    mean, std = self.get_policy(obs)
    action = mean if deterministic else torch.normal(mean, std)
    return action.detach().cpu().numpy() #.squeeze(-1)

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
    mean, std = self.get_policy(obs)
    logprob = -0.5 * (((act - mean)**2) / std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
    entropy = (torch.log(std) + 0.5 * (1 + torch.log(torch.tensor(2*torch.pi))))
    assert logprob.shape == act.shape
    print('logprob', logprob.shape, 'entropy', entropy.shape)
    return logprob.sum(dim=-1), entropy.sum(dim=-1)

class MLPBeta(nn.Module):
  '''Beta distribution for bounded continuous control, output between 0 and 1'''
  def __init__(self, obs_dim, hidden_sizes, act_dim, activation=nn.Tanh, bias=True, act_bound: tuple[float, float] = (0, 1)):
    super(MLPBeta, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [act_dim*2], activation)
    self.act_dim = act_dim
    self.act_bound = act_bound
    
  def forward(self, x: torch.Tensor):
    return self.mlp(x)
  
  def get_policy(self, obs: torch.Tensor):
    alpha_beta = self.forward(obs)
    alpha, beta = torch.split(alpha_beta, self.act_dim, dim=-1)
    alpha = F.softplus(alpha) + 1
    beta = F.softplus(beta) + 1
    return alpha, beta

  def get_action(self, obs: torch.Tensor, deterministic=False):
    alpha, beta = self.get_policy(obs)
    action = alpha / (alpha + beta) if deterministic else torch.distributions.Beta(alpha, beta).sample()
    action = action.detach()
    # scaled_action = action * (self.act_bound[1] - self.act_bound[0]) + self.act_bound[0] # scale to act_bound
    return action.detach().cpu().numpy() #.squeeze(-1)

  def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
    # act = (act - self.act_bound[0]) / (self.act_bound[1] - self.act_bound[0]) # scale back to (0,1)
    alpha, beta = self.get_policy(obs)
    dist = torch.distributions.Beta(alpha, beta)
    logprob = dist.log_prob(act)
    entropy = dist.entropy()
    return logprob.sum(dim=-1), entropy.sum(dim=-1)

class MLPCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, activation=nn.Tanh, seed=42):
    super(MLPCritic, self).__init__()
    self.mlp = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

  def forward(self, x: torch.Tensor):
    return self.mlp(x)

# class ActorCritic(nn.Module):
#   def __init__(self, obs_dim, hidden_sizes, act_dim, discrete=False):
#     super(ActorCritic, self).__init__()
#     self.discrete = discrete
#     self.actor = MLPGaussian(obs_dim, hidden_sizes["pi"], act_dim)
#     self.critic = MLPCritic(obs_dim, hidden_sizes["vf"])

#   def forward(self, x: torch.Tensor):
#     actor_out, _ = self.actor(x) # mean
#     critic_out = self.critic(x)
#     return actor_out, critic_out

class ActorCritic(nn.Module):
  def __init__(self, obs_dim: int, hidden_sizes: dict[str, list[int]], act_dim: int, discrete: bool = False, shared_layers: bool = True, act_bound: tuple[float, float] = None) -> None:
    super(ActorCritic, self).__init__()
    model_class = MLPGaussian if not act_bound else MLPBeta
      
    if model_class == MLPBeta: # if bounded then use MLPBeta
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim, act_bound=act_bound)
      act_dim *= 2 # MLPBeta outputs two parameters alpha, beta
    else:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = MLPCritic(obs_dim, hidden_sizes["vf"])

    if shared_layers and len(hidden_sizes["pi"]) > 1:
      self.shared = mlp([obs_dim] + hidden_sizes["pi"][:-1], nn.Tanh)
      self.actor.mlp = nn.Sequential( # override
        self.shared,
        mlp([hidden_sizes["pi"][-2], hidden_sizes["pi"][-1], act_dim], nn.Tanh)
      )
      self.critic.mlp = nn.Sequential(
        self.shared,
        mlp([hidden_sizes["vf"][-2], hidden_sizes["vf"][-1], 1], nn.Tanh)
      )

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    actor_out = self.actor(x)
    critic_out = self.critic(x)
    return actor_out, critic_out

#----------------------------------------------------------------------------#

from efficient_kan import KAN

class KANGaussian(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, act_dim, grid_size=5, spline_order=3, log_std=0.):
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

    def forward(self, x: torch.Tensor):
        return self.kan(x)

    def get_policy(self, obs: torch.Tensor):
        mean = self.forward(obs)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic=False):
        mean, std = self.get_policy(obs)
        action = mean if deterministic else torch.normal(mean, std)
        return action.detach().cpu().numpy()

    def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
        mean, std = self.get_policy(obs)
        logprob = -0.5 * (((act - mean)**2) / std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
        entropy = (torch.log(std) + 0.5 * (1 + torch.log(torch.tensor(2*torch.pi))))
        return logprob.sum(dim=-1), entropy.sum(dim=-1)

class KANCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, grid_size=5, spline_order=3, seed=42):
    super(KANCritic, self).__init__()
    layers = [obs_dim] + list(hidden_sizes) + [1]
    self.kan = KAN(layers, grid_size=grid_size, spline_order=spline_order)

  def forward(self, x: torch.Tensor):
    return self.kan(x)

class KANBeta(nn.Module):
    '''Beta distribution for bounded continuous control using KAN architecture, output between 0 and 1'''
    def __init__(self, obs_dim, hidden_sizes, act_dim, grid_size=5, spline_order=3, act_bound=(0, 1)):
        super(KANBeta, self).__init__()
        layers = [obs_dim] + list(hidden_sizes) + [act_dim * 2]  # *2 for alpha and beta parameters
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
        self.act_dim = act_dim
        self.act_bound = act_bound

    def forward(self, x: torch.Tensor):
        return self.kan(x)

    def get_policy(self, obs: torch.Tensor):
        alpha_beta = self.forward(obs)
        alpha, beta = torch.split(alpha_beta, self.act_dim, dim=-1)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta) + 1
        return alpha, beta

    def get_action(self, obs: torch.Tensor, deterministic=False):
        alpha, beta = self.get_policy(obs)
        action = alpha / (alpha + beta) if deterministic else torch.distributions.Beta(alpha, beta).sample()
        return action.detach().cpu().numpy()

    def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
        alpha, beta = self.get_policy(obs)
        dist = torch.distributions.Beta(alpha, beta)
        logprob = dist.log_prob(act)
        entropy = dist.entropy()
        return logprob.sum(dim=-1), entropy.sum(dim=-1)

class KANActorCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, act_bound=None):
    super(KANActorCritic, self).__init__()
    model_class = KANGaussian if not act_bound else KANBeta
    
    if model_class == KANBeta:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim, act_bound=act_bound)
    else:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = KANCritic(obs_dim, hidden_sizes["vf"])

  def forward(self, x: torch.Tensor):
    actor_out = self.actor(x)
    critic_out = self.critic(x)
    return actor_out, critic_out

#----------------------------------------------------------------------------#

from fftKAN import FourierKAN

class FourierKANGaussian(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, act_dim, grid_size=5, log_std=0.):
        super(FourierKANGaussian, self).__init__()
        layers = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.kan = FourierKAN(
        layers,
        grid_size=grid_size,
    )
        self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return self.kan(x)

    def get_policy(self, obs: torch.Tensor):
        mean = self.forward(obs)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic=False):
        mean, std = self.get_policy(obs)
        action = mean if deterministic else torch.normal(mean, std)
        return action.detach().cpu().numpy()

    def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
        mean, std = self.get_policy(obs)
        logprob = -0.5 * (((act - mean)**2) / std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
        entropy = (torch.log(std) + 0.5 * (1 + torch.log(torch.tensor(2*torch.pi))))
        return logprob.sum(dim=-1), entropy.sum(dim=-1)

class FourierKANCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, grid_size=5, seed=42):
    super(FourierKANCritic, self).__init__()
    layers = [obs_dim] + list(hidden_sizes) + [1]
    self.kan = FourierKAN(layers, grid_size=grid_size)

  def forward(self, x: torch.Tensor):
    return self.kan(x)

class FourierKANBeta(nn.Module):
    '''Beta distribution for bounded continuous control using KAN architecture, output between 0 and 1'''
    def __init__(self, obs_dim, hidden_sizes, act_dim, grid_size=5, act_bound=(0, 1)):
        super(FourierKANBeta, self).__init__()
        layers = [obs_dim] + list(hidden_sizes) + [act_dim * 2]  # *2 for alpha and beta parameters
        self.kan = FourierKAN(
            layers,
            grid_size=grid_size,
        )
        self.act_dim = act_dim
        self.act_bound = act_bound

    def forward(self, x: torch.Tensor):
        return self.kan(x)

    def get_policy(self, obs: torch.Tensor):
        alpha_beta = self.forward(obs)
        alpha, beta = torch.split(alpha_beta, self.act_dim, dim=-1)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta) + 1
        return alpha, beta

    def get_action(self, obs: torch.Tensor, deterministic=False):
        alpha, beta = self.get_policy(obs)
        action = alpha / (alpha + beta) if deterministic else torch.distributions.Beta(alpha, beta).sample()
        return action.detach().cpu().numpy()

    def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
        alpha, beta = self.get_policy(obs)
        dist = torch.distributions.Beta(alpha, beta)
        logprob = dist.log_prob(act)
        entropy = dist.entropy()
        return logprob.sum(dim=-1), entropy.sum(dim=-1)

class FourierKANActorCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, act_bound=None):
    super(FourierKANActorCritic, self).__init__()
    model_class = FourierKANGaussian if not act_bound else FourierKANBeta
    
    if model_class == FourierKANBeta:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim, act_bound=act_bound)
    else:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = FourierKANCritic(obs_dim, hidden_sizes["vf"])

  def forward(self, x: torch.Tensor):
    actor_out = self.actor(x)
    critic_out = self.critic(x)
    return actor_out, critic_out

#----------------------------------------------------------------------------#

from waveletKAN import WaveletKAN

class WaveletKANGaussian(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, act_dim, grid_size=5, log_std=0.):
        super(WaveletKANGaussian, self).__init__()
        layers = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.kan = WaveletKAN(
        layers,
        grid_size=grid_size,
    )
        self.log_std = torch.nn.Parameter(torch.full((act_dim,), log_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return self.kan(x)

    def get_policy(self, obs: torch.Tensor):
        mean = self.forward(obs)
        std = self.log_std.exp()
        return mean, std

    def get_action(self, obs: torch.Tensor, deterministic=False):
        mean, std = self.get_policy(obs)
        action = mean if deterministic else torch.normal(mean, std)
        return action.detach().cpu().numpy()

    def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
        mean, std = self.get_policy(obs)
        logprob = -0.5 * (((act - mean)**2) / std**2 + 2 * self.log_std + torch.log(torch.tensor(2*torch.pi)))
        entropy = (torch.log(std) + 0.5 * (1 + torch.log(torch.tensor(2*torch.pi))))
        return logprob.sum(dim=-1), entropy.sum(dim=-1)

class WaveletKANCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, grid_size=5, seed=42):
    super(WaveletKANCritic, self).__init__()
    layers = [obs_dim] + list(hidden_sizes) + [1]
    self.kan = WaveletKAN(layers, grid_size=grid_size)

  def forward(self, x: torch.Tensor):
    return self.kan(x)

class WaveletKANBeta(nn.Module):
    '''Beta distribution for bounded continuous control using KAN architecture, output between 0 and 1'''
    def __init__(self, obs_dim, hidden_sizes, act_dim, grid_size=5, act_bound=(0, 1)):
        super(WaveletKANBeta, self).__init__()
        layers = [obs_dim] + list(hidden_sizes) + [act_dim * 2]  # *2 for alpha and beta parameters
        self.kan = WaveletKAN(
            layers,
            grid_size=grid_size,
        )
        self.act_dim = act_dim
        self.act_bound = act_bound

    def forward(self, x: torch.Tensor):
        return self.kan(x)

    def get_policy(self, obs: torch.Tensor):
        alpha_beta = self.forward(obs)
        alpha, beta = torch.split(alpha_beta, self.act_dim, dim=-1)
        alpha = F.softplus(alpha) + 1
        beta = F.softplus(beta) + 1
        return alpha, beta

    def get_action(self, obs: torch.Tensor, deterministic=False):
        alpha, beta = self.get_policy(obs)
        action = alpha / (alpha + beta) if deterministic else torch.distributions.Beta(alpha, beta).sample()
        return action.detach().cpu().numpy()

    def get_logprob(self, obs: torch.Tensor, act: torch.Tensor):
        alpha, beta = self.get_policy(obs)
        dist = torch.distributions.Beta(alpha, beta)
        logprob = dist.log_prob(act)
        entropy = dist.entropy()
        return logprob.sum(dim=-1), entropy.sum(dim=-1)

class WaveletKANActorCritic(nn.Module):
  def __init__(self, obs_dim, hidden_sizes, act_dim, act_bound=None):
    super(WaveletKANActorCritic, self).__init__()
    model_class = WaveletKANGaussian if not act_bound else WaveletKANBeta
    
    if model_class == WaveletKANBeta:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim, act_bound=act_bound)
    else:
      self.actor = model_class(obs_dim, hidden_sizes["pi"], act_dim)
    self.critic = WaveletKANCritic(obs_dim, hidden_sizes["vf"])

  def forward(self, x: torch.Tensor):
    actor_out = self.actor(x)
    critic_out = self.critic(x)
    return actor_out, critic_out

