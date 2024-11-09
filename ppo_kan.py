import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
# import gym_cartlataccel
from gym_cartlataccel.env import BatchedCartLatAccelEnv as CartLatAccelEnv
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict
from model import ActorCritic, KANActorCritic

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class PPO:
  def __init__(self, env, model, lr=1e-1, gamma=0.99, lam=0.95, clip_range=0.2, epochs=1, n_steps=100, ent_coeff=0.01, bs=100, env_bs=1, device='cuda', debug=False, seed=42):
    self.env = env
    self.env_bs = env_bs
    self.model = model.to(device)
    self.gamma = gamma
    self.lam = lam
    self.clip_range = clip_range
    self.epochs = epochs
    self.n_steps = n_steps
    self.ent_coeff = ent_coeff
    self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
    self.replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=10000, device=device), batch_size=bs)
    self.bs = bs
    self.hist = []
    self.start = time.time()
    self.device = device
    self.debug = debug
    self.seed = seed
    self.seed_env()
    self.is_mlp = isinstance(self.model, ActorCritic)

  def seed_env(self):
    np.random.seed(self.seed)
    torch.manual_seed(self.seed)
    if self.device == 'cuda':
      torch.cuda.manual_seed(self.seed)
    self.env.reset(seed=self.seed)

  def compute_gae(self, rewards, values, done, next_value):
    returns, advantages = np.zeros_like(rewards), np.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
      delta = rewards[t] + self.gamma*next_value*(1-done[t]) - values[t]
      gae = delta + self.gamma*self.lam*(1-done[t])*gae
      advantages[t] = gae
      returns[t] = gae + values[t]
      next_value = values[t]
    return returns, advantages

  def evaluate_cost(self, states, actions, returns, advantages, logprob):
    kan_reg_loss = 0
    # kan_reg_loss = 0.01 * (self.model.actor.kan.regularization_loss()) if not self.is_mlp else 0
    new_logprob = self.model.actor.get_logprob(states, actions)
    entropy = (torch.log(self.model.actor.std) + 0.5 * (1 + torch.log(torch.tensor(2 * torch.pi)))).sum(dim=-1)
    ratio = torch.exp(new_logprob-logprob).squeeze()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1-self.clip_range, 1+self.clip_range) * advantages
    actor_loss = -torch.min(surr1, surr2).mean()
    critic_loss = nn.MSELoss()(self.model.critic(states).squeeze(), returns)
    entropy_loss = -self.ent_coeff * entropy.mean()
    return {"actor": actor_loss, "critic": critic_loss, "entropy": entropy_loss, "kan_reg": kan_reg_loss}

  @staticmethod
  def rollout(env, model, max_steps=1000, deterministic=False, device='cuda'):
    states, actions, rewards, dones  = [], [], [], []
    state, _ = env.reset()

    for _ in range(max_steps):
      state_tensor = torch.FloatTensor(state).to(device)
      action = model.get_action(state_tensor, deterministic=deterministic)
      next_state, reward, terminated, truncated, info = env.step(action)
      states.append(state)
      actions.append(action)
      rewards.append(reward)
      done = terminated or truncated
      dones.append(done)

      state = next_state
      if done:
        state, _ = env.reset()
    return states, actions, rewards, dones, next_state

  def train(self, max_evals=1000):
    eps = 0
    while True:
      # rollout
      start = time.perf_counter()
      states, actions, rewards, dones, next_state = self.rollout(self.env, self.model.actor, self.n_steps, device=self.device)
      rollout_time = time.perf_counter()-start

      # compute gae
      start = time.perf_counter()
      with torch.no_grad():
        state_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.FloatTensor(np.array(actions)).to(self.device)
        values = self.model.critic(state_tensor).cpu().numpy().squeeze()
        next_values = self.model.critic(next_state_tensor).cpu().numpy().squeeze()

        self.model.actor.std = self.model.actor.log_std.exp().to(self.device) # update std
        logprobs_tensor = self.model.actor.get_logprob(state_tensor, action_tensor).cpu().numpy().squeeze()

      returns, advantages = self.compute_gae(np.array(rewards), values, np.array(dones), next_values)
      gae_time = time.perf_counter()-start

      # add to buffer
      start = time.perf_counter()
      episode_dict = TensorDict(
        {
          "states": state_tensor,
          "actions": action_tensor,
          "returns": torch.FloatTensor(returns).to(self.device),
          "advantages": torch.FloatTensor(advantages).to(self.device),
          "logprobs": logprobs_tensor,
        },
        batch_size=self.n_steps
      )
      self.replay_buffer.extend(episode_dict)
      buffer_time = time.perf_counter() - start

      # update
      start = time.perf_counter()
      for _ in range(self.epochs):
        for i, batch in enumerate(self.replay_buffer):
          advantages = (batch['advantages']-torch.mean(batch['advantages']))/(torch.std(batch['advantages'])+1e-8)
          costs = self.evaluate_cost(batch['states'], batch['actions'], batch['returns'], advantages, batch['logprobs'])
          loss = costs["actor"] + 0.5 * costs["critic"] + costs["entropy"] + costs["kan_reg"]
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          break
      self.replay_buffer.empty() # clear buffer
      update_time = time.perf_counter() - start

      # debug info
      if self.debug:
        print(f"critic loss {costs['critic'].item():.3f} entropy {costs['entropy'].item():.3f} mean action {np.mean(abs(np.array(actions)))}")
        print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")

      eps += self.env_bs
      avg_reward = np.sum(rewards)/self.env_bs

      if eps > max_evals:
        print(f"Total time: {time.time() - self.start}")
        break
      else:
        # print(f'actor KAN weights {self.model.actor.kan.layers[0].scaled_spline_weight.mean():3f}')
        # check weights of mlp to make sure seeded properly
        if self.is_mlp:
          print(f"mlp weights {self.model.actor.mlp[0].weight.mean().item()}")
        print(f"mean action {np.mean(abs(np.array(actions)))} std {self.model.actor.std.mean().item()}")
        print(f"eps {eps:.2f}, reward {avg_reward:.3f}, t {time.time()-self.start:.2f}")
        print(f"Runtimes: rollout {rollout_time:.3f}, gae {gae_time:.3f}, buffer {buffer_time:.3f}, update {update_time:.3f}")
        self.hist.append((eps, avg_reward))

    return self.model.actor, self.hist

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--max_evals", type=int, default=50000)
  parser.add_argument("--env_bs", type=int, default=1000)
  parser.add_argument("--save_model", default=False)
  parser.add_argument("--noise_mode", default=None)
  parser.add_argument("--model", default="kan")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--render", default="human")
  args = parser.parse_args()

  print(f"training ppo with max_evals {args.max_evals}") 
  # env = gym.make("CartLatAccel-v0", noise_mode=args.noise_mode, env_bs=args.env_bs)
  env = CartLatAccelEnv(noise_mode=args.noise_mode, env_bs=args.env_bs)
  if args.model == "kan":
    model = KANActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1])
  else:
    model = ActorCritic(env.observation_space.shape[-1], {"pi": [32], "vf": [32]}, env.action_space.shape[-1])
  ppo = PPO(env, model, env_bs=args.env_bs, seed=args.seed)
  best_model, hist = ppo.train(args.max_evals)

  print(f"rolling out best model") 
  # env = gym.make("CartLatAccel-v0", noise_mode=args.noise_mode, env_bs=1, render_mode=args.render)
  # env = CartLatAccelEnv(noise_mode=args.noise_mode, env_bs=1, render_mode=args.render)
  states, actions, rewards, dones, next_state= ppo.rollout(env, best_model, max_steps=200, deterministic=True)
  print(f"reward {sum(rewards)[0]}")

  if args.save_model:
    os.makedirs('out', exist_ok=True)
    torch.save(best_model, 'out/best.pt')
