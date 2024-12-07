import os
import base64
import torch
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from pydantic import BaseModel
from gym_cartlataccel.env import BatchedCartLatAccelEnv as CartLatAccelEnv
from model import KANActorCritic, ActorCritic
from ppo_kan import PPO

class ReportConfig(BaseModel):
  model: str = "mlp"
  max_evals: int = 50000
  env_bs: int = 1000
  noise_mode: str = None
  n_runs: int = 5
  base_seed: int = 42
  render: str = "rgb_array"
  out_dir: str = "out"

def plot_loss_curves(histories, out_path):
  """Creates html plotly curve with multiple runs"""
  fig = go.Figure()
  
  for i, hist in enumerate(histories):
    steps, rewards = zip(*hist)
    fig.add_trace(go.Scatter(x=steps, y=rewards, mode='lines', name=f'Run {i+1}', opacity=0.3))
  
  # plot mean and std
  all_rewards = np.array([[r for _, r in hist] for hist in histories])
  mean_rewards = np.mean(all_rewards, axis=0)
  std_rewards = np.std(all_rewards, axis=0)
  steps = [s for s, _ in histories[0]]
  
  fig.add_trace(go.Scatter(x=steps, y=mean_rewards, mode='lines',
                          name='Mean', line=dict(width=2, color='black')))
  fig.add_trace(go.Scatter(x=steps, y=mean_rewards+std_rewards, mode='lines',
                          name='Std', line=dict(width=0), showlegend=False))
  fig.add_trace(go.Scatter(x=steps, y=mean_rewards-std_rewards, mode='lines',
                          fill='tonexty', line=dict(width=0), showlegend=False))
  
  fig.update_layout(
    title="Training Progress",
    xaxis_title="Episodes",
    yaxis_title="Reward",
    showlegend=True
  )
  
  fig_html = fig.to_html(full_html=False, include_plotlyjs='cdn')
  with open(out_path, 'w') as f:
    f.write(fig_html)

def encode_base64(video_path):
  with open(video_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode('utf-8')
  return encoded

def make_report(cfg: ReportConfig):
  os.makedirs(cfg.out_dir, exist_ok=True)
  histories = []
  final_rewards = []
  
  print(f"Training {cfg.model} model for {cfg.n_runs} runs")
  
  for run in range(cfg.n_runs):
    seed = cfg.base_seed + run
    print(f"\nRun {run+1} with seed {seed}")
    
    # Training
    env = CartLatAccelEnv(noise_mode=cfg.noise_mode, env_bs=cfg.env_bs)
    if cfg.model == "kan":
      model = KANActorCritic(env.observation_space.shape[-1], 
                              {"pi": [32], "vf": [32]}, 
                              env.action_space.shape[-1])
    else:
      model = ActorCritic(env.observation_space.shape[-1], 
                          {"pi": [32], "vf": [32]}, 
                          env.action_space.shape[-1])
        
    ppo = PPO(env, model, env_bs=cfg.env_bs, seed=seed)
    best_model, hist = ppo.train(cfg.max_evals)
    histories.append(hist)
    
    # Evaluation
    eval_env = CartLatAccelEnv(noise_mode=cfg.noise_mode, env_bs=1, 
                              render_mode=cfg.render)
    eval_env.reset(seed=seed)
    _, _, rewards, _, _ = ppo.rollout(eval_env, best_model, 
                                    max_steps=200, deterministic=True)
    final_rewards.append(sum(rewards)[0])
    
    # Save model
    torch.save(best_model, f"{cfg.out_dir}/best_model_run_{run}.pt")
  
  # Generate report
  report_path = f"{cfg.out_dir}/{cfg.model}_report.html"
  loss_plot_path = f"{cfg.out_dir}/{cfg.model}_loss_curves.html"
  plot_loss_curves(histories, loss_plot_path)
  
  with open(report_path, 'w') as f:
    f.write(f"<html><body>\n")
    f.write(f"<h1>Training Report</h1>\n")
    f.write(f"<pre>{cfg.dict()}</pre>\n")
    
    f.write(f"<h2>Results Summary</h2>\n")
    f.write(f"<p>Final rewards across {cfg.n_runs} runs:</p>\n")
    f.write(f"<ul>\n")
    for i, reward in enumerate(final_rewards):
        f.write(f"<li>Run {i+1}: {reward:.2f}</li>\n")
    f.write(f"</ul>\n")
    f.write(f"<p>Mean final reward: {np.mean(final_rewards):.2f} ± {np.std(final_rewards):.2f}</p>\n")
    
    f.write(f"<h2>Learning Curves</h2>\n")
    f.write(f'<iframe src="{cfg.model}_loss_curves.html" width="100%" height="600"></iframe>\n')
    
    f.write(f"</body></html>\n")
  
  print(f"\nReport saved at {report_path}")

if __name__ == "__main__":
  cfg = ReportConfig()
  make_report(cfg)