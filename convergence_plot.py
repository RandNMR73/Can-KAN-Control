import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from ppo_kan import PPO, CartLatAccelEnv, KANActorCritic, ActorCritic

def train_model(eq_num, model_type='kan', hidden_size=32, max_evals=1000, env_bs=100, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # initialize env with equation number
    env = CartLatAccelEnv(noise_mode=None, env_bs=env_bs, eq=eq_num)
    if model_type == 'kan':
        model = KANActorCritic(
            env.observation_space.shape[-1], 
            {"pi": [hidden_size], "vf": [32]}, 
            env.action_space.shape[-1], 
            act_bound=(-1,1)
        )
    else:
        model = ActorCritic(
            env.observation_space.shape[-1], 
            {"pi": [hidden_size], "vf": [32]}, 
            env.action_space.shape[-1], 
            act_bound=(-1,1)
        )

    # train model
    ppo = PPO(env, model, env_bs=env_bs, device=device, seed=seed, debug=False)
    best_model, hist = ppo.train(max_evals)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return best_model, hist

def plot_convergence(eq_num, seeds, model_type='kan', max_evals=100000, env_bs=1000, hidden_size=32):
    plt.figure(figsize=(10, 6))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_histories = {model_type: {}}
    rewards = []
    
    # run same trajectory with different seeds
    for seed in seeds:
        print(f"Training with seed {seed}")
        model, hist = train_model(eq_num, model_type, hidden_size, max_evals, env_bs, seed)
        
        all_histories[model_type][seed] = hist
        rewards.append(hist['reward'])
    
    # save
    os.makedirs('out/convergence_histories', exist_ok=True)
    history_path = f'out/convergence_histories/eq{eq_num}_{model_type}_histories.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(all_histories, f)
    
    rewards = np.array(rewards)
    mean_reward = np.mean(rewards, axis=0)
    std_reward = np.std(rewards, axis=0)
    
    # plot with std
    iterations = hist['iter']
    color = 'blue' if model_type == 'kan' else 'red'
    plt.plot(iterations, mean_reward, label=model_type.upper(), color=color)
    plt.fill_between(iterations, mean_reward - std_reward, mean_reward + std_reward, 
                    alpha=0.2, color=color)
    
    plt.xlabel('Iterations')
    plt.ylabel('Average Reward')
    plt.title(f'Training Convergence - Equation {eq_num} ({model_type.upper()})')
    plt.legend()
    plt.grid(True)
    
    os.makedirs('out/convergence_plots', exist_ok=True)
    save_path = f'out/convergence_plots/eq{eq_num}_{model_type}_convergence.png'
    plt.savefig(save_path)
    plt.close()

def main(args):
    convergence_seeds = [42, 43, 44, 45, 46]
    max_evals = 100000
    env_bs = 1000
    hidden_size = 32
    
    for eq_num in [args.eq_num]:
        print(f"\nProcessing equation {eq_num}")
        plot_convergence(eq_num, convergence_seeds, args.model_type, max_evals, env_bs, hidden_size)
        print(f"Saved convergence plot and history for equation {eq_num}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eq_num', type=int, default=12)
    parser.add_argument('--model_type', type=str, default='mlp')
    args = parser.parse_args()
    
    main(args) 