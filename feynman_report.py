import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ppo_kan import PPO, CartLatAccelEnv, KANActorCritic, ActorCritic

def plot_losses(hist, save_path=None, title=None):
    plt.figure(figsize=(10, 6))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(hist['iter'], hist['total_loss'], label='Total Loss')
    ax1.plot(hist['iter'], hist['value_loss'], label='Value Loss')
    ax1.plot(hist['iter'], hist['policy_loss'], label='Policy Loss')
    ax1.set_xlabel('n_iters')
    ax1.set_ylabel('loss')
    ax1.legend()

    ax2.plot(hist['iter'], hist['reward'], label='Average Reward', color='green')
    ax2.set_xlabel('n_iters')
    ax2.set_ylabel('reward')
    ax2.legend()
    
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()
    plt.close(fig)

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
    
    return best_model, hist

def evaluate_model(model, eq_num, eval_seeds, device):
    results = []
    
    for seed in eval_seeds:
        eval_env = CartLatAccelEnv(noise_mode=None, env_bs=1, eq=eq_num)
        eval_env.reset(seed=seed)
        _, _, rewards, _, _ = PPO.rollout(eval_env, model, max_steps=200, device=device, deterministic=True)
        final_reward = sum(rewards)[0]
        
        results.append({
            'seed': seed,
            'reward': final_reward,
        })
        print(f"Seed {seed} final reward: {final_reward:.3f}")
    
    return results

def main():
    train_seed = 42
    eval_seeds = range(10)
    max_evals = 100000
    env_bs = 1000
    hidden_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    os.makedirs('results', exist_ok=True)
    
    all_results = {'kan': [], 'mlp': []}
    
    for eq_num in range(1, 10): #121):
        print(f"equation {eq_num}")
        
        kan_model, kan_hist = train_model(eq_num, 'kan', hidden_size, max_evals, env_bs, train_seed)
        mlp_model, mlp_hist = train_model(eq_num, 'mlp', hidden_size, max_evals, env_bs, train_seed)
        kan_results = evaluate_model(kan_model, eq_num, eval_seeds, device)
        mlp_results = evaluate_model(mlp_model, eq_num, eval_seeds, device)
        kan_rewards = [r['reward'] for r in kan_results]
        mlp_rewards = [r['reward'] for r in mlp_results]
        
        all_results['kan'].append({
            'eq': eq_num,
            'mean': np.mean(kan_rewards),
            'std': np.std(kan_rewards)
        })
        all_results['mlp'].append({
            'eq': eq_num,
            'mean': np.mean(mlp_rewards),
            'std': np.std(mlp_rewards)
        })
        
        for model_type, hist in [('KAN', kan_hist), ('MLP', mlp_hist)]:
            plot_losses(hist, 
                       save_path=f'results/eq{eq_num}_{model_type.lower()}_learning_curve.png',
                       title=f'{model_type} Learning Curve - Equation {eq_num}')
    
    print("\nOverall Results:")
    for model_type in ['kan', 'mlp']:
        means = [r['mean'] for r in all_results[model_type]]
        print(f"\n{model_type.upper()}:")
        print(f"Average across all equations - Mean: {np.mean(means):.3f}, Std: {np.std(means):.3f}")
        
        with open(f'results/{model_type}_detailed_results.txt', 'w') as f:
            for result in all_results[model_type]:
                f.write(f"Equation {result['eq']}: Mean = {result['mean']:.3f}, Std = {result['std']:.3f}\n")

if __name__ == "__main__":
    main()