import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from ppo_kan import PPO, CartLatAccelEnv, KANActorCritic, FourierKANActorCritic, WaveletKANActorCritic, LegendreKANActorCritic, LaplaceKANActorCritic, MixedKANActorCritic, ActorCritic, ChebyKANActorCritic

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
    elif model_type == 'Fkan':
        model = FourierKANActorCritic(
            env.observation_space.shape[-1], 
            {"pi": [hidden_size], "vf": [32]}, 
            env.action_space.shape[-1], 
            act_bound=(-1,1)
        )
    elif model_type == 'Wkan':
        model = WaveletKANActorCritic(
            env.observation_space.shape[-1], 
            {"pi": [hidden_size], "vf": [32]}, 
            env.action_space.shape[-1], 
            act_bound=(-1,1)
        )
    elif model_type == 'Lkan':
        model = LaplaceKANActorCritic(
            env.observation_space.shape[-1], 
            {"pi": [hidden_size], "vf": [32]}, 
            env.action_space.shape[-1], 
            act_bound=(-1,1)
        )
    elif model_type == 'Ckan':
        model = LaplaceKANActorCritic(
            env.observation_space.shape[-1], 
            {"pi": [hidden_size], "vf": [32]}, 
            env.action_space.shape[-1], 
            act_bound=(-1,1)
        )
    # elif model_type == 'Lekan':
    #     model = LegendreKANActorCritic(
    #         env.observation_space.shape[-1], 
    #         {"pi": [hidden_size], "vf": [32]}, 
    #         env.action_space.shape[-1], 
    #         act_bound=(-1,1)
    #     )
    elif model_type == 'Mkan':
        model = MixedKANActorCritic(
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
    
    all_results = {'kan': [], 'Fkan':[], 'Wkan':[], 'Lkan':[], 'Mkan':[], 'Ckan':[],'mlp': []}
    
    for eq_num in range(111, -1, -1): #121):
        if eq_num == 102:
            continue 
        print(f"equation {eq_num}")
        
        # kan_model, kan_hist = train_model(eq_num, 'kan', hidden_size, max_evals, env_bs, train_seed)
        Fkan_model, Fkan_hist = train_model(eq_num, 'Fkan', hidden_size, max_evals, env_bs, train_seed)
        # Lekan_model, Lekan_hist = train_model(eq_num, 'Lekan', hidden_size, max_evals, env_bs, train_seed)
        # Mkan_model, Mkan_hist = train_model(eq_num, 'Mkan', hidden_size, max_evals, env_bs, train_seed)
        # Ckan_model, Ckan_hist = train_model(eq_num, 'Ckan', hidden_size, max_evals, env_bs, train_seed)
        # mlp_model, mlp_hist = train_model(eq_num, 'mlp', hidden_size, max_evals, env_bs, train_seed)
        # kan_results = evaluate_model(kan_model, eq_num, eval_seeds, device)
        Fkan_results = evaluate_model(Fkan_model, eq_num, eval_seeds, device)
        # Wkan_results = evaluate_model(Wkan_model, eq_num, eval_seeds, device)
        # Lkan_results = evaluate_model(Lkan_model, eq_num, eval_seeds, device)
        # Ckan_results = evaluate_model(Ckan_model, eq_num, eval_seeds, device)
        # Lekan_results = evaluate_model(Lekan_model, eq_num, eval_seeds, device)
        # Mkan_results = evaluate_model(Mkan_model, eq_num, eval_seeds, device)
        # mlp_results = evaluate_model(mlp_model, eq_num, eval_seeds, device)
        # kan_rewards = [r['reward'] for r in kan_results]
        Fkan_rewards = [r['reward'] for r in Fkan_results]
        # Wkan_rewards = [r['reward'] for r in Wkan_results]
        # Lkan_rewards = [r['reward'] for r in Lkan_results]
        # Ckan_rewards = [r['reward'] for r in Ckan_results]
        # Lekan_rewards = [r['reward'] for r in Lekan_results]
        # Mkan_rewards = [r['reward'] for r in Mkan_results]
        # mlp_rewards = [r['reward'] for r in mlp_results]
        
        # all_results['kan'].append({
        #     'eq': eq_num,
        #     'mean': np.mean(kan_rewards),
        #     'std': np.std(kan_rewards)
        # })
        all_results['Fkan'].append({
            'eq': eq_num,
            'mean': np.mean(Fkan_rewards),
            'std': np.std(Fkan_rewards)
        })
        # all_results['Wkan'].append({
        #     'eq': eq_num,
        #     'mean': np.mean(Wkan_rewards),
        #     'std': np.std(Wkan_rewards)
        # })
        # all_results['Lkan'].append({
        #     'eq': eq_num,
        #     'mean': np.mean(Lkan_rewards),
        #     'std': np.std(Lkan_rewards)
        # })
        # all_results['Ckan'].append({
        #     'eq': eq_num,
        #     'mean': np.mean(Ckan_rewards),
        #     'std': np.std(Ckan_rewards)
        # })
        # all_results['Lekan'].append({
        #     'eq': eq_num,
        #     'mean': np.mean(Lekan_rewards),
        #     'std': np.std(Lekan_rewards)
        # })
        # all_results['Mkan'].append({
        #     'eq': eq_num,
        #     'mean': np.mean(Mkan_rewards),
        #     'std': np.std(Mkan_rewards)
        # })
        # all_results['mlp'].append({
        #     'eq': eq_num,
        #     'mean': np.mean(mlp_rewards),
        #     'std': np.std(mlp_rewards)
        # })
        
        for model_type, hist in [('FKAN', Fkan_hist)]:
            plot_losses(hist, 
                       save_path=f'results/eq{eq_num}_{model_type.lower()}_learning_curve.png',
                       title=f'{model_type} Learning Curve - Equation {eq_num}')
    
    print("\nOverall Results:")
    for model_type in ['Fkan']:
        means = [r['mean'] for r in all_results[model_type]]
        print(f"\n{model_type.upper()}:")
        print(f"Average across all equations - Mean: {np.mean(means):.3f}, Std: {np.std(means):.3f}")
        
        with open(f'results/{model_type}_detailed_results.txt', 'w') as f:
            for result in all_results[model_type]:
                f.write(f"Equation {result['eq']}: Mean = {result['mean']:.3f}, Std = {result['std']:.3f}\n")

if __name__ == "__main__":
    main()