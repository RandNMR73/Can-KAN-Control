import re
from collections import defaultdict

def extract_and_average_rewards_per_equation(filename):
    # Dictionaries to store rewards per equation
    kan_rewards_per_equation = defaultdict(list)
    mlp_rewards_per_equation = defaultdict(list)
    
    # Open and read the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Regex patterns to identify rewards
    equation_pattern = re.compile(r'^equation\s+(\d+)')
    kan_pattern = re.compile(r'kan')
    mlp_pattern = re.compile(r'mlp')
    reward_pattern = re.compile(r'reward:\s+(-?\d+\.\d+)')
    
    current_equation = None
    current_model = None  # Tracks whether we're in 'kan' or 'mlp'

    # Loop through lines and parse
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        
        # Check for equation line
        equation_match = equation_pattern.match(line)
        if equation_match:
            current_equation = int(equation_match.group(1))
            continue
        
        # Check if we're switching to kan or mlp
        if kan_pattern.match(line):
            current_model = 'kan'
            continue
        
        if mlp_pattern.match(line):
            current_model = 'mlp'
            continue
        
        # Check for reward lines
        reward_match = reward_pattern.search(line)
        if reward_match and current_equation is not None and current_model is not None:
            reward = float(reward_match.group(1))
            if current_model == 'kan':
                kan_rewards_per_equation[current_equation].append(reward)
            elif current_model == 'mlp':
                mlp_rewards_per_equation[current_equation].append(reward)
    
    # Calculate averages per equation
    kan_averages = {eq: sum(rewards) / len(rewards) for eq, rewards in kan_rewards_per_equation.items()}
    mlp_averages = {eq: sum(rewards) / len(rewards) for eq, rewards in mlp_rewards_per_equation.items()}
    
    return kan_averages, mlp_averages

# Filepath to your text document
filename = 'results.txt'

# Extract averages
kan_avg_per_eq, mlp_avg_per_eq = extract_and_average_rewards_per_equation(filename)

kan_dict = {}
mlp_dict = {}

print("\nAverage rewards per equation for kan:")
if kan_avg_per_eq:
    for equation, avg in sorted(kan_avg_per_eq.items()):
        kan_dict[equation] = avg
print(kan_dict)
# save kan dict
import pickle
with open('kan_dict.pkl', 'wb') as f:
    pickle.dump(kan_dict, f)

print("\nAverage rewards per equation for mlp:")
if mlp_avg_per_eq:
    for equation, avg in sorted(mlp_avg_per_eq.items()):
        mlp_dict[equation] = avg
print(mlp_dict)
# save mlp dict
with open('mlp_dict.pkl', 'wb') as f:
    pickle.dump(mlp_dict, f)

eqs = [2,3,5,12,14,17,20,21,26,27,29,30,31,38,43,48,51,52,56,62,64,80,82,84,90,91,98]
for elem in [f"KAN: {kan_dict[i]}, MLP: {mlp_dict[i]}" for i in eqs]:
    print(elem)