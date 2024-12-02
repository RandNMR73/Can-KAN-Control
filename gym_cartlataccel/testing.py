from env import BatchedCartLatAccelEnv as CartLatAccelEnv
import matplotlib.pyplot as plt

n_traj = 20

env = CartLatAccelEnv(noise_mode=None, env_bs=n_traj)

orig = env.generate_traj(n_traj)
points, new = env.generate_feynman_traj(n_traj, eq=4)
print(orig)
print(orig.shape)
print()

print(new)
print(new.shape)
print(points.shape)

for i, x in enumerate(new):
    plt.figure(figsize=(10, 4))
    for j in range(points.shape[2]):
        plt.plot(points[i,:,j], label=f'v{j}', linewidth=1)
    plt.plot(x, label=f'Series {i + 1}', linewidth=3)
    plt.title(f'Series {i + 1}')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()