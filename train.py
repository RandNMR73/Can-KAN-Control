# train kan supervised off of feynman dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from gym_cartlataccel.feynman import get_feynman_dataset
import numpy as np
from efficient_kan import KAN
from model import mlp

def get_dataset(name):
  dataset = get_feynman_dataset(name)
  symbol, expr, f, ranges = dataset

  ranges = np.array(ranges)
  num_symbols = len(symbol)

  if ranges.ndim == 1:
    # copy ranges to match num_symbols
    ranges = np.array([ranges] * num_symbols)

  xs = torch.rand(N, num_symbols)
  # scale each column by range
  for i in range(num_symbols):
    xs[:,i] = xs[:,i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
  ys = f(xs)

  return xs, ys

def train(model, xs, ys):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    batch_size = 32
    train_dataset = torch.utils.data.TensorDataset(xs, ys)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs): 
        model.train()
        total_loss = 0
        # with tqdm(trainloader) as pbar:
        for inputs, targets in trainloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # pbar.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / len(trainloader)
        # print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")
    return model


def get_error(model, x_test, y_test):
  model.eval()
  x_test = x_test.to(device)
  y_test = y_test.to(device)
  y_pred = model(x_test)
  return criterion(y_pred, y_test)

def train_test(name):
  xs, ys = get_dataset(name)
  hidden_sizes = [32]
  layers = [xs.shape[1]] + hidden_sizes + [ys.shape[1]]
  model = KAN(layers, grid_size=5, spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
  mlp_model = mlp(layers)

  model = train(model, xs, ys)
  mlp_model = train(mlp_model, xs, ys)

  x_test, y_test = get_dataset(name)
  kan_error = get_error(model, x_test, y_test)
  mlp_error = get_error(mlp_model, x_test, y_test)
  print(f"{name} KAN error: {kan_error.item()}, MLP error: {mlp_error.item()}")

if __name__ == "__main__":
  N = 1000
  n_epochs = 10
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  criterion = nn.MSELoss()
  # name = 'test'
  for i in range(1, 10):
    train_test(i)
