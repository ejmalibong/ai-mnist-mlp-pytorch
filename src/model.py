import torch
import torch.nn as nn

class MLP(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(28*28, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 10)
    )

  def forward(self, x):
    x = x.flatten(start_dim=1)
    return self.net(x)