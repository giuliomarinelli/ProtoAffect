import torch
from torch import nn

class Appraisal(nn.Module):
    def __init__(self, learnable: bool=False):
        super().__init__()
        if learnable:
            self.net = nn.Sequential(nn.Linear(4,32), nn.Tanh(),
                                     nn.Linear(32,3))
        else:
            self.register_buffer("W", torch.tensor([...], dtype=torch.float32))
            self.net = None

    def forward(self, x):
        if self.net: ds = self.net(x)
        else:        ds = torch.clamp(self.W @ x, -0.3, 0.3)
        return ds
