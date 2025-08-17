import torch, torch.nn as nn

class Appraisal(nn.Module):
    def __init__(self, learnable: bool = False):
        super().__init__()
        self.learnable = learnable
        if learnable:
            self.net = nn.Sequential(nn.Linear(4, 32), nn.Tanh(), nn.Linear(32, 3))
        else:
            self.register_buffer("W", torch.tensor([
                [ +0.4, -0.5, +0.5, +0.2],  # -> Δvalence
                [ +0.6, +0.3, +0.1, -0.2],  # -> Δarousal
                [ +0.1, -0.2, +0.1, +0.6],  # -> Δdominance
            ], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ds = self.net(x) if self.learnable else (self.W @ x)
        return torch.clamp(ds, -0.3, 0.3)
