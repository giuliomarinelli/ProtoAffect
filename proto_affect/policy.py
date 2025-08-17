import torch, torch.nn as nn

class Policy(nn.Module):
    def __init__(self, n_actions: int = 4):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, n_actions))

    def act(self, b: torch.Tensor, s: torch.Tensor, ne_level: float = 0.5):
        z = torch.cat([b, s], dim=-1)
        logits = self.actor(z)
        tau = 1.0 + 0.8*ne_level
        probs = torch.softmax(logits / tau, dim=-1)
        dist  = torch.distributions.Categorical(probs=probs)
        a     = dist.sample()
        return a.item(), dist.log_prob(a), probs
