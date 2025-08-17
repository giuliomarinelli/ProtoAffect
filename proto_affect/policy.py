from torch import nn
import torch

class Policy(nn.Module):
    def __init__(self, n_actions=4):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(6,16), nn.Tanh(), nn.Linear(16,n_actions))

    def act(self, b, s, tau):
        z = torch.cat([b, s], dim=-1)
        logits = self.actor(z)
        probs = torch.softmax(logits / tau, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        a = dist.sample()
        return a, dist.log_prob(a), probs
