import torch, random

class ToyEnv:
    def __init__(self, regime_len: int = 50, seed: int = 42):
        self.regime_len = regime_len
        random.seed(seed); torch.manual_seed(seed)

    def step(self, t: int) -> torch.Tensor:
        # alterna blocchi ostili/benigni per creare conflitti
        phase = (t // self.regime_len) % 2
        novelty = torch.rand(1).item()
        threat  = 0.7 + 0.2*torch.rand(1).item() if phase == 0 else 0.1*torch.rand(1).item()
        support = 0.7*torch.rand(1).item() if phase == 1 else 0.2*torch.rand(1).item()
        control = 0.5 + 0.5*torch.rand(1).item()
        return torch.tensor([novelty, threat, support, control], dtype=torch.float32)
