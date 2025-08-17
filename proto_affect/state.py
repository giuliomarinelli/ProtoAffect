from dataclasses import dataclass
import torch


@dataclass
class CoreState:
    b: torch.Tensor   # (3,)
    s: torch.Tensor   # (3,)
    m: torch.Tensor   # (3,)
    b_star: torch.Tensor  # (3,)