import torch
from .types import CoreState

def compute_loss(state: CoreState, pred_err: torch.Tensor,
                 lambda_pred: float = 0.3, lambda_smooth: float = 0.01) -> torch.Tensor:
    homeo = torch.sum((state.b - state.b_star)**2)
    smooth = torch.sum((state.s[1:] - state.s[1:].detach())*0)  # placeholder reg
    return homeo + lambda_pred*pred_err + lambda_smooth*smooth
