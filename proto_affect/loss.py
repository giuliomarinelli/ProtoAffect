import torch
from torch import Tensor
from .types import CoreState, RunConfig

def homeostasis_loss(state: CoreState) -> Tensor:
    return torch.sum((state.b - state.b_star) ** 2)

def total_loss(state: CoreState, pred_err: Tensor, cfg: RunConfig) -> Tensor:
    # V1: pred_err = 0.0; reg su s la aggiungiamo dopo
    return homeostasis_loss(state) + cfg.lambda_pred * pred_err
