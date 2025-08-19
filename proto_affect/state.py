import torch
from .types import CoreState

CLAMP_V = (-1.0, 1.0); CLAMP_A = (0.0, 1.0); CLAMP_D = (-1.0, 1.0)

def init_state() -> CoreState:
    return CoreState(
        b=torch.tensor([0.6, 0.6, 0.6]), 
        b_star=torch.tensor([0.7, 0.7, 0.7]),
        s=torch.tensor([0.0, 0.2, 0.0]),
        m=torch.tensor([0.5, 0.5, 0.5]),
    )

def apply_affect(state: CoreState, ds: torch.Tensor, inertia: float = 0.96) -> None:
    s = state.s
    s = torch.stack([
        torch.clamp(inertia*s[0] + ds[0], *CLAMP_V),
        torch.clamp(inertia*s[1] + ds[1], *CLAMP_A),
        torch.clamp(inertia*s[2] + ds[2], *CLAMP_D),
    ])
    state.s = s

def apply_transition(state: CoreState, action: int, decay_b: float = 0.015) -> None:
    state.b = torch.clamp(state.b - decay_b, 0.0, 1.0)
    if action == 0:   # rest
        state.b[0] = torch.clamp(state.b[0] + 0.08, 0, 1); state.s[2] -= 0.02
    elif action == 1: # seek_support
        state.b[2] = torch.clamp(state.b[2] + 0.06, 0, 1); state.s[1] += 0.02
    elif action == 2: # avoid  (più forte)
        state.b[1] = torch.clamp(state.b[1] + 0.10, 0, 1); state.s[0] -= 0.02
    elif action == 3: # explore (meno punitivo)
        state.b[1] = torch.clamp(state.b[1] - 0.015, 0, 1); state.s[1] += 0.03
