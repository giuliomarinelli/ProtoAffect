import torch, torch.optim as optim
from .env import ToyEnv
from .state import init_state, apply_affect, apply_transition
from .appraisal import Appraisal
from .policy import Policy
from .loss import compute_loss

def run(T: int = 500):
    env = ToyEnv()
    state = init_state()
    appraisal = Appraisal(learnable=False)
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    for t in range(T):
        x = env.step(t)
        ds = appraisal(x)
        apply_affect(state, ds)

        a, logp, _ = policy.act(state.b, state.s, ne_level=state.m[2].item())
        apply_transition(state, a)

        pred_err = torch.tensor(0.0)  # V1: disattivato
        loss = compute_loss(state, pred_err)

        # policy gradient minimalista
        opt.zero_grad()
        (-loss.detach() * logp).backward()
        opt.step()
