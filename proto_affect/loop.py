import torch, torch.optim as optim
from .types import RunConfig, to_step_record
from .env import ToyEnv
from .state import init_state, apply_affect, apply_transition
from .appraisal import Appraisal
from .policy import Policy
from .loss import total_loss
from .log import SQLiteLogger

def run(cfg: RunConfig = RunConfig(), db_path: str = "data/runs.sqlite"):
    torch.manual_seed(cfg.seed)

    env = ToyEnv(regime_len=50, seed=cfg.seed)
    state = init_state()
    appraisal = Appraisal(learnable=False)
    policy = Policy()
    opt = optim.Adam(policy.parameters(), lr=1e-3)

    logger = SQLiteLogger(db_path=db_path, flush_every=200)
    run_id = logger.start_run(cfg)

    try:
        for t in range(cfg.T):
            x = env.step(t)
            ds = appraisal(x)
            apply_affect(state, ds, inertia=cfg.inertia_s)

            ne = float(state.m[2].item())
            a, logp, _ = policy.act(state.b, state.s, ne_level=ne)
            apply_transition(state, a, decay_b=cfg.decay_b)

            pred_err = torch.tensor(0.0)
            L = total_loss(state, pred_err, cfg)
            reward = -L.detach()

            # REINFORCE minimale
            opt.zero_grad()
            (-reward * logp).backward()
            opt.step()

            # ---- LOG SU SQLITE ----
            rec = to_step_record(t=t, state=state, x=x, action=a, reward=float(reward), loss=float(L.item()))
            logger.append(run_id, rec)

            if (t + 1) % 200 == 0:
                print(f"[t={t+1}] loss={L.item():.4f}")

    finally:
        logger.close()
