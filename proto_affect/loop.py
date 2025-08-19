# proto_affect/loop.py (sostituisci l'intera run)
import torch
import torch.optim as optim
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

    # baseline per REINFORCE con advantage
    baseline = torch.tensor(0.0)
    beta = 0.99       # EMA per la baseline
    ent_coef = 0.005   # bonus entropia per mantenere esplorazione

    try:
        for t in range(cfg.T):
            # 1) evento dall'ambiente
            x = env.step(t)

            # 2) appraisal -> Δs e applico inerzia
            ds = appraisal(x)
            apply_affect(state, ds, inertia=cfg.inertia_s)

            # 3) azione dalla policy (temperatura modulata da NE)
            ne = float(state.m[2].item())
            a, logp, probs = policy.act(state.b, state.s, ne_level=ne)

            # 4) transizione di stato (omeostasi + effetti azione)
            apply_transition(state, a, decay_b=cfg.decay_b)

            # 5) loss (per ora senza pred_err reale)
            pred_err = torch.tensor(0.0)
            L = total_loss(state, pred_err, cfg)

            # 6) Advantage: quanto questa azione è stata peggiore/migliore della baseline
            adv = L.detach() - baseline
            baseline = beta * baseline + (1.0 - beta) * L.detach()

            # 7) Entropia per non collassare su un'unica azione
            # (evitiamo log(0) con un clamp)
            entropy = -(probs * torch.clamp(probs, min=1e-8).log()).sum()

            # 8) Policy gradient con advantage e bonus entropia
            policy_loss = adv * logp - ent_coef * entropy
            opt.zero_grad()
            policy_loss.backward()
            opt.step()

            # 9) Logging
            reward = -L.detach()
            rec = to_step_record(
                t=t, state=state, x=x, action=a, reward=float(reward), loss=float(L.item())
            )
            logger.append(run_id, rec)

            if (t + 1) % 200 == 0:
                print(f"[t={t+1}] loss={L.item():.4f}  adv={float(adv):.4f}  H={float(entropy):.3f}  a={a}")

    finally:
        logger.close()
