from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TypedDict, Optional
from torch import Tensor

# ---------- Alias tensor ----------
Vec3 = Tensor            # shape (3,)
Vec4 = Tensor            # shape (4,)  -> [novelty, threat, support, control]

# ---------- Azioni disponibili ----------
class Action(IntEnum):
    REST = 0          # riposo/recupero energia
    SEEK_SUPPORT = 1  # cerca legame/supporto
    AVOID = 2         # evita/minaccia
    EXPLORE = 3       # esplora/novità

# ---------- Stato interno ----------
@dataclass
class CoreState:
    """
    Stato interno "vivo" dell'agente.
    b: bisogni [energia, sicurezza, legame] ∈ [0,1]
    b_star: setpoint di omeostasi (stessa semantica di b)
    s: affect [valence(-1..1), arousal(0..1), dominance(-1..1)]
    m: modulatori [DA, 5HT, NE] ∈ [0,1]
    """
    b: Vec3
    b_star: Vec3
    s: Vec3
    m: Vec3

    def clone(self) -> "CoreState":
        # Utile per logging senza aliasing dei tensori
        return CoreState(b=self.b.clone(), b_star=self.b_star.clone(),
                         s=self.s.clone(), m=self.m.clone())

# ---------- Evento/Osservazione dall'ambiente ----------
@dataclass
class Event:
    """
    x: vettore evento (novità, minaccia, supporto, controllabilità)
    """
    x: Vec4

# ---------- Record di uno step (per logging/persistenza) ----------
class StepRecord(TypedDict, total=False):
    t: int
    novelty: float
    threat: float
    support: float
    control: float

    b_energy: float
    b_safety: float
    b_belong: float

    s_valence: float
    s_arousal: float
    s_dominance: float

    m_da: float
    m_5ht: float
    m_ne: float

    action: int
    reward: float
    loss: float

# ---------- Configurazione run (minima) ----------
@dataclass
class RunConfig:
    seed: int = 42
    T: int = 1000
    decay_b: float = 0.02
    inertia_s: float = 0.96
    tau0: float = 1.0      # temperatura base della policy
    kappa_ne: float = 0.8  # quanto NE modula la temperatura
    lambda_pred: float = 0.0   # V1: spesso 0, si attiva dopo
    lambda_smooth: float = 0.01

    # clamp per s
    v_min: float = -1.0
    v_max: float = 1.0
    a_min: float = 0.0
    a_max: float = 1.0
    d_min: float = -1.0
    d_max: float = 1.0

# ---------- Helper piccoli (opzionali) ----------
def to_step_record(t: int, state: CoreState, x: Optional[Vec4],
                   action: Optional[Action], reward: float, loss: float) -> StepRecord:
    rec: StepRecord = {"t": t, "reward": float(reward), "loss": float(loss)}
    if x is not None:
        rec.update(dict(
            novelty=float(x[0]), threat=float(x[1]),
            support=float(x[2]), control=float(x[3])
        ))
    rec.update(dict(
        b_energy=float(state.b[0]), b_safety=float(state.b[1]), b_belong=float(state.b[2]),
        s_valence=float(state.s[0]), s_arousal=float(state.s[1]), s_dominance=float(state.s[2]),
        m_da=float(state.m[0]), m_5ht=float(state.m[1]), m_ne=float(state.m[2]),
    ))
    if action is not None:
        rec["action"] = int(action)
    return rec
