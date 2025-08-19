from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch import Tensor

Vec4 = Tensor  # [novelty, threat, support, control]

class ToyEnv:
    """
    Ambiente sintetico che genera eventi x_t = [novità, minaccia, supporto, controllabilità]
    alternando blocchi 'ostili' e 'benigni' per creare conflitti.

    - Riproducibile: usa un torch.Generator locale.
    - Parametrico: base ostile/benigno + rumore.
    - Shock opzionale: finestra temporale che altera i segnali (es. minaccia alta).
    """

    def __init__(
        self,
        regime_len: int = 50,
        seed: int = 42,
        noise: float = 0.10,
        # base dei due regimi: valori in [0,1] per [novelty, threat, support, control]
        base_hostile: Tuple[float, float, float, float] = (0.5, 0.8, 0.2, 0.6),
        base_benign:  Tuple[float, float, float, float] = (0.6, 0.2, 0.7, 0.7),
        device: Optional[torch.device] = None,
        # shock: (start_t, end_t, delta_vec) con delta in [-1,1] per ciascuna dimensione
        shock: Optional[Tuple[int, int, Tuple[float, float, float, float]]] = None,
    ):
        self.regime_len = regime_len
        self.noise = noise
        self.device = device if device is not None else torch.device("cpu")

        self._gen = torch.Generator(device=self.device)
        self._gen.manual_seed(seed)

        self._base_hostile = torch.tensor(base_hostile, dtype=torch.float32, device=self.device)
        self._base_benign  = torch.tensor(base_benign,  dtype=torch.float32, device=self.device)

        self._shock_window: Optional[Tuple[int, int]] = None
        self._shock_delta: Optional[Tensor] = None
        if shock is not None:
            (s, e, d) = shock
            self.schedule_shock(s, e, torch.tensor(d, dtype=torch.float32, device=self.device))

    # ---------- API ----------
    def reset(self) -> None:
        """Per simmetria con ambienti RL. Non mantiene stato interno complesso per ora."""
        pass

    def set_seed(self, seed: int) -> None:
        self._gen.manual_seed(seed)

    def schedule_shock(self, start_t: int, end_t: int, delta: Tensor) -> None:
        """Aggiunge un delta ai segnali durante [start_t, end_t)."""
        self._shock_window = (start_t, end_t)
        self._shock_delta = delta

    def step(self, t: int) -> Vec4:
        """
        Genera x_t. La fase alterna ogni `regime_len` step:
          - fase 0: ostile (minaccia↑, supporto↓)
          - fase 1: benigno (supporto↑, minaccia↓)
        """
        phase = (t // self.regime_len) % 2
        base = self._base_hostile if phase == 0 else self._base_benign

        # rumore uniforme in [-noise, +noise]
        eps = (torch.rand(4, generator=self._gen, device=self.device) - 0.5) * (2.0 * self.noise)

        x = base + eps
        # opzionale: shock
        if self._shock_window is not None and self._shock_delta is not None:
            s, e = self._shock_window
            if s <= t < e:
                x = x + self._shock_delta

        # clamp in [0,1]
        return torch.clamp(x, 0.0, 1.0)
