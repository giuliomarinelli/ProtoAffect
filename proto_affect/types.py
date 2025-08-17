from dataclasses import dataclass
from typing import Tuple
import torch

Vec3 = torch.Tensor  # shape (3,)

@dataclass
class CoreState:
    b: Vec3        # bisogni: [energia, sicurezza, legame] in [0,1]
    b_star: Vec3   # setpoint
    s: Vec3        # affect: [valence(-1..1), arousal(0..1), dominance(-1..1)]
    m: Vec3        # modulatori: [DA, 5HT, NE] in [0,1]
