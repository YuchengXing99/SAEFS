"""Small training helpers (seeding, meters, optimizer / schedule builders)."""

from __future__ import annotations

import math
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

__all__ = ["seed_everything", "AverageMeter", "build_optimizer", "build_scheduler", "count_params"]


def seed_everything(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """Running mean of a scalar."""

    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float, opt: str = "adamW"):
    """AdamW with weight decay disabled for biases / 1-D parameters."""

    def no_decay(name, p):
        return p.ndim < 2 or name.endswith("bias") or "bn" in name or "ln" in name

    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    groups = [
        {"params": [p for n, p in named if no_decay(n, p)], "weight_decay": 0.0},
        {"params": [p for n, p in named if not no_decay(n, p)], "weight_decay": weight_decay},
    ]

    if opt == "adamW":
        return torch.optim.AdamW(groups, lr=lr)
    if opt == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=0.9)
    if opt == "radam":
        return torch.optim.RAdam(groups, lr=lr)
    raise ValueError(f"Unknown optimizer {opt!r}")


def build_scheduler(optimizer, name: str, num_training_steps: int, num_warmup_steps: int = 0):
    """Cosine / linear / constant schedule with linear warmup (per optimizer step)."""

    def warmup(step):
        return float(step) / float(max(1, num_warmup_steps))

    if name == "cosine":
        def fn(step):
            if step < num_warmup_steps:
                return warmup(step)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    elif name == "linear":
        def fn(step):
            if step < num_warmup_steps:
                return warmup(step)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 1.0 - progress)
    elif name == "constant":
        def fn(step):
            return warmup(step) if step < num_warmup_steps else 1.0
    else:
        raise ValueError(f"Unknown lr schedule {name!r}")

    return LambdaLR(optimizer, fn)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
