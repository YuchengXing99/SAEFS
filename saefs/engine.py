"""Training / evaluation loops for SAEFS."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader

from .losses import EvidentialSurvLoss
from .utils import AverageMeter

try:  # optional: IBS / INBLL
    from pycox.evaluation import EvalSurv
except ImportError:  # pragma: no cover
    EvalSurv = None

__all__ = ["train_one_epoch", "evaluate", "build_time_grid", "c_index"]


# keys of ``model.forward`` output that are one scalar per case and worth dumping
_SCALAR_DIAGNOSTICS = (
    "u_wsi", "u_tg", "u_path", "u_text", "u_text_raw", "u_fused", "K_conflict", "K_pre", "cos_ht",
)


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, device, accum_steps: int = 1,
                    log_every: int = 0) -> dict:
    """One training pass. ``log_every > 0`` prints a running loss every N batches."""
    model.train()
    loss_fn.train()          # advances the loss's KL-annealing counter
    meters = {"loss": AverageMeter(), "bag_size": AverageMeter()}
    risks, censorships, event_times = [], [], []

    optimizer.zero_grad()
    for batch_idx, batch in enumerate(loader):
        x_path = batch["img"].to(device)
        x_text = batch["text"].to(device)
        label = batch["label"].to(device)
        censorship = batch["censorship"].to(device)
        event_time = batch["survival_time"].to(device)

        out, log_dict = model(x_path, x_text, label=label, censorship=censorship, loss_fn=loss_fn)
        if out.get("loss") is None:
            continue

        (out["loss"] / accum_steps).backward()
        if (batch_idx + 1) % accum_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        risks.append(out["risk"].detach().cpu().numpy())
        censorships.append(censorship.cpu().numpy())
        event_times.append(event_time.cpu().numpy())

        meters["loss"].update(log_dict["surv_loss"], n=x_path.size(0))
        meters["bag_size"].update(x_path.size(1), n=x_path.size(0))

        if log_every and ((batch_idx + 1) % log_every == 0 or batch_idx == len(loader) - 1):
            print(f"  batch {batch_idx + 1}/{len(loader)}  loss {meters['loss'].avg:.4f}")

    results = {k: m.avg for k, m in meters.items()}
    results["c_index"] = c_index(np.concatenate(risks), np.concatenate(censorships),
                                 np.concatenate(event_times))
    results["lr"] = optimizer.param_groups[0]["lr"]
    return results


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, dump: bool = True) -> tuple[dict, dict]:
    """Evaluate and (optionally) return per-case predictions + evidential diagnostics."""
    model.eval()
    loss_fn.eval()           # freezes the KL-annealing counter
    meters = {"loss": AverageMeter(), "bag_size": AverageMeter()}
    risks, censorships, event_times, survivals, pmfs = [], [], [], [], []
    diagnostics = {k: [] for k in _SCALAR_DIAGNOSTICS}
    case_ids = []

    for batch in loader:
        x_path = batch["img"].to(device)
        x_text = batch["text"].to(device)
        label = batch["label"].to(device)
        censorship = batch["censorship"].to(device)
        event_time = batch["survival_time"].to(device)

        out, log_dict = model(x_path, x_text, label=label, censorship=censorship, loss_fn=loss_fn)

        risks.append(out["risk"].cpu().numpy())
        censorships.append(censorship.cpu().numpy())
        event_times.append(event_time.cpu().numpy())
        survivals.append(np.atleast_2d(out["survival"].cpu().numpy()))
        pmfs.append(np.atleast_2d(out["pmf"].cpu().numpy()))
        case_ids.extend(batch["case_id"])
        for k in diagnostics:
            v = out.get(k)
            if torch.is_tensor(v):
                diagnostics[k].append(v.reshape(v.shape[0], -1).mean(dim=1).cpu().numpy())

        meters["loss"].update(log_dict["surv_loss"], n=x_path.size(0))
        meters["bag_size"].update(x_path.size(1), n=x_path.size(0))

    risks = np.concatenate(risks).squeeze(-1)
    censorships = np.concatenate(censorships).squeeze(-1)
    event_times = np.concatenate(event_times).squeeze(-1)
    survivals = np.concatenate(survivals, axis=0)

    results = {k: m.avg for k, m in meters.items()}
    results["c_index"] = c_index(risks, censorships, event_times)
    results.update(_time_dependent_metrics(loader, survivals, event_times, censorships))

    dumps = {}
    if dump:
        dumps = {
            "case_ids": np.asarray(case_ids),
            "risk": risks,
            "censorship": censorships,
            "event_time": event_times,
            "survival": survivals,
            "pmf": np.concatenate(pmfs, axis=0),
            **{k: np.concatenate(v) for k, v in diagnostics.items() if v},
        }
    return results, dumps


def c_index(risks, censorships, event_times) -> float:
    """Harrell's concordance index. ``censorships``: 1 = censored, 0 = event."""
    risks = np.asarray(risks).reshape(-1)
    censorships = np.asarray(censorships).reshape(-1)
    event_times = np.asarray(event_times).reshape(-1)
    return float(
        concordance_index_censored(
            (1 - censorships).astype(bool), event_times, risks, tied_tol=1e-8
        )[0]
    )


def build_time_grid(loader, n_cols: Optional[int] = None):
    """Time points matching the discrete survival columns, derived from the bin edges."""
    dataset = getattr(loader, "dataset", None)
    bins = dataset.get_label_bins() if hasattr(dataset, "get_label_bins") else None
    if bins is None:
        return None

    grid = np.asarray(bins, dtype=float)[1:].copy()
    if grid.size == 0:
        return None

    # the rightmost edge is a +inf sentinel; pull it back to just past the horizon
    max_t = float(np.max(dataset.survival_time.numpy())) if hasattr(dataset, "survival_time") else None
    if max_t is not None and np.isfinite(grid[-1]) and grid[-1] > max_t * 10:
        grid[-1] = max_t + 1e-6
    grid[~np.isfinite(grid)] = (max_t + 1e-6) if max_t is not None else 1.0
    grid = np.maximum.accumulate(np.maximum(grid, 1e-8))
    for i in range(1, len(grid)):
        if grid[i] <= grid[i - 1]:
            grid[i] = grid[i - 1] + 1e-6

    if n_cols is not None and len(grid) != n_cols:
        return None
    return grid


def _time_dependent_metrics(loader, survivals, event_times, censorships) -> dict:
    """Antolini's time-dependent C-index, integrated Brier score and integrated NBLL."""
    if EvalSurv is None or survivals.ndim != 2:
        return {}
    grid = build_time_grid(loader, n_cols=survivals.shape[1])
    if grid is None:
        return {}
    try:
        ev = EvalSurv(
            pd.DataFrame(survivals.T, index=grid),
            event_times.astype(float),
            (1 - censorships).astype(bool),
            censor_surv="km",
        )
        return {
            "c_index_td": float(ev.concordance_td("adj_antolini")),
            "ibs": float(ev.integrated_brier_score(grid)),
            "inbll": float(ev.integrated_nbll(grid)),
        }
    except Exception as exc:  # pragma: no cover - metric is optional
        print(f"[warn] skipping IBS/INBLL: {exc}")
        return {}
