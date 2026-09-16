"""End-to-end experiment driver shared by the CLI and the reproduction notebook."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import SlideTextSurvivalDataset, load_anchor_table
from .engine import evaluate, train_one_epoch
from .losses import EvidentialSurvLoss
from .model import SAEFS
from .utils import build_optimizer, build_scheduler, count_params, seed_everything

__all__ = ["SAEFSConfig", "build_datasets", "run_experiment"]


@dataclass
class SAEFSConfig:
    """Everything needed to reproduce one SAEFS run."""

    # ---- data ----
    feat_dirs: List[str] = field(default_factory=list)
    split_dir: str = ""
    train_split: str = "train_tcga.csv"
    test_split: str = "test_cptacucec_unfold.csv"
    anchor_csv: str = ""
    anchor_emb: str = ""
    slide_to_case_json: Optional[str] = None
    target_col: str = "dss_survival_days"
    n_label_bins: int = 4
    num_questions: int = 12
    bag_size: int = -1
    #: reuse the training bin edges for the test split (off by default to match
    #: the protocol the reported numbers were produced with)
    share_label_bins: bool = False

    # ---- model ----
    emb_dim: int = 512
    tau: float = 1.0
    #: evidence-level mix of the WSI-only and the text-guided branch
    alpha: float = 0.5
    #: reliability of the text BBA; 1.0 = no discounting
    r: float = 1.0
    fusion: str = "cautious"
    use_discount: bool = True
    text_drop_prob: float = 0.0

    # ---- optimisation ----
    lr: float = 2e-4
    wd: float = 1e-5
    opt: str = "adamW"
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 0
    max_epochs: int = 20
    accum_steps: int = 1
    #: KL weight ramps as min(1, training_step / annealing_step). 10500 reproduces
    #: the ramp the reported results used; on a 480-case cohort trained for 20
    #: epochs the coefficient reaches 0.91 by the end.
    annealing_step: int = 10500
    n_mc_samples: int = 20

    # ---- runtime ----
    seed: int = 1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 8
    cache_features: bool = False
    cache_device: str = "cpu"
    results_dir: Optional[str] = None
    #: print one line per epoch
    verbose: bool = True
    #: also print a running loss every N batches (0 = off)
    batch_log_every: int = 0

    @property
    def censorship_col(self) -> str:
        return self.target_col.split("_")[0] + "_censorship"


def build_datasets(cfg: SAEFSConfig):
    """Build the train / test datasets and their dataloaders."""
    anchors = load_anchor_table(
        cfg.anchor_csv, cfg.anchor_emb,
        num_questions=cfg.num_questions,
        slide_to_case_json=cfg.slide_to_case_json,
    )

    common = dict(
        anchors=anchors,
        feat_dirs=cfg.feat_dirs,
        survival_time_col=cfg.target_col,
        censorship_col=cfg.censorship_col,
        n_label_bins=cfg.n_label_bins,
        cache=cfg.cache_features,
        cache_device=cfg.cache_device,
    )

    print(f"train split: {cfg.train_split}")
    train_ds = SlideTextSurvivalDataset(
        pd.read_csv(os.path.join(cfg.split_dir, cfg.train_split)),
        bag_size=cfg.bag_size, label_bins=None, **common,
    )
    print(f"  n = {len(train_ds)} cases")

    print(f"test split: {cfg.test_split}")
    test_ds = SlideTextSurvivalDataset(
        pd.read_csv(os.path.join(cfg.split_dir, cfg.test_split)),
        bag_size=-1,
        label_bins=train_ds.get_label_bins() if cfg.share_label_bins else None,
        **common,
    )
    print(f"  n = {len(test_ds)} cases")

    # a cached dataset already holds tensors, so worker processes would only add
    # pickling overhead
    workers = 0 if cfg.cache_features else cfg.num_workers
    loaders = {
        "train": DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=workers),
        "test": DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=workers),
    }
    return {"train": train_ds, "test": test_ds}, loaders


def run_experiment(cfg: SAEFSConfig, datasets=None, loaders=None) -> dict:
    """Train SAEFS on ``cfg.train_split`` and evaluate on ``cfg.test_split``.

    Pass ``datasets`` / ``loaders`` from :func:`build_datasets` to reuse an
    already-cached cohort across several runs (e.g. a hyper-parameter sweep).

    Returns a dict with ``history`` (per-epoch metrics), ``metrics`` (final
    metrics per split), ``dumps`` (per-case predictions and evidential
    diagnostics), ``model`` and ``config``.
    """
    seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    if loaders is None:
        datasets, loaders = build_datasets(cfg)

    model = SAEFS(
        emb_dim=cfg.emb_dim,
        n_classes=cfg.n_label_bins,
        num_questions=cfg.num_questions,
        tau=cfg.tau,
        alpha=cfg.alpha,
        r=cfg.r,
        fusion=cfg.fusion,
        use_discount=cfg.use_discount,
        text_drop_prob=cfg.text_drop_prob,
    ).to(device)
    print(f"SAEFS: {count_params(model):,} trainable parameters "
          f"(alpha={cfg.alpha}, r={cfg.r}, fusion={cfg.fusion})")

    loss_fn = EvidentialSurvLoss(annealing_step=cfg.annealing_step, n_samples=cfg.n_mc_samples)
    optimizer = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.wd, opt=cfg.opt)
    steps_per_epoch = len(loaders["train"]) // cfg.accum_steps
    scheduler = build_scheduler(
        optimizer, cfg.lr_scheduler,
        num_training_steps=steps_per_epoch * cfg.max_epochs,
        num_warmup_steps=steps_per_epoch * cfg.warmup_epochs,
    )

    history = []
    for epoch in range(cfg.max_epochs):
        if cfg.batch_log_every:
            print(f"\n=== epoch {epoch + 1}/{cfg.max_epochs} ===")
        tr_epoch = train_one_epoch(model, loaders["train"], optimizer, scheduler, loss_fn, device,
                                   accum_steps=cfg.accum_steps, log_every=cfg.batch_log_every)
        tr, _ = evaluate(model, loaders["train"], loss_fn, device, dump=False)
        te, _ = evaluate(model, loaders["test"], loss_fn, device, dump=False)
        history.append({"epoch": epoch, "lr": tr_epoch["lr"],
                        **{f"train_{k}": v for k, v in tr.items()},
                        **{f"test_{k}": v for k, v in te.items()}})
        if cfg.verbose:
            print(f"epoch {epoch + 1:>3}/{cfg.max_epochs}  loss {tr['loss']:.3f}  "
                  f"train C-index {tr['c_index']:.4f}  test C-index {te['c_index']:.4f}")

    metrics, dumps = {}, {}
    for split, loader in loaders.items():
        metrics[split], dumps[split] = evaluate(model, loader, loss_fn, device, dump=True)
    print(f"\nFinal: train C-index {metrics['train']['c_index']:.4f} | "
          f"test C-index {metrics['test']['c_index']:.4f}")

    out = {
        "history": pd.DataFrame(history),
        "metrics": metrics,
        "dumps": dumps,
        "model": model,
        "datasets": datasets,
        "config": cfg,
    }

    if cfg.results_dir:
        os.makedirs(cfg.results_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cfg.results_dir, "saefs.pt"))
        out["history"].to_csv(os.path.join(cfg.results_dir, "history.csv"), index=False)
        with open(os.path.join(cfg.results_dir, "config.json"), "w") as f:
            json.dump(asdict(cfg), f, indent=2)
        with open(os.path.join(cfg.results_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        for split, d in dumps.items():
            pd.DataFrame({k: v for k, v in d.items() if v.ndim == 1}).to_csv(
                os.path.join(cfg.results_dir, f"predictions_{split}.csv"), index=False
            )
        print(f"saved to {cfg.results_dir}")

    return out
