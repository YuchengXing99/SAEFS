#!/usr/bin/env python
"""Train and evaluate SAEFS from the command line.

Example (TCGA-UCEC -> CPTAC-UCEC external validation):

    python scripts/train.py \
        --feat-dirs /data/tcga_cptac_ucec/feats_pt \
        --split-dir splits/survival/TCGA_UCEC_overall_survival_k=0 \
        --train-split train_tcga.csv \
        --test-split  test_cptacucec_unfold.csv \
        --anchor-csv  data/captions/UCEC/wsi_questions_wide.csv \
        --anchor-emb  splits/survival/TCGA_UCEC_overall_survival_k=0/text_embs_conch.npy \
        --target-col dss_survival_days \
        --alpha 0.5 --r 1.0 --lr 2e-4 --max-epochs 20 \
        --results-dir results/ucec
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from saefs import SAEFSConfig, run_experiment  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--feat-dirs", nargs="+", required=True, help="dir(s) of <slide_id>.pt patch features")
    g.add_argument("--split-dir", required=True)
    g.add_argument("--train-split", default="train_tcga.csv")
    g.add_argument("--test-split", default="test_cptacucec_unfold.csv")
    g.add_argument("--anchor-csv", required=True)
    g.add_argument("--anchor-emb", required=True)
    g.add_argument("--slide-to-case-json", default=None,
                   help="optional slide_id -> case_id map for cohorts without parseable barcodes")
    g.add_argument("--target-col", default="dss_survival_days")
    g.add_argument("--n-label-bins", type=int, default=4)
    g.add_argument("--num-questions", type=int, default=12)
    g.add_argument("--bag-size", type=int, default=-1, help="cap patches per case (-1 = keep all)")
    g.add_argument("--share-label-bins", action="store_true",
                   help="reuse the train bin edges on the test split")

    g = p.add_argument_group("model")
    g.add_argument("--emb-dim", type=int, default=512)
    g.add_argument("--alpha", type=float, default=0.5, help="WSI-only vs text-guided evidence mixing")
    g.add_argument("--r", type=float, default=1.0,
                   help="reliability of the text BBA; 1.0 = no discounting")
    g.add_argument("--tau", type=float, default=1.0)
    g.add_argument("--fusion", choices=["cautious", "ds"], default="cautious")
    g.add_argument("--no-discount", action="store_true")
    g.add_argument("--text-drop-prob", type=float, default=0.0)

    g = p.add_argument_group("optimisation")
    g.add_argument("--lr", type=float, default=2e-4)
    g.add_argument("--wd", type=float, default=1e-5)
    g.add_argument("--opt", default="adamW", choices=["adamW", "sgd", "radam"])
    g.add_argument("--lr-scheduler", default="cosine", choices=["cosine", "linear", "constant"])
    g.add_argument("--warmup-epochs", type=int, default=0)
    g.add_argument("--max-epochs", type=int, default=20)
    g.add_argument("--accum-steps", type=int, default=1)
    g.add_argument("--annealing-step", type=int, default=10500)
    g.add_argument("--n-mc-samples", type=int, default=20)

    g = p.add_argument_group("runtime")
    g.add_argument("--seed", type=int, default=1)
    g.add_argument("--device", default=None)
    g.add_argument("--num-workers", type=int, default=8)
    g.add_argument("--cache-features", action="store_true",
                   help="preload every bag into memory once (much faster if the cohort fits)")
    g.add_argument("--cache-device", default="cpu")
    g.add_argument("--results-dir", default=None)
    g.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    kwargs = dict(
        feat_dirs=args.feat_dirs,
        split_dir=args.split_dir,
        train_split=args.train_split,
        test_split=args.test_split,
        anchor_csv=args.anchor_csv,
        anchor_emb=args.anchor_emb,
        slide_to_case_json=args.slide_to_case_json,
        target_col=args.target_col,
        n_label_bins=args.n_label_bins,
        num_questions=args.num_questions,
        bag_size=args.bag_size,
        share_label_bins=args.share_label_bins,
        emb_dim=args.emb_dim,
        alpha=args.alpha,
        r=args.r,
        tau=args.tau,
        fusion=args.fusion,
        use_discount=not args.no_discount,
        text_drop_prob=args.text_drop_prob,
        lr=args.lr,
        wd=args.wd,
        opt=args.opt,
        lr_scheduler=args.lr_scheduler,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        accum_steps=args.accum_steps,
        annealing_step=args.annealing_step,
        n_mc_samples=args.n_mc_samples,
        seed=args.seed,
        num_workers=args.num_workers,
        cache_features=args.cache_features,
        cache_device=args.cache_device,
        results_dir=args.results_dir,
        verbose=not args.quiet,
    )
    if args.device:
        kwargs["device"] = args.device

    run_experiment(SAEFSConfig(**kwargs))


if __name__ == "__main__":
    main()
