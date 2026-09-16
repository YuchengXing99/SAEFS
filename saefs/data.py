"""Dataset that pairs a bag of patch features with its semantic anchors.

Expected inputs
---------------
* **Split CSV** (one row per slide) with at least
  ``case_id``, ``slide_id``, ``<prefix>_survival_days``, ``<prefix>_censorship``
  where ``censorship == 1`` means *censored*.
* **Feature directory** with one ``<slide_id>.pt`` (or ``.h5``) per slide, each
  holding an ``(n_patches, D)`` float array of patch embeddings.
* **Anchor CSV** (one row per slide) with ``slide_id`` and ``Q1..QN`` columns,
  plus the matching ``.npy`` of CONCH text embeddings, shape
  ``(n_unique_cases, Q, D)``.  See ``scripts/encode_text.py``.

See ``docs/data_preparation.md`` for the full contract.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional, Sequence

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = [
    "SlideTextSurvivalDataset",
    "default_case_id_from_slide_id",
    "load_anchor_table",
    "compute_discretization",
]


def default_case_id_from_slide_id(slide_ids: pd.Series, mapping: Optional[dict] = None) -> pd.Series:
    """Derive ``case_id`` from ``slide_id`` for the cohorts used in the paper.

    TCGA barcodes -> first 12 characters (``TCGA-2E-A9G8``);
    CPTAC barcodes -> first 9 characters (``C3L-00006``);
    anything else -> looked up in ``mapping`` (e.g. NLST), otherwise unchanged.
    """
    sid = slide_ids.astype(str)
    fallback = sid.map(mapping) if mapping else sid
    return pd.Series(
        np.where(
            sid.str.startswith("TCGA"),
            sid.str[:12],
            np.where(sid.str.startswith("C3"), sid.str[:9], fallback),
        ),
        index=sid.index,
    )


def load_anchor_table(
    anchor_csv: str,
    anchor_emb_npy: str,
    num_questions: Optional[int] = None,
    case_id_fn: Callable[[pd.Series], pd.Series] = default_case_id_from_slide_id,
    slide_to_case_json: Optional[str] = None,
) -> pd.DataFrame:
    """Load the anchor CSV + its pre-computed embeddings, indexed by ``case_id``.

    Rows are de-duplicated by ``case_id`` keeping the **first** occurrence, which
    is the order the embedding ``.npy`` was written in -- do not reorder the CSV
    without re-running ``scripts/encode_text.py``.
    """
    df = pd.read_csv(anchor_csv, engine="python")
    if "slide_id" not in df.columns:
        raise KeyError(f"'slide_id' column missing from {anchor_csv}")

    if "case_id" not in df.columns:
        mapping = None
        if slide_to_case_json:
            with open(slide_to_case_json) as f:
                mapping = json.load(f)
        df["case_id"] = case_id_fn(df["slide_id"], mapping) if slide_to_case_json \
            else case_id_fn(df["slide_id"])
    df["case_id"] = df["case_id"].astype(str)

    df = df[~df["case_id"].duplicated()].reset_index(drop=True)

    emb = np.load(anchor_emb_npy)
    if emb.shape[0] != len(df):
        raise ValueError(
            f"Anchor embedding / CSV mismatch: {anchor_emb_npy} has {emb.shape[0]} rows "
            f"but {anchor_csv} has {len(df)} unique cases. Re-run scripts/encode_text.py."
        )
    if num_questions is not None:
        if num_questions > emb.shape[1]:
            raise ValueError(f"Requested Q={num_questions} but embeddings only have {emb.shape[1]}")
        emb = emb[:, :num_questions]

    df["anchor_emb"] = list(emb)
    return df.set_index("case_id")


def compute_discretization(
    df: pd.DataFrame,
    survival_time_col: str,
    censorship_col: str,
    n_label_bins: int = 4,
    label_bins=None,
    sample_col: str = "case_id",
):
    """Quantile-bin event times into ``n_label_bins`` discrete bins.

    Bin edges are estimated on the **uncensored** cases, as is standard for
    discrete-time survival models.

    Returns:
        (pd.Series indexed by ``sample_col`` holding the bin index, bin edges).
    """
    df = df[~df[sample_col].duplicated()]

    if label_bins is not None:
        if len(label_bins) != n_label_bins + 1:
            raise ValueError(f"Expected {n_label_bins + 1} bin edges, got {len(label_bins)}")
        q_bins = label_bins
    else:
        uncensored = df[df[censorship_col] == 0]
        _, q_bins = pd.qcut(uncensored[survival_time_col], q=n_label_bins, retbins=True, labels=False)
        q_bins[-1] = 1e6      # right edge -> +inf
        q_bins[0] = -1e-6     # left edge -> 0

    disc_labels, q_bins = pd.cut(
        df[survival_time_col], bins=q_bins, retbins=True, labels=False, include_lowest=True
    )
    disc_labels = pd.Series(disc_labels.to_numpy(), index=df[sample_col].to_numpy(), name="disc_label")
    return disc_labels, q_bins


def _read_features(path: str) -> np.ndarray:
    if path.endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as f:
            feats = f["features"][:]
    else:
        feats = torch.load(path, weights_only=False)
    feats = np.asarray(feats) if not torch.is_tensor(feats) else feats.numpy()
    if feats.ndim > 2:
        if feats.shape[0] != 1:
            raise ValueError(f"{path}: expected (n_patches, D) or (1, n_patches, D), got {feats.shape}")
        feats = np.squeeze(feats, axis=0)
    return feats


class SlideTextSurvivalDataset(Dataset):
    """One item per *case*: all its patch features + its ``Q`` semantic anchors."""

    def __init__(
        self,
        df_histo: pd.DataFrame,
        anchors: pd.DataFrame,
        feat_dirs: Sequence[str],
        survival_time_col: str = "dss_survival_days",
        censorship_col: str = "dss_censorship",
        n_label_bins: int = 4,
        label_bins=None,
        bag_size: int = -1,
        sample_col: str = "case_id",
        slide_col: str = "slide_id",
        cache: bool = False,
        cache_device: str = "cpu",
    ):
        self.sample_col = sample_col
        self.slide_col = slide_col
        self.survival_time_col = survival_time_col
        self.censorship_col = censorship_col
        self.bag_size = bag_size
        self.n_label_bins = n_label_bins

        df = df_histo.copy()
        df[sample_col] = df[sample_col].astype(str)
        df[slide_col] = df[slide_col].astype(str)

        # --- drop rows with missing labels -------------------------------------
        n0 = len(df)
        df = df[df[censorship_col].notna() & df[survival_time_col].notna()]
        if len(df) < n0:
            print(f"  dropped {n0 - len(df)} slides with missing survival label")
        if not df[censorship_col].isin([0, 1]).all():
            raise ValueError(f"{censorship_col} must be binary 0/1")
        if (df[survival_time_col] < 0).any():
            raise ValueError(f"{survival_time_col} must be non-negative")

        # --- resolve feature paths ---------------------------------------------
        feats = self._scan_feature_dirs(feat_dirs)
        missing = set(df[slide_col]) - set(feats)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} slide(s) in the split have no feature file, e.g. "
                f"{sorted(missing)[:5]}"
            )
        df["fpath"] = df[slide_col].map(feats)

        # --- keep only cases that also have semantic anchors --------------------
        self.sample_ids = sorted(set(df[sample_col]) & set(anchors.index))
        n_dropped = df[sample_col].nunique() - len(self.sample_ids)
        if n_dropped:
            print(f"  dropped {n_dropped} case(s) without semantic anchors")
        if not self.sample_ids:
            raise RuntimeError("No case is present in both the split CSV and the anchor table.")
        df = df[df[sample_col].isin(self.sample_ids)].reset_index(drop=True)

        # --- discretize survival times -----------------------------------------
        disc, self.label_bins = compute_discretization(
            df, survival_time_col, censorship_col, n_label_bins, label_bins, sample_col
        )
        df["disc_label"] = df[sample_col].map(disc)

        self.data_df = df.set_index(sample_col, drop=False)
        self.anchors = anchors.loc[self.sample_ids]

        labels = self.data_df.loc[self.sample_ids, [survival_time_col, censorship_col, "disc_label"]]
        labels = labels[~labels.index.duplicated()]
        self.survival_time = torch.tensor(labels[survival_time_col].to_numpy(), dtype=torch.float32)
        self.censorship = torch.tensor(labels[censorship_col].to_numpy(), dtype=torch.float32)
        self.disc_label = torch.tensor(labels["disc_label"].to_numpy(), dtype=torch.float32)

        # Reading the ``.pt`` bags dominates the epoch time; for cohorts that fit in
        # memory (TCGA-UCEC + CPTAC-UCEC is ~8 GB) caching them once makes an epoch
        # roughly two orders of magnitude faster.
        self._cache = None
        if cache:
            self._cache = [self._load_bag(cid).to(cache_device) for cid in self.sample_ids]
            n_bytes = sum(t.element_size() * t.nelement() for t in self._cache)
            print(f"  cached {len(self._cache)} bags on {cache_device} ({n_bytes / 1e9:.2f} GB)")

    @staticmethod
    def _scan_feature_dirs(feat_dirs: Sequence[str]) -> dict:
        feats = {}
        for d in feat_dirs:
            if not os.path.isdir(d):
                raise NotADirectoryError(f"Feature directory not found: {d}")
            for fname in os.listdir(d):
                stem, ext = os.path.splitext(fname)
                if ext.lower() in (".pt", ".h5", ".hdf5"):
                    if stem in feats:
                        raise ValueError(f"Duplicate feature file for slide {stem}")
                    feats[stem] = os.path.join(d, fname)
        return feats

    def __len__(self) -> int:
        return len(self.sample_ids)

    def get_label_bins(self):
        return self.label_bins

    def bag_sizes(self) -> np.ndarray:
        """Number of patches per case (reads from the cache when there is one)."""
        if self._cache is not None:
            return np.array([len(t) for t in self._cache])
        return np.array([len(self._load_bag(cid)) for cid in self.sample_ids])

    def _load_bag(self, case_id: str) -> torch.Tensor:
        rows = self.data_df.loc[[case_id]]
        feats = np.concatenate([_read_features(p) for p in rows["fpath"]], axis=0)
        return torch.from_numpy(feats).float()

    def __getitem__(self, idx: int) -> dict:
        case_id = self.sample_ids[idx]
        feats = self._cache[idx] if self._cache is not None else self._load_bag(case_id)

        out = {
            "img": self._maybe_subsample(feats),
            "text": torch.as_tensor(np.asarray(self.anchors.loc[case_id, "anchor_emb"])).float(),
            "survival_time": self.survival_time[idx].view(1),
            "censorship": self.censorship[idx].view(1),
            "label": self.disc_label[idx].view(1),
            "case_id": case_id,
        }
        return out

    def _maybe_subsample(self, feats: torch.Tensor) -> torch.Tensor:
        """Optionally cap the bag at ``bag_size`` patches (``-1`` keeps them all)."""
        if self.bag_size <= 0 or feats.shape[0] <= self.bag_size:
            return feats
        idx = np.random.choice(feats.shape[0], self.bag_size, replace=False)
        return feats[idx]
