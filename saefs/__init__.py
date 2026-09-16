"""SAEFS -- Semantic-Anchored Evidential Fusion for Domain-Robust Whole-Slide Survival Analysis."""

from .data import SlideTextSurvivalDataset, load_anchor_table
from .engine import c_index, evaluate, train_one_epoch
from .losses import EvidentialSurvLoss, process_surv
from .model import SAEFS, load_pretrained
from .pipeline import SAEFSConfig, build_datasets, run_experiment

__version__ = "1.0.0"

__all__ = [
    "SAEFS",
    "SAEFSConfig",
    "SlideTextSurvivalDataset",
    "EvidentialSurvLoss",
    "build_datasets",
    "c_index",
    "evaluate",
    "load_anchor_table",
    "load_pretrained",
    "process_surv",
    "run_experiment",
    "train_one_epoch",
]
