"""SAEFS -- Semantic-Anchored Evidential Fusion for whole-slide survival analysis.

The slide is represented by a bag of patch embeddings ``x_path`` of shape
``(B, N, D)``; the semantic anchors are ``Q`` VQA answer embeddings ``x_text`` of
shape ``(B, Q, D)``, lying in the *same* vision-language space (CONCH).

Three evidence sources are produced and combined:

    (A) WSI-only        ABMIL attention pooling          -> e_wsi
    (B) Text-guided WSI anchor-conditioned patch pooling -> e_tg
    (C) Text-only       mean of the anchors              -> e_text

(A) and (B) are mixed at the evidence level with a fixed weight ``alpha``,
yielding the pathology BBA.  The text BBA is then discounted by a fixed
reliability ``r`` and fused with the pathology BBA using the cautious rule
(default) or Dempster's rule.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .evidence import (
    cautious_fuse_bayesian_bba,
    dirichlet_to_mass,
    discount,
    ds_fuse_bayesian_bba,
    mass_to_evidence,
)
from .losses import process_surv

__all__ = ["SAEFS", "load_pretrained"]


class SAEFS(nn.Module):
    """Semantic-Anchored Evidential Fusion.

    Args:
        emb_dim: dimensionality of the shared vision-language space (CONCH: 512).
        n_classes: number of discrete survival bins.
        num_questions: number of semantic anchors ``Q`` per case.
        tau: temperature of the anchor -> patch soft selector.
        alpha: evidence-level mixing weight between the WSI-only and the
            text-guided branch, ``e_path = (1 - alpha) * e_wsi + alpha * e_tg``.
            ``alpha = 0`` is plain ABMIL, ``alpha = 1`` is purely anchor-driven.
        r: reliability discount applied to the *text-only* BBA before fusion.
            ``r = 1`` trusts the anchors fully, ``r = 0`` discounts them to the
            vacuous BBA (i.e. the text branch abstains).
        fusion: ``"cautious"`` (idempotent, default) or ``"ds"`` (Dempster).
        text_drop_prob: probability of zeroing the whole text modality during
            training (modality dropout); ``0`` disables it.
    """

    def __init__(
        self,
        emb_dim: int = 512,
        n_classes: int = 4,
        num_questions: int = 12,
        tau: float = 1.0,
        alpha: float = 0.5,
        r: float = 1.0,
        fusion: str = "cautious",
        use_discount: bool = True,
        text_drop_prob: float = 0.0,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_classes = n_classes
        self.num_questions = num_questions
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.r = float(r)
        self.fusion = fusion.lower()
        self.use_discount = bool(use_discount)
        self.text_drop_prob = float(text_drop_prob)

        if self.fusion not in ("cautious", "ds"):
            raise ValueError(f"fusion must be 'cautious' or 'ds', got {fusion!r}")

        # (A) WSI-only branch: minimal ABMIL pooling + evidence head
        self.abmil_V = nn.Linear(emb_dim, emb_dim)
        self.abmil_w = nn.Linear(emb_dim, 1, bias=False)
        self.wsi_head = nn.Linear(emb_dim, n_classes)

        # (B) text-guided WSI branch and (C) text-only branch
        self.path_head = nn.Linear(emb_dim, n_classes)
        self.text_head = nn.Linear(emb_dim, n_classes)

    def forward_no_loss(self, x_path: torch.Tensor, x_text: torch.Tensor) -> dict:
        """Run the three branches and the evidential fusion.

        Args:
            x_path: (B, N, D) patch embeddings.
            x_text: (B, Q, D) semantic anchor embeddings.

        Returns:
            dict with the fused ``evidence`` plus per-branch diagnostics
            (uncertainties, conflict, attention maps).
        """
        B = x_path.size(0)
        Q = x_text.size(1)
        if Q != self.num_questions:
            raise ValueError(f"Expected Q={self.num_questions} anchors, got {Q}")

        if self.training and self.text_drop_prob > 0:
            keep = (torch.rand(B, 1, 1, device=x_text.device) > self.text_drop_prob).float()
            x_text = x_text * keep

        # ---------------- (A) WSI-only branch ----------------
        a_wsi = torch.softmax(self.abmil_w(torch.tanh(self.abmil_V(x_path))), dim=1)  # (B,N,1)
        h_wsi = torch.sum(a_wsi * x_path, dim=1)                                      # (B,D)
        e_wsi = F.softplus(self.wsi_head(h_wsi))
        b_wsi, u_wsi = dirichlet_to_mass(e_wsi + 1.0)

        # ---------------- (B) text-guided WSI branch ----------------
        # cosine similarity between each anchor and each patch, softmax over patches
        sim = torch.einsum(
            "bqd,bnd->bqn", F.normalize(x_text, dim=-1), F.normalize(x_path, dim=-1)
        ) / self.tau                                                                  # (B,Q,N)
        A = torch.softmax(sim, dim=-1)                                                # (B,Q,N)
        h_tg = torch.einsum("bqn,bnd->bqd", A, x_path).mean(dim=1)                    # (B,D)
        e_tg = F.softplus(self.path_head(h_tg))
        b_tg, u_tg = dirichlet_to_mass(e_tg + 1.0)

        # ---------------- (C) text-only branch ----------------
        h_text = x_text.mean(dim=1)                                                   # (B,D)
        e_text = F.softplus(self.text_head(h_text))
        b_text, u_text = dirichlet_to_mass(e_text + 1.0)

        # ---------------- evidence-level mixing of (A) and (B) ----------------
        e_path = (1.0 - self.alpha) * e_wsi + self.alpha * e_tg
        b_path, u_path = dirichlet_to_mass(e_path + 1.0)

        # pre-fusion conflict between the pathology and the text BBA
        K_pre = ((1.0 - u_path) * (1.0 - u_text)
                 - (b_path * b_text).sum(dim=1, keepdim=True)).clamp(0.0, 1.0)
        cos_ht = F.cosine_similarity(h_tg, h_text, dim=-1).unsqueeze(1)

        # ---------------- reliability discounting ----------------
        # keep the undiscounted text mass around: with r = 0 the discounted u_text
        # is identically 1 and carries no information for diagnostics/plots.
        u_text_raw, b_text_raw = u_text, b_text
        r_text = torch.full_like(u_text, self.r)
        if self.use_discount:
            b_path, u_path = discount(b_path, u_path, torch.ones_like(u_path))
            b_text, u_text = discount(b_text, u_text, r_text)

        # ---------------- fusion ----------------
        if self.fusion == "cautious":
            b_fused, u_fused = cautious_fuse_bayesian_bba(b_path, u_path, b_text, u_text)
            K_conflict = K_pre
        else:
            b_fused, u_fused, K_conflict = ds_fuse_bayesian_bba(b_path, u_path, b_text, u_text)

        return {
            "evidence": mass_to_evidence(b_fused, u_fused),
            # attention maps
            "A_coattn": A,
            "A_wsi": a_wsi,
            # uncertainties (mass on Theta) per branch
            "u_wsi": u_wsi,
            "u_tg": u_tg,
            "u_path": u_path,
            "u_text": u_text,
            "u_text_raw": u_text_raw,
            "u_fused": u_fused,
            # conflict / agreement diagnostics
            "K_conflict": K_conflict,
            "K_pre": K_pre,
            "cos_ht": cos_ht,
            "r_text": r_text,
            "alpha_mix": self.alpha,
            # raw evidences
            "e_wsi": e_wsi,
            "e_tg": e_tg,
            "e_text": e_text,
            "e_path": e_path,
            # belief masses (handy for plotting)
            "b_wsi": b_wsi,
            "b_tg": b_tg,
            "b_text": b_text,
            "b_text_raw": b_text_raw,
            "b_fused": b_fused,
        }

    def forward(self, x_path, x_text, label=None, censorship=None, loss_fn=None, **kwargs):
        out = self.forward_no_loss(x_path, x_text)
        results_dict, log_dict = process_surv(out["evidence"], label, censorship, loss_fn)
        results_dict.update(out)
        return results_dict, log_dict


def load_pretrained(path: str, map_location="cpu", **kwargs) -> SAEFS:
    """Build a :class:`SAEFS` and load a released checkpoint into it.

    Checkpoints trained with the original research code also carry two gating
    MLPs (``alpha_gate``, ``gate``) whose outputs that code computed and then
    overwrote with the fixed ``alpha`` / ``r``. They are inert, so they are
    dropped here; the remaining weights fully determine the predictions.

    Args:
        path: ``.pt`` holding a ``state_dict``.
        **kwargs: forwarded to :class:`SAEFS` -- pass the same ``alpha``, ``r``,
            ``tau``, ``fusion`` and ``n_classes`` the checkpoint was trained with.
    """
    state = torch.load(path, map_location=map_location, weights_only=False)
    state = state.get("model", state)

    model = SAEFS(**kwargs)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"checkpoint is missing required weights: {sorted(missing)}")
    stray = [k for k in unexpected if not k.startswith(("alpha_gate.", "gate."))]
    if stray:
        raise RuntimeError(f"unrecognised weights in checkpoint: {sorted(stray)}")
    return model.eval()
