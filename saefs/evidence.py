"""Subjective-logic / Dempster-Shafer primitives used by SAEFS.

All functions operate on *Bayesian* basic belief assignments (BBAs), i.e. mass
functions whose only focal elements are the singletons ``{1..K}`` and the frame
``Theta``.  Such a BBA is represented by a pair

    b : (B, K)  mass assigned to each singleton
    u : (B, 1)  mass assigned to Theta (the "I don't know" mass)

with ``b.sum(-1) + u == 1``.
"""

from __future__ import annotations

import torch

__all__ = [
    "dirichlet_to_mass",
    "mass_to_evidence",
    "discount",
    "ds_fuse_bayesian_bba",
    "cautious_fuse_bayesian_bba",
    "betp_from_mass",
]


def dirichlet_to_mass(alpha: torch.Tensor, eps: float = 1e-8):
    """Map Dirichlet parameters to a Bayesian BBA (Subjective Logic).

    ``e = alpha - 1``, ``E = sum_k e_k``, then

        b_k = e_k / (E + K),      u = K / (E + K)

    Args:
        alpha: (B, K) Dirichlet parameters, ``alpha_k > 0``.

    Returns:
        b: (B, K) singleton masses.
        u: (B, 1) mass on Theta.
    """
    _, K = alpha.shape
    e = torch.clamp(alpha - 1.0, min=0.0)
    E = e.sum(dim=1, keepdim=True)
    denom = E + float(K) + eps
    return e / denom, float(K) / denom


def mass_to_evidence(b: torch.Tensor, u: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Inverse of :func:`dirichlet_to_mass`: recover the evidence ``e`` from (b, u)."""
    K = b.shape[1]
    S = float(K) / (u + eps)
    return b * S


def discount(b: torch.Tensor, u: torch.Tensor, r):
    """Shafer's discounting with reliability ``r`` in [0, 1].

        b' = r * b,      u' = u + (1 - r) * (1 - u)

    ``r = 1`` keeps the source untouched; ``r = 0`` turns it into the vacuous
    BBA (``b' = 0``, ``u' = 1``), which is neutral for both fusion rules below.
    """
    if not torch.is_tensor(r):
        r = torch.tensor(r, device=b.device, dtype=b.dtype)
    if r.dim() == 0:
        r = r.view(1, 1).expand(b.size(0), 1)
    return r * b, u + (1.0 - r) * (1.0 - u)


def ds_fuse_bayesian_bba(b1, u1, b2, u2, eps: float = 1e-8):
    """Dempster's rule of combination for two Bayesian BBAs.

    Conflict ``K = sum_{i != j} b1_i b2_j``; the combined masses are

        b12_i = (b1_i b2_i + b1_i u2 + u1 b2_i) / (1 - K)
        u12   = (u1 u2) / (1 - K)

    Returns ``(b12, u12, K)``.
    """
    s1 = b1.sum(dim=1, keepdim=True)
    s2 = b2.sum(dim=1, keepdim=True)
    agree = (b1 * b2).sum(dim=1, keepdim=True)

    K = torch.clamp(s1 * s2 - agree, min=0.0, max=1.0 - 1e-6)
    norm = 1.0 - K + eps

    b12 = (b1 * b2 + b1 * u2 + u1 * b2) / norm
    u12 = (u1 * u2) / norm
    return b12, u12, K


def cautious_fuse_bayesian_bba(b1, u1, b2, u2, eps: float = 1e-12):
    """Denoeux's cautious (conjunctive) rule, specialised to Bayesian BBAs.

    For focal sets ``{i}`` and ``Theta`` the commonality function is
    ``q({i}) = b_i + u`` and ``q(Theta) = u``; the cautious rule takes the
    element-wise minimum of the commonalities, which -- unlike Dempster's rule --
    is idempotent and therefore does not double-count evidence shared by two
    non-independent sources (here: two views of the *same* slide).
    """
    q = torch.minimum(b1 + u1, b2 + u2)
    u = torch.minimum(u1, u2)
    b = torch.clamp(q - u, min=0.0)
    u = torch.clamp(u, min=0.0)

    s = b.sum(dim=1, keepdim=True) + u
    return b / (s + eps), u / (s + eps)


def betp_from_mass(b: torch.Tensor, u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pignistic transform with a uniform base rate: ``BetP_i = b_i + u / K``."""
    K = b.size(1)
    return b + u / float(K) + eps
