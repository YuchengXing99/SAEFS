"""Evidential discrete-time survival loss.

The network emits non-negative *evidence* ``e`` over ``K`` survival bins, which
defines a Dirichlet ``alpha = e + 1`` over the bin-membership simplex.  The loss
has two parts:

* a data term -- for an uncensored case the standard evidential cross-entropy on
  the observed bin; for a censored case the negative log of the (sampled)
  probability of surviving past the censoring bin;
* an annealed KL term that pulls the evidence of the *wrong* bins towards the
  uniform Dirichlet, following Sensoy et al. (2018).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["EvidentialSurvLoss", "process_surv"]


def _kl_to_uniform_dirichlet(alpha: torch.Tensor, c: int) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ) for a single sample."""
    beta = torch.ones((1, c), device=alpha.device, dtype=alpha.dtype)
    S_alpha = torch.sum(alpha)
    S_beta = torch.sum(beta)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha))
    lnB_uni = torch.sum(torch.lgamma(beta)) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    return torch.sum((alpha - beta) * (dg1 - dg0)) + lnB + lnB_uni


def _hazard_term(alpha: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Evidential cross-entropy (Bayes risk of the CE loss) on the observed bin."""
    S = torch.sum(alpha)
    return torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)))


def _survival_term(alpha: torch.Tensor, L: int, start_bin: int) -> torch.Tensor:
    """-log P(bin >= start_bin), estimated with ``L`` Dirichlet samples."""
    p = torch.distributions.Dirichlet(alpha).rsample((L,))          # (L, K)
    tail = p[:, start_bin:].sum(dim=1).clamp_min(1e-12)
    return -torch.log(tail).mean()


def _evidential_surv_loss(evidence, n_bins, global_step, annealing_step, L, censor, y, device):
    alpha = evidence + 1.0
    label = torch.zeros(n_bins, device=device, dtype=alpha.dtype)
    label[y] = 1.0

    if censor == 1:
        # censored at bin y: only know the event happens strictly after bin y
        data_term = _survival_term(alpha, L, y + 1)
    else:
        data_term = _hazard_term(alpha, label)

    annealing_coef = min(1.0, global_step / annealing_step)
    alpha_wrong = (alpha - 1.0) * (1.0 - label) + 1.0
    return data_term + annealing_coef * _kl_to_uniform_dirichlet(alpha_wrong, n_bins)


class EvidentialSurvLoss(nn.Module):
    """Batch wrapper around :func:`_evidential_surv_loss`.

    The KL weight ramps as ``min(1, global_step / annealing_step)``.
    ``global_step`` counts **training** calls only, so the schedule does not
    depend on how often you evaluate -- put the module in eval mode
    (``loss_fn.eval()``) around validation, exactly as you would a model.
    ``annealing_step`` is therefore in optimiser steps and its effect scales with
    the number of cases per epoch and the number of epochs.
    """

    def __init__(self, annealing_step: int = 10500, n_samples: int = 20, reduction: str = "mean"):
        super().__init__()
        self.annealing_step = annealing_step
        self.L = n_samples
        self.reduction = reduction
        self.global_step = 1

    def forward(self, logits, times, censorships):
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        # the model already emits non-negative evidence; keep the guard for safety
        evidence = logits if torch.all(logits >= 0) else torch.nn.functional.softplus(logits)

        times = times.long().view(-1)
        censorships = censorships.long().view(-1)
        n_bins = logits.shape[1]

        losses = []
        for i in range(logits.shape[0]):
            y = int(torch.clamp(times[i], min=0, max=n_bins - 1).item())
            losses.append(
                _evidential_surv_loss(
                    evidence=evidence[i],
                    n_bins=n_bins,
                    global_step=self.global_step,
                    annealing_step=self.annealing_step,
                    L=self.L,
                    censor=int(censorships[i].item()),
                    y=y,
                    device=logits.device,
                )
            )

        losses = torch.stack(losses)
        loss = losses.sum() if self.reduction == "sum" else losses.mean()
        if self.training:
            self.global_step += 1
        return {"loss": loss}


def process_surv(evidence, label, censorship, loss_fn=None):
    """Turn fused evidence into survival predictions (+ loss, if a label is given).

    The Dirichlet mean ``p = alpha / sum(alpha)`` is read as the PMF over the
    survival bins; ``S(t_k) = sum_{j > k} p_j`` is the discrete survival
    function and the scalar risk is ``-sum_k S(t_k)``.
    """
    results_dict = {"logits": evidence}
    log_dict = {}

    alpha = evidence + 1.0
    p = alpha / alpha.sum(dim=1, keepdim=True)
    survival = torch.flip(torch.cumsum(torch.flip(p, dims=[1]), dim=1), dims=[1]) - p
    risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
    results_dict.update({"risk": risk, "survival": survival, "pmf": p})

    if loss_fn is not None and label is not None:
        surv_loss_dict = loss_fn(logits=evidence, times=label, censorships=censorship)
        log_dict["surv_loss"] = surv_loss_dict["loss"].item()
        log_dict.update({k: v.item() for k, v in surv_loss_dict.items() if torch.is_tensor(v)})
        results_dict["loss"] = surv_loss_dict["loss"]

    return results_dict, log_dict
