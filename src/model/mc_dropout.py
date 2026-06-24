"""Monte Carlo Dropout uncertainty at inference.

N stochastic forward passes with dropout active (model in eval mode, but Dropout
layers forced to train via CxrClassifier.enable_mc_dropout) give a predictive
mean and variance per label. Includes the sanity check required before trusting
MC Dropout: wrong predictions should carry higher uncertainty than correct ones.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def mc_dropout_predict(model, x, n_passes: int = 30):
    """N stochastic sigmoid passes -> (mean, var), each (B, num_classes)."""
    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    samples = [torch.sigmoid(model(x)) for _ in range(n_passes)]
    stacked = torch.stack(samples, 0)
    return stacked.mean(0), stacked.var(0)


def uncertainty_sanity_check(mean_probs, variance, targets, mask, threshold: float = 0.5) -> dict:
    """Wrong predictions should be more uncertain than correct ones."""
    pred = (mean_probs >= threshold).float()
    correct = pred == targets
    msel = mask > 0
    correct_sel = msel & correct
    wrong_sel = msel & (~correct)
    uc = variance[correct_sel].mean().item() if correct_sel.any() else float("nan")
    uw = variance[wrong_sel].mean().item() if wrong_sel.any() else float("nan")
    passes = (uw > uc) if (uw == uw and uc == uc) else False
    return {"unc_correct": uc, "unc_wrong": uw, "passes": bool(passes)}
