"""Monte Carlo Dropout uncertainty at inference.

N stochastic forward passes with dropout active (model in eval mode, but Dropout
layers forced to train via CxrClassifier.enable_mc_dropout) give a predictive
mean plus a decomposed uncertainty per label (Kwon et al. 2020; Gal & Ghahramani
2016). Includes the sanity check required before trusting MC Dropout: wrong
predictions should carry higher uncertainty than correct ones.

Decomposition (law of total variance, exact for this Bernoulli-mixture model):
  epistemic = Var_t(p_t)          -- disagreement across passes: model uncertainty
  aleatoric = E_t[p_t(1-p_t)]     -- average per-pass Bernoulli variance: data/noise uncertainty
  total     = epistemic + aleatoric
"""

from __future__ import annotations

import torch


@torch.no_grad()
def mc_dropout_predict(model, x, n_passes: int = 30):
    """N stochastic sigmoid passes -> (mean, epistemic, aleatoric), each (B, num_classes).
    total predictive variance = epistemic + aleatoric."""
    model.eval()
    if hasattr(model, "enable_mc_dropout"):
        model.enable_mc_dropout()
    samples = torch.stack([torch.sigmoid(model(x)) for _ in range(n_passes)], 0)  # (T, B, C)
    mean = samples.mean(0)
    epistemic = samples.var(0, unbiased=False)          # population variance -> exact identity below
    aleatoric = (samples * (1 - samples)).mean(0)
    return mean, epistemic, aleatoric


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


def _demo():
    """Self-check: epistemic + aleatoric must equal the exact total variance of
    the underlying Bernoulli-mixture (not just the raw across-pass variance of
    the sigmoid outputs, which is epistemic alone)."""
    torch.manual_seed(0)
    T, B, C = 500, 4, 6
    p = torch.rand(T, B, C).clamp(0.05, 0.95)  # T stochastic pass outputs
    epistemic = p.var(0, unbiased=False)
    aleatoric = (p * (1 - p)).mean(0)
    total = epistemic + aleatoric

    # Monte Carlo estimate of the true total variance: draw a Bernoulli per pass,
    # per label, then take variance over passes -- should match `total` closely.
    y = torch.bernoulli(p)
    empirical_total = y.var(0, unbiased=False)
    assert (total - empirical_total).abs().mean() < 0.01, "decomposition identity broke"
    assert (epistemic >= 0).all() and (aleatoric >= 0).all()
    print("OK: epistemic + aleatoric matches empirical total variance "
          f"(mean abs diff={(total - empirical_total).abs().mean():.4f})")


if __name__ == "__main__":
    _demo()
