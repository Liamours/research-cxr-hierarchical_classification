# G. Uncertainty-gated coarse fallback

For each IS-A edge (parent g, child c), we apply a gate at inference that
combines the Monte Carlo Dropout statistics (Section F) with the hierarchy
(Section C). Let tau_u be an uncertainty threshold on the child's predictive
variance and tau_p a probability threshold. The rule for the pair (g, c) is:

- If mu_c >= tau_p and sigma_c^2 <= tau_u: report the child finding c.
- Otherwise, if mu_g >= tau_p: fall back and report the parent finding g,
  suppressing the child prediction.
- Otherwise, abstain for this pair.

Findings with no IS-A parent follow the first rule only: report if
mu_c >= tau_p, abstain otherwise. The parent signal mu_g is read directly
from the same sigmoid head that produces mu_c; no separate head or
aggregation step is needed, because the parent is itself one of the 51
canonical findings.

The gate trades prediction specificity for reliability: a smaller tau_u
causes more child predictions to be suppressed in favor of their parent or
abstention. tau_u is selected on the validation split only; the test split is
not used for threshold selection.
