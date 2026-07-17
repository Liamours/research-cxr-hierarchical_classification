"""Multi-label iterative stratification (Sechidis, Tsoumakas & Vlahavas 2011).

Splits samples into k subsets with target proportions while keeping each label's
positive rate close to that proportion in every subset. Rare labels are placed
first, so a label with only a handful of positives is spread across subsets
rather than landing entirely in one -- which is exactly what a naive random or
single-column stratified split fails to do for long-tailed multi-label data.

Deterministic given `seed` (only tie-breaks use the RNG).
"""

from __future__ import annotations

import numpy as np


def iterative_stratification(Y: np.ndarray, proportions, seed: int = 42) -> np.ndarray:
    """Assign each of the n rows of binary label matrix Y (n, L) to one of
    len(proportions) subsets. Returns an int array (n,) of subset indices."""
    Y = np.asarray(Y, dtype=np.int64)
    n, L = Y.shape
    k = len(proportions)
    rng = np.random.default_rng(seed)
    props = np.asarray(proportions, dtype=np.float64)

    c = props * n                                   # desired remaining count per subset
    cij = np.outer(props, Y.sum(axis=0))            # desired remaining per (subset, label)

    assigned = np.full(n, -1, dtype=np.int64)
    remaining = np.ones(n, dtype=bool)

    while remaining.any():
        rem_idx = np.flatnonzero(remaining)
        label_counts = Y[rem_idx].sum(axis=0)       # positives left per label
        pos = np.flatnonzero(label_counts > 0)
        if pos.size == 0:                           # only all-negative rows left
            for i in rem_idx:
                j = int(np.argmax(c))
                assigned[i] = j
                c[j] -= 1
            break

        li = int(pos[np.argmin(label_counts[pos])])  # rarest remaining label
        examples = rem_idx[Y[rem_idx, li] == 1]
        for i in examples:
            col = cij[:, li]
            cand = np.flatnonzero(col == col.max())          # max desired for this label
            if cand.size > 1:
                c_cand = c[cand]
                cand = cand[c_cand == c_cand.max()]          # tie-break: max overall desired
            j = int(cand[0]) if cand.size == 1 else int(rng.choice(cand))
            assigned[i] = j
            cij[j, Y[i] == 1] -= 1
            c[j] -= 1
            remaining[i] = False

    return assigned


def split_labels(Y, seed=42, val=0.1, test=0.1):
    """Convenience: iterative-stratify into train/val/test string labels."""
    train = 1.0 - val - test
    idx = iterative_stratification(Y, [train, val, test], seed=seed)
    return np.array(["train", "val", "test"])[idx]


def _demo():
    """Self-check: a rare label (5 positives) must appear in >1 subset, and
    common-label proportions must land near target."""
    rng = np.random.default_rng(0)
    n, L = 4000, 6
    Y = (rng.random((n, L)) < np.array([0.4, 0.2, 0.1, 0.05, 0.01, 0.00125])).astype(int)
    # force a very rare label with exactly 5 positives
    Y[:, 5] = 0
    Y[rng.choice(n, 5, replace=False), 5] = 1
    s = split_labels(Y, seed=42)
    for name, frac in [("train", 0.8), ("val", 0.1), ("test", 0.1)]:
        share = (s == name).mean()
        assert abs(share - frac) < 0.03, (name, share)
        # common label proportion within tolerance
        p0 = Y[s == name, 0].sum() / Y[:, 0].sum()
        assert abs(p0 - frac) < 0.05, (name, p0)
    rare_spread = len(set(s[Y[:, 5] == 1]))
    assert rare_spread >= 2, f"rare label landed in only {rare_spread} subset"
    print(f"OK: proportions near target; 5-positive rare label spread across {rare_spread} subsets")


if __name__ == "__main__":
    _demo()
