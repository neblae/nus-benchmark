"""
schedules.py
------------
Generators for NUS sampling schedules over the indirect dimension(s).

A "schedule" here is a sorted 1D array of integer increment indices that are
*kept* (sampled), drawn from range(0, n_total). For 2D HSQC the indirect
dimension is t1; schedules index which t1 increments you acquire.

All generators share the same signature:

    schedule = generator(n_total, n_keep, seed=..., **kwargs)

so they're interchangeable in the pipeline. Add your own by matching it.

References
----------
Poisson-gap sampling: Hyberts, Takeuchi, Wagner, J. Am. Chem. Soc. 2010,
132, 2145. The sine-weight concentrates sampled points early in t1 where the
signal envelope is strongest, which is the usual reason it outperforms
uniform random for decaying NMR signals.
"""

from __future__ import annotations

import numpy as np


def uniform_random(n_total: int, n_keep: int, seed: int | None = None) -> np.ndarray:
    """Uniform random sampling without replacement.

    The simplest baseline. Useful as the control schedule that any smarter
    scheme should beat.
    """
    _validate(n_total, n_keep)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_total, size=n_keep, replace=False)
    return np.sort(chosen)


def poisson_gap(
    n_total: int,
    n_keep: int,
    seed: int | None = None,
    sine_weight: float = 2.0,
    max_iter: int = 100_000,
) -> np.ndarray:
    """Poisson-gap sampling with sine weighting (Hyberts/Wagner).

    Points are placed by drawing Poisson-distributed gaps whose mean is
    modulated by sin() across the dimension, so gaps are small early (dense
    sampling where signal is strong) and large late.

    Parameters
    ----------
    sine_weight
        Strength of the sine modulation. 0 approaches uniform-random gaps;
        larger values bias more aggressively toward early t1. Typical: 1-3.
    max_iter
        Safety cap on the rejection loop that tunes gap scaling to land on
        exactly n_keep points.

    Notes
    -----
    The number of points produced by a given gap scale is not known in
    closed form, so we binary-search the scale until we hit n_keep. This is
    the standard practical approach.
    """
    _validate(n_total, n_keep)
    rng = np.random.default_rng(seed)

    def build(scale: float) -> np.ndarray:
        points = []
        i = 0.0
        while i < n_total:
            idx = int(round(i))
            if idx < n_total:
                points.append(idx)
            # sine weighting: gap mean varies across the dimension
            frac = i / n_total
            weight = np.sin(np.pi * frac) ** sine_weight if sine_weight else 1.0
            lam = max(scale * weight, 1e-6)
            gap = rng.poisson(lam) + 1
            i += gap
        return np.unique(np.asarray(points, dtype=int))

    # Binary search on the gap scale to produce ~n_keep points.
    lo, hi = 0.01, float(n_total)
    best = None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        pts = build(mid)
        if len(pts) == n_keep:
            best = pts
            break
        if len(pts) > n_keep:
            lo = mid  # too many points -> larger gaps
        else:
            hi = mid  # too few points -> smaller gaps
        best = pts

    # Trim or pad to exactly n_keep (rare edge correction).
    best = _force_count(best, n_total, n_keep, rng)
    return np.sort(best)


def quantile_biased(
    n_total: int,
    n_keep: int,
    seed: int | None = None,
    decay: float = 3.0,
) -> np.ndarray:
    """Exponentially-biased sampling toward early increments.

    Draws indices from an exponential-like density so early t1 points are
    favoured. A simpler, fully-deterministic-density alternative to
    Poisson-gap that's handy for ablations.

    Parameters
    ----------
    decay
        Larger -> more strongly concentrated at the start of the dimension.
    """
    _validate(n_total, n_keep)
    rng = np.random.default_rng(seed)
    x = np.arange(n_total)
    weights = np.exp(-decay * x / n_total)
    weights /= weights.sum()
    chosen = rng.choice(n_total, size=n_keep, replace=False, p=weights)
    return np.sort(chosen)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _validate(n_total: int, n_keep: int) -> None:
    if n_keep > n_total:
        raise ValueError(f"n_keep ({n_keep}) cannot exceed n_total ({n_total})")
    if n_keep < 1:
        raise ValueError("n_keep must be >= 1")
    # The first increment (t1=0) is almost always kept in real NUS; callers
    # who care can enforce it. We don't force it here to keep generators pure.


def _force_count(
    pts: np.ndarray, n_total: int, n_keep: int, rng: np.random.Generator
) -> np.ndarray:
    """Trim or pad a point set to exactly n_keep unique indices."""
    pts = np.unique(pts)
    if len(pts) > n_keep:
        keep = rng.choice(len(pts), size=n_keep, replace=False)
        pts = pts[np.sort(keep)]
    elif len(pts) < n_keep:
        missing = np.setdiff1d(np.arange(n_total), pts)
        need = n_keep - len(pts)
        extra = rng.choice(missing, size=need, replace=False)
        pts = np.concatenate([pts, extra])
    return np.unique(pts)


def sampling_density(schedule: np.ndarray, n_total: int) -> float:
    """Fraction of the full dimension that is sampled (0-1)."""
    return len(schedule) / n_total
