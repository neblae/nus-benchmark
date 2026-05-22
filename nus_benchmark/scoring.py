"""
scoring.py
----------
Turn a (reference, reconstruction) pair of integral vectors into a score.

This module IS the label generator for the ML dataset, so correctness here
matters more than anywhere else: a bug here silently corrupts every training
example. Each metric is computed independently and reported, plus one
combined score, so you can choose what your model regresses against later.

All comparisons are done on integrals NORMALIZED to a reference peak, so we
measure *relative* quantification (the stated goal) rather than absolute
scaling differences between reconstructions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class ScoreReport:
    """All metrics for one NUS reconstruction vs the reference."""

    n_peaks: int
    pearson_r: float            # correlation of per-peak volumes (relative fidelity)
    rmsd_norm: float            # RMSD of normalized volumes (lower = better)
    mean_abs_ratio_err: float   # mean |ratio - 1| of per-peak volume ratios
    weak_peak_ratio_err: float  # same, but for the weakest third of peaks
    n_missing: int              # reference peaks with ~zero recovered volume
    combined_score: float       # single scalar in [0, 1], higher = better

    def as_dict(self) -> dict:
        return asdict(self)


def _normalize(volumes: np.ndarray, ref_index: int = 0) -> np.ndarray:
    """Normalize volumes to a chosen reference peak (default: strongest)."""
    denom = volumes[ref_index]
    if denom == 0 or not np.isfinite(denom):
        denom = np.nanmax(np.abs(volumes)) or 1.0
    return volumes / denom


def score_spectrum(
    ref_volumes: np.ndarray,
    nus_volumes: np.ndarray,
    ref_peak_index: int = 0,
    missing_frac: float = 0.05,
    weights: dict | None = None,
) -> ScoreReport:
    """Score one reconstruction's integrals against the reference's.

    Parameters
    ----------
    ref_volumes, nus_volumes : np.ndarray
        Per-peak volumes over the SAME footprints (peaks.integrate_footprints).
        Must be the same length and ordering.
    ref_peak_index
        Which peak to normalize to (default 0 = strongest, from peak picking).
    missing_frac
        A peak counts as "missing" if its normalized NUS volume falls below
        this fraction of its normalized reference volume.
    weights
        Optional weights for the combined score, keys:
        {"corr", "rmsd", "weak", "missing"}. Defaults sum to 1.

    Returns
    -------
    ScoreReport
    """
    ref_volumes = np.asarray(ref_volumes, dtype=float)
    nus_volumes = np.asarray(nus_volumes, dtype=float)
    if ref_volumes.shape != nus_volumes.shape:
        raise ValueError("ref and nus volume vectors must match in shape")
    n = len(ref_volumes)
    if n == 0:
        raise ValueError("no peaks to score")

    ref_n = _normalize(ref_volumes, ref_peak_index)
    nus_n = _normalize(nus_volumes, ref_peak_index)

    # 1) Pearson correlation of per-peak volumes.
    if n >= 2 and np.std(ref_n) > 0 and np.std(nus_n) > 0:
        pearson_r = float(np.corrcoef(ref_n, nus_n)[0, 1])
    else:
        pearson_r = float("nan")

    # 2) RMSD of normalized volumes.
    rmsd_norm = float(np.sqrt(np.mean((ref_n - nus_n) ** 2)))

    # 3) Per-peak ratio error (guard against divide-by-zero).
    safe = np.abs(ref_n) > 1e-12
    ratio = np.ones(n)
    ratio[safe] = nus_n[safe] / ref_n[safe]
    abs_ratio_err = np.abs(ratio - 1.0)
    mean_abs_ratio_err = float(np.mean(abs_ratio_err))

    # 4) Weak-peak ratio error: NUS usually fails the small peaks first, so
    #    track them separately — they're the most informative for ML.
    order = np.argsort(np.abs(ref_n))
    weak_count = max(n // 3, 1)
    weak_idx = order[:weak_count]
    weak_peak_ratio_err = float(np.mean(abs_ratio_err[weak_idx]))

    # 5) Missing peaks.
    n_missing = int(np.sum(np.abs(nus_n) < missing_frac * np.abs(ref_n)))

    # Combined score in [0, 1].
    w = {"corr": 0.4, "rmsd": 0.3, "weak": 0.2, "missing": 0.1}
    if weights:
        w.update(weights)

    corr_term = 0.0 if np.isnan(pearson_r) else (pearson_r + 1) / 2  # -> [0,1]
    rmsd_term = 1.0 / (1.0 + rmsd_norm)                              # -> (0,1]
    weak_term = 1.0 / (1.0 + weak_peak_ratio_err)
    missing_term = 1.0 - (n_missing / n)

    combined = (
        w["corr"] * corr_term
        + w["rmsd"] * rmsd_term
        + w["weak"] * weak_term
        + w["missing"] * missing_term
    )

    return ScoreReport(
        n_peaks=n,
        pearson_r=pearson_r,
        rmsd_norm=rmsd_norm,
        mean_abs_ratio_err=mean_abs_ratio_err,
        weak_peak_ratio_err=weak_peak_ratio_err,
        n_missing=n_missing,
        combined_score=float(combined),
    )
