"""
undersample.py
--------------
Retrospective undersampling: take a fully-sampled FID and zero out (or drop)
the t1 increments that a schedule does not keep.

This is the heart of why retrospective undersampling gives clean ML labels:
the kept points are *bit-identical* to the full experiment, so the only
variable between training examples is the schedule itself.

Two output conventions are supported:
- "masked": same shape as input, unsampled t1 rows set to zero. This is what
  most reconstruction algorithms (IST, MDD) expect as input — a full grid
  with gaps.
- "compact": only the kept rows, plus the index array. Smaller, but you must
  carry the schedule to know where the points belong.
"""

from __future__ import annotations

import numpy as np


def apply_schedule(
    fid: np.ndarray,
    schedule: np.ndarray,
    mode: str = "masked",
) -> np.ndarray:
    """Apply a t1 sampling schedule to a 2D FID.

    Parameters
    ----------
    fid : np.ndarray, shape (n1, n2)
        Fully-sampled complex FID, t1 along axis 0.
    schedule : np.ndarray
        Sorted integer indices of kept t1 increments (from schedules.py).
    mode : {"masked", "compact"}
        "masked" returns shape (n1, n2) with unsampled rows zeroed.
        "compact" returns shape (len(schedule), n2).

    Returns
    -------
    np.ndarray
    """
    if fid.ndim != 2:
        raise ValueError("apply_schedule expects a 2D FID (n1, n2)")
    n1 = fid.shape[0]
    schedule = np.asarray(schedule, dtype=int)
    if schedule.max(initial=-1) >= n1 or schedule.min(initial=0) < 0:
        raise ValueError("schedule indices out of range for this FID")

    if mode == "masked":
        out = np.zeros_like(fid)
        out[schedule, :] = fid[schedule, :]
        return out
    elif mode == "compact":
        return fid[schedule, :].copy()
    else:
        raise ValueError(f"unknown mode {mode!r}; use 'masked' or 'compact'")


def sampling_mask(schedule: np.ndarray, n1: int) -> np.ndarray:
    """Boolean mask over t1 (True where sampled). Handy for plotting."""
    mask = np.zeros(n1, dtype=bool)
    mask[np.asarray(schedule, dtype=int)] = True
    return mask
