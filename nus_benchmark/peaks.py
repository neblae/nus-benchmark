"""
peaks.py
--------
Peak picking and 2D volume integration.

Design rule for quantification (important): peaks are picked ONCE on the
fully-sampled reference spectrum, producing a set of fixed "footprints".
Every reconstructed NUS spectrum is then integrated over those SAME
footprints. We never re-pick per spectrum — if a peak weakens or vanishes
under a schedule, we want that to show up as a smaller integral at a fixed
location, not as a missing entry. This keeps integrals directly comparable
across schedules, which is exactly what clean ML labels require.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from scipy.ndimage import maximum_filter, label
    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False


@dataclass
class Footprint:
    """A fixed integration region around one peak, in spectrum indices."""

    row: int
    col: int
    half_r: int          # half-height of the box in points (axis 0)
    half_c: int          # half-width of the box in points (axis 1)
    peak_id: int = -1

    def slices(self, shape: tuple[int, int]) -> tuple[slice, slice]:
        r0 = max(self.row - self.half_r, 0)
        r1 = min(self.row + self.half_r + 1, shape[0])
        c0 = max(self.col - self.half_c, 0)
        c1 = min(self.col + self.half_c + 1, shape[1])
        return slice(r0, r1), slice(c0, c1)


def estimate_noise(spec: np.ndarray, corner_frac: float = 0.1) -> float:
    """Estimate noise std from the corners of the spectrum (signal-free)."""
    n1, n2 = spec.shape
    r = max(int(n1 * corner_frac), 1)
    c = max(int(n2 * corner_frac), 1)
    corners = np.concatenate([
        spec[:r, :c].ravel(),
        spec[:r, -c:].ravel(),
        spec[-r:, :c].ravel(),
        spec[-r:, -c:].ravel(),
    ])
    return float(np.std(corners))


def pick_reference_peaks(
    spec: np.ndarray,
    snr_threshold: float = 10.0,
    min_distance: int = 3,
    footprint_half: tuple[int, int] = (4, 4),
) -> list[Footprint]:
    """Pick peaks on the REFERENCE spectrum and return fixed footprints.

    Parameters
    ----------
    spec : np.ndarray
        Real, fully-sampled reference spectrum.
    snr_threshold
        Keep local maxima above noise_std * snr_threshold.
    min_distance
        Minimum separation (points) between accepted maxima; suppresses
        picking multiple points on one peak.
    footprint_half
        (half_rows, half_cols) of the integration box assigned to each peak.
        For overlapping peaks you'd shrink these or move to lineshape fitting.

    Returns
    -------
    list[Footprint]
    """
    noise = estimate_noise(spec)
    threshold = noise * snr_threshold

    if _HAVE_SCIPY:
        size = 2 * min_distance + 1
        local_max = (spec == maximum_filter(spec, size=size)) & (spec > threshold)
        coords = np.argwhere(local_max)
    else:  # pragma: no cover - fallback without scipy
        coords = _local_maxima_numpy(spec, threshold, min_distance)

    # Sort by intensity (strongest first) so peak_id 0 is the base peak.
    intensities = spec[coords[:, 0], coords[:, 1]]
    order = np.argsort(intensities)[::-1]
    coords = coords[order]

    footprints = [
        Footprint(
            row=int(r),
            col=int(c),
            half_r=footprint_half[0],
            half_c=footprint_half[1],
            peak_id=i,
        )
        for i, (r, c) in enumerate(coords)
    ]
    return footprints


def integrate_footprints(
    spec: np.ndarray,
    footprints: list[Footprint],
    local_baseline: bool = True,
) -> np.ndarray:
    """Volume-integrate a spectrum over fixed footprints.

    Parameters
    ----------
    spec : np.ndarray
        Spectrum to integrate (reference or a reconstruction).
    footprints : list[Footprint]
        From pick_reference_peaks() on the REFERENCE.
    local_baseline
        If True, subtract a per-footprint baseline estimated from the box
        border before summing. Cheap correction for residual offsets.

    Returns
    -------
    np.ndarray, shape (len(footprints),)
        Volume per peak, ordered by footprint peak_id.
    """
    volumes = np.zeros(len(footprints), dtype=float)
    for k, fp in enumerate(footprints):
        rs, cs = fp.slices(spec.shape)
        box = spec[rs, cs]
        if local_baseline and box.size > 0:
            border = np.concatenate([
                box[0, :], box[-1, :], box[:, 0], box[:, -1]
            ])
            box = box - np.median(border)
        volumes[k] = float(box.sum())
    return volumes


def _local_maxima_numpy(spec, threshold, min_distance):  # pragma: no cover
    """Pure-numpy local maxima fallback (slower, used only without scipy)."""
    coords = []
    n1, n2 = spec.shape
    d = min_distance
    for i in range(d, n1 - d):
        for j in range(d, n2 - d):
            v = spec[i, j]
            if v <= threshold:
                continue
            window = spec[i - d:i + d + 1, j - d:j + d + 1]
            if v >= window.max():
                coords.append((i, j))
    return np.array(coords) if coords else np.empty((0, 2), dtype=int)
