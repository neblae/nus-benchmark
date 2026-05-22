"""
synthetic.py
------------
Generate synthetic 2D HSQC time-domain data (FIDs) with known peaks.

This is the key enabler for developing the pipeline *without spectrometer
access*: you create a fully-sampled FID whose peaks you defined yourself, so
you have perfect ground truth. Undersample it, reconstruct, integrate, and
check whether your scores recover the peaks you put in.

The model here is deliberately simple but physically reasonable for HSQC:
each peak is a 2D damped complex exponential (a Lorentzian line in each
dimension after FT), plus complex Gaussian noise.

    s(t1, t2) = sum_k A_k * exp(i*2*pi*f1_k*t1) * exp(-t1/T2_1_k)
                          * exp(i*2*pi*f2_k*t2) * exp(-t2/T2_2_k)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SynthPeak:
    """One synthetic HSQC cross-peak.

    Frequencies are given as fractions of the spectral width in each
    dimension (0-1), which keeps the generator independent of absolute
    sweep widths. amplitude sets the true relative volume — this is the
    quantity your scoring should recover.
    """

    f1_frac: float          # indirect-dim frequency (fraction of SW1)
    f2_frac: float          # direct-dim frequency (fraction of SW2)
    amplitude: float        # true peak amplitude (drives volume)
    t2_1: float = 0.15      # decay time in t1 (s) -> linewidth in F1
    t2_2: float = 0.15      # decay time in t2 (s) -> linewidth in F2
    phase: float = 0.0      # initial phase (radians)


@dataclass
class SynthConfig:
    n1: int = 128           # complex points in t1 (indirect)
    n2: int = 512           # complex points in t2 (direct)
    sw1: float = 2000.0     # spectral width F1 (Hz)
    sw2: float = 5000.0     # spectral width F2 (Hz)
    noise: float = 0.01     # std of complex Gaussian noise (rel. to unit amp)
    seed: int | None = 0


def default_peaklist() -> list[SynthPeak]:
    """A small, well-separated HSQC-like peak set with a wide dynamic range.

    The wide range of amplitudes (50x) is intentional: NUS error typically
    hits weak peaks hardest, so a good benchmark dataset must include them.
    """
    return [
        SynthPeak(f1_frac=0.30, f2_frac=0.25, amplitude=1.00),
        SynthPeak(f1_frac=0.55, f2_frac=0.40, amplitude=0.60),
        SynthPeak(f1_frac=0.45, f2_frac=0.65, amplitude=0.35),
        SynthPeak(f1_frac=0.70, f2_frac=0.55, amplitude=0.12),
        SynthPeak(f1_frac=0.20, f2_frac=0.80, amplitude=0.04),  # weak
    ]


def synth_hsqc_fid(
    peaks: list[SynthPeak] | None = None,
    cfg: SynthConfig | None = None,
) -> tuple[np.ndarray, SynthConfig, list[SynthPeak]]:
    """Build a fully-sampled complex 2D FID.

    Returns
    -------
    fid : np.ndarray, shape (n1, n2), complex
        Time-domain data, t1 along axis 0, t2 along axis 1.
    cfg : SynthConfig
        The configuration used (so downstream code knows SW, sizes).
    peaks : list[SynthPeak]
        The ground-truth peaks (so you can validate recovered volumes).
    """
    cfg = cfg or SynthConfig()
    peaks = peaks if peaks is not None else default_peaklist()
    rng = np.random.default_rng(cfg.seed)

    t1 = np.arange(cfg.n1) / cfg.sw1          # seconds
    t2 = np.arange(cfg.n2) / cfg.sw2
    T1, T2 = np.meshgrid(t1, t2, indexing="ij")

    fid = np.zeros((cfg.n1, cfg.n2), dtype=complex)
    for pk in peaks:
        f1 = pk.f1_frac * cfg.sw1
        f2 = pk.f2_frac * cfg.sw2
        signal = (
            pk.amplitude
            * np.exp(1j * (2 * np.pi * f1 * T1 + pk.phase))
            * np.exp(-T1 / pk.t2_1)
            * np.exp(1j * 2 * np.pi * f2 * T2)
            * np.exp(-T2 / pk.t2_2)
        )
        fid += signal

    if cfg.noise > 0:
        noise = cfg.noise * (
            rng.standard_normal(fid.shape) + 1j * rng.standard_normal(fid.shape)
        )
        fid += noise

    return fid, cfg, peaks


def true_peak_positions(
    peaks: list[SynthPeak], cfg: SynthConfig, out_shape: tuple[int, int]
) -> list[tuple[int, int]]:
    """Map ground-truth peaks to (row, col) indices in a spectrum.

    Useful for validating that your peak picker finds the peaks you planted.
    Assumes a simple FFT layout with fftshift (see reconstruct.py). out_shape
    is the reconstructed spectrum shape (after any zero-filling).
    """
    n1, n2 = out_shape
    positions = []
    for pk in peaks:
        # frequency fraction -> bin index after fftshift
        r = int(round((pk.f1_frac) * n1)) % n1
        c = int(round((pk.f2_frac) * n2)) % n2
        positions.append((r, c))
    return positions
