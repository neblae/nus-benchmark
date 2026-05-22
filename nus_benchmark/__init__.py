"""
nus-benchmark
=============

Groundwork for generating ML training data that compares non-uniform
sampling (NUS) schedules for 2D HSQC NMR against a fully-sampled reference.

Pipeline
--------
    generate schedule  ->  retrospectively undersample a full FID
    ->  reconstruct      ->  peak-pick + volume-integrate
    ->  score against the fully-sampled reference

Every public piece is importable from the top level:

    from nus_benchmark import (
        poisson_gap, uniform_random, quantile_biased,   # schedules
        synth_hsqc_fid,                                  # synthetic data
        apply_schedule,                                  # undersampling
        Reconstructor, ZeroFillFFT,                      # reconstruction
        integrate_footprints, pick_reference_peaks,      # peaks/integrals
        score_spectrum,                                  # labelling
    )
"""

from .schedules import poisson_gap, uniform_random, quantile_biased
from .synthetic import synth_hsqc_fid, SynthPeak
from .undersample import apply_schedule
from .reconstruct import Reconstructor, ZeroFillFFT
from .peaks import pick_reference_peaks, integrate_footprints, Footprint
from .scoring import score_spectrum, ScoreReport
from .load_real_reference import load_real_reference, RealReference

__all__ = [
    "load_real_reference",
    "RealReference",
    "poisson_gap",
    "uniform_random",
    "quantile_biased",
    "synth_hsqc_fid",
    "SynthPeak",
    "apply_schedule",
    "Reconstructor",
    "ZeroFillFFT",
    "pick_reference_peaks",
    "integrate_footprints",
    "Footprint",
    "score_spectrum",
    "ScoreReport",
]

__version__ = "0.1.0"
