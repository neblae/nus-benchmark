"""
Tests for the nus-benchmark pipeline. Run with:  pytest -q
These exercise the synthetic path so they need no real data or NMR software.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nus_benchmark import (
    poisson_gap,
    uniform_random,
    quantile_biased,
    synth_hsqc_fid,
    apply_schedule,
    ZeroFillFFT,
    pick_reference_peaks,
    integrate_footprints,
    score_spectrum,
)
from nus_benchmark.synthetic import SynthConfig


# --------------------------- schedules ------------------------------------- #
@pytest.mark.parametrize("gen", [poisson_gap, uniform_random, quantile_biased])
def test_schedule_count_and_range(gen):
    sched = gen(128, 40, seed=1)
    assert len(sched) == 40
    assert sched.min() >= 0 and sched.max() < 128
    assert len(np.unique(sched)) == 40          # no duplicates
    assert np.all(np.diff(sched) > 0)           # sorted, strictly increasing


def test_schedule_reproducible():
    a = poisson_gap(128, 40, seed=7)
    b = poisson_gap(128, 40, seed=7)
    assert np.array_equal(a, b)


def test_keep_exceeds_total_raises():
    with pytest.raises(ValueError):
        uniform_random(10, 20)


def test_poisson_concentrates_early():
    # With sine weighting, mean index should sit earlier than uniform's ~midpoint.
    pg = poisson_gap(256, 80, seed=3, sine_weight=3.0)
    assert pg.mean() < 256 / 2


# --------------------------- undersampling --------------------------------- #
def test_apply_schedule_masked_zeros_unsampled():
    fid, cfg, _ = synth_hsqc_fid(cfg=SynthConfig(n1=64, n2=64, seed=0))
    sched = uniform_random(64, 20, seed=0)
    masked = apply_schedule(fid, sched, mode="masked")
    assert masked.shape == fid.shape
    # Kept rows identical, dropped rows zero.
    assert np.allclose(masked[sched], fid[sched])
    dropped = np.setdiff1d(np.arange(64), sched)
    assert np.allclose(masked[dropped], 0)


def test_apply_schedule_compact_shape():
    fid, cfg, _ = synth_hsqc_fid(cfg=SynthConfig(n1=64, n2=64, seed=0))
    sched = uniform_random(64, 20, seed=0)
    compact = apply_schedule(fid, sched, mode="compact")
    assert compact.shape == (20, 64)


# --------------------------- peaks / integrals ----------------------------- #
def test_pick_and_integrate_reference():
    fid, cfg, true_peaks = synth_hsqc_fid(cfg=SynthConfig(seed=0))
    spec = ZeroFillFFT().reconstruct(fid)
    fps = pick_reference_peaks(spec, snr_threshold=8.0)
    # Should find a reasonable number of the planted peaks.
    assert len(fps) >= 3
    vols = integrate_footprints(spec, fps)
    assert np.all(vols > 0)
    # Strongest footprint (peak_id 0) should have the largest volume.
    assert np.argmax(vols) == 0


# --------------------------- scoring --------------------------------------- #
def test_identical_spectra_score_high():
    ref = np.array([1.0, 0.6, 0.3, 0.1])
    report = score_spectrum(ref, ref.copy())
    assert report.pearson_r == pytest.approx(1.0, abs=1e-9)
    assert report.rmsd_norm == pytest.approx(0.0, abs=1e-9)
    assert report.n_missing == 0
    assert report.combined_score > 0.95


def test_missing_peak_detected():
    ref = np.array([1.0, 0.6, 0.3, 0.1])
    nus = np.array([1.0, 0.6, 0.3, 0.0])   # weakest peak vanished
    report = score_spectrum(ref, nus)
    assert report.n_missing == 1


def test_full_pipeline_runs_and_orders_schedules():
    """Smoke test: full sampling should beat heavy undersampling on score."""
    cfg = SynthConfig(n1=128, n2=128, noise=0.01, seed=0)
    fid, cfg, _ = synth_hsqc_fid(cfg=cfg)
    recon = ZeroFillFFT()
    ref_spec = recon.reconstruct(fid)
    fps = pick_reference_peaks(ref_spec, snr_threshold=8.0)
    ref_vol = integrate_footprints(ref_spec, fps)

    def score_at(density):
        n_keep = int(128 * density)
        sched = poisson_gap(128, n_keep, seed=0)
        spec = recon.reconstruct(apply_schedule(fid, sched, mode="masked"))
        return score_spectrum(ref_vol, integrate_footprints(spec, fps)).combined_score

    dense = score_at(0.9)
    sparse = score_at(0.2)
    assert dense >= sparse


# --------------------------- real-reference loader ------------------------- #
from nus_benchmark.load_real_reference import _to_increment_indexed  # noqa: E402


def test_hypercomplex_increment_indexing():
    """States/echo-antiecho: 2 raw rows per increment -> collapse to 1 row."""
    n_inc, n2 = 8, 16
    raw = np.zeros((n_inc * 2, n2), dtype=complex)
    for i in range(n_inc):
        raw[2 * i] = (i + 1)        # 'A' row carries the increment id
        raw[2 * i + 1] = -(i + 1)   # 'B' row
    for fnmode in (4, 5, 6):        # States, States-TPPI, echo-antiecho
        out = _to_increment_indexed(raw, fnmode)
        assert out.shape == (n_inc, n2)
        assert np.allclose(out[:, 0], np.arange(1, n_inc + 1))


def test_tppi_passthrough():
    """TPPI/QF: raw rows already correspond to increments -> unchanged."""
    raw = np.arange(6 * 4).reshape(6, 4).astype(complex)
    out = _to_increment_indexed(raw, 3)  # TPPI
    assert out.shape == (6, 4)
    assert np.allclose(out, raw)


def test_schedule_indexes_real_increments_correctly():
    """A schedule on the increment-indexed FID selects whole increments."""
    n_inc, n2 = 20, 8
    raw = np.zeros((n_inc * 2, n2), dtype=complex)
    for i in range(n_inc):
        raw[2 * i] = i             # tag each increment's A row by its id
        raw[2 * i + 1] = i + 0.5
    fid = _to_increment_indexed(raw, 4)        # States -> (20, 8)
    sched = uniform_random(n_inc, 8, seed=0)
    kept = apply_schedule(fid, sched, mode="compact")
    # Each kept row's tag should equal its original increment index.
    assert np.allclose(kept[:, 0].real, sched)
