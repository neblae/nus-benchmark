#!/usr/bin/env python3
"""
run_benchmark.py
----------------
End-to-end demo: generate a synthetic fully-sampled HSQC, benchmark several
NUS schedules against it, and write a results table. Runs with NO real data
and NO external NMR software — only numpy (+ scipy if available).

    python scripts/run_benchmark.py

This is the script to read first to understand how the pieces connect.
"""

from __future__ import annotations

import csv
import os
import shutil
import sys

import numpy as np

# Allow running from the repo root without installing.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nus_benchmark import (  # noqa: E402
    poisson_gap,
    uniform_random,
    quantile_biased,
    synth_hsqc_fid,
    apply_schedule,
    ZeroFillFFT,
    NmrPipeIST,
    pick_reference_peaks,
    integrate_footprints,
    score_spectrum,
)
from nus_benchmark.synthetic import SynthConfig  # noqa: E402


def _pick_reconstructor():
    """Use NmrPipeIST if hmsIST + nmrglue are available, else fall back."""
    try:
        import nmrglue  # noqa: F401
    except ImportError:
        print("nmrglue not found (pip install nmrglue). Using ZeroFillFFT.")
        return ZeroFillFFT(zerofill=1, apodize=True), "zerofill_fft"

    if shutil.which("hmsIST") is None:
        print("hmsIST not found on PATH. Using ZeroFillFFT.")
        print("  Install NMRPipe: https://www.ibbr.umd.edu/nmrpipe/install.html")
        print("  Install hmsIST:  http://comdnmr.uconn.edu/software")
        return ZeroFillFFT(zerofill=1, apodize=True), "zerofill_fft"

    print("hmsIST found — using NmrPipeIST reconstruction.")
    return NmrPipeIST(), "nmrpipe_ist"


def main() -> None:
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "results",
    )
    os.makedirs(out_dir, exist_ok=True)

    # 1) Ground-truth fully-sampled FID.
    cfg = SynthConfig(n1=128, n2=256, noise=0.01, seed=0)
    fid, cfg, true_peaks = synth_hsqc_fid(cfg=cfg)
    print(f"Synthetic FID: shape={fid.shape}, {len(true_peaks)} true peaks")

    # 2) Reconstruct the reference and pick peaks ONCE.
    recon, recon_name = _pick_reconstructor()
    print(f"Reconstructor: {recon_name}")
    ref_spec = recon.reconstruct(fid)
    footprints = pick_reference_peaks(ref_spec, snr_threshold=8.0)
    print(f"Picked {len(footprints)} peaks on the reference spectrum")
    ref_volumes = integrate_footprints(ref_spec, footprints)

    # 3) Define the schedules to compare, across a few sampling densities.
    n1 = cfg.n1
    densities = [0.50, 0.35, 0.25]
    generators = {
        "poisson_gap": lambda k, s: poisson_gap(n1, k, seed=s, sine_weight=2.0),
        "uniform_random": lambda k, s: uniform_random(n1, k, seed=s),
        "quantile_biased": lambda k, s: quantile_biased(n1, k, seed=s, decay=3.0),
    }

    rows = []
    for density in densities:
        n_keep = max(int(round(n1 * density)), 2)
        for name, gen in generators.items():
            # A few seeds per setting -> more training rows, variance estimate.
            for seed in range(3):
                schedule = gen(n_keep, seed)
                masked = apply_schedule(fid, schedule, mode="masked")
                nus_spec = recon.reconstruct(masked)
                nus_volumes = integrate_footprints(nus_spec, footprints)
                report = score_spectrum(ref_volumes, nus_volumes)

                row = {
                    "schedule": name,
                    "reconstructor": recon_name,
                    "density": density,
                    "n_keep": n_keep,
                    "seed": seed,
                    **report.as_dict(),
                }
                rows.append(row)
                print(
                    f"{name:16s} d={density:.2f} seed={seed} "
                    f"-> score={report.combined_score:.3f} "
                    f"r={report.pearson_r:.3f} missing={report.n_missing}"
                )

    # 4) Write results.
    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    # 5) Quick summary: mean score per schedule (the headline comparison).
    print("\nMean combined score by schedule (higher = closer to reference):")
    for name in generators:
        scores = [r["combined_score"] for r in rows if r["schedule"] == name]
        print(f"  {name:16s} {np.mean(scores):.3f}")


if __name__ == "__main__":
    main()
