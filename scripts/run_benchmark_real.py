#!/usr/bin/env python3
"""
run_benchmark_real.py
---------------------
Benchmark synthetic NUS schedules against YOUR real fully-sampled HSQC.

Usage:
    python scripts/run_benchmark_real.py /path/to/sample/10

This is the real-data twin of run_benchmark.py. The ONLY difference is the
first step: instead of synth_hsqc_fid(), it loads your real `ser` file via
load_real_reference(). Everything after that — undersample, reconstruct,
pick peaks once on the reference, integrate fixed footprints, score — is
byte-for-byte the same code path as the synthetic demo.

Requires nmrglue (pip install nmrglue).
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nus_benchmark import (  # noqa: E402
    poisson_gap,
    uniform_random,
    quantile_biased,
    apply_schedule,
    ZeroFillFFT,
    pick_reference_peaks,
    integrate_footprints,
    score_spectrum,
)
from nus_benchmark.load_real_reference import load_real_reference  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_benchmark_real.py /path/to/experiment_number")
        print("  (the folder containing the `ser`, `acqus`, `acqu2s` files)")
        sys.exit(1)
    expdir = sys.argv[1]

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "results",
    )
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load the REAL fully-sampled reference FID.
    ref = load_real_reference(expdir, expect_full=True)
    print(ref.summary())
    fid = ref.fid
    n1 = ref.n1

    # 2) Reconstruct the reference and pick peaks ONCE.
    recon = ZeroFillFFT(zerofill=1, apodize=True, mode="real")
    ref_spec = recon.reconstruct(fid)
    footprints = pick_reference_peaks(ref_spec, snr_threshold=10.0)
    print(f"Picked {len(footprints)} peaks on the real reference spectrum")
    if len(footprints) == 0:
        print("No peaks found — try lowering snr_threshold in this script.")
        sys.exit(1)
    ref_volumes = integrate_footprints(ref_spec, footprints)

    # 3) Benchmark schedules across sampling densities.
    densities = [0.50, 0.35, 0.25, 0.15]
    generators = {
        "poisson_gap": lambda k, s: poisson_gap(n1, k, seed=s, sine_weight=2.0),
        "uniform_random": lambda k, s: uniform_random(n1, k, seed=s),
        "quantile_biased": lambda k, s: quantile_biased(n1, k, seed=s, decay=3.0),
    }

    rows = []
    for density in densities:
        n_keep = max(int(round(n1 * density)), 2)
        for name, gen in generators.items():
            for seed in range(5):     # more seeds -> more training rows
                schedule = gen(n_keep, seed)
                masked = apply_schedule(fid, schedule, mode="masked")
                nus_spec = recon.reconstruct(masked)
                nus_volumes = integrate_footprints(nus_spec, footprints)
                report = score_spectrum(ref_volumes, nus_volumes)
                rows.append({
                    "schedule": name, "density": density, "n_keep": n_keep,
                    "seed": seed, **report.as_dict(),
                })

    # 4) Save and summarize.
    csv_path = os.path.join(out_dir, "benchmark_results_real.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {csv_path}")

    print("\nMean combined score by schedule (higher = closer to reference):")
    for name in generators:
        s = [r["combined_score"] for r in rows if r["schedule"] == name]
        print(f"  {name:16s} {np.mean(s):.3f}")

    print("\nMean score by density (shows where quantification breaks down):")
    for d in densities:
        s = [r["combined_score"] for r in rows if r["density"] == d]
        print(f"  density {d:.2f}: {np.mean(s):.3f}")


if __name__ == "__main__":
    main()
