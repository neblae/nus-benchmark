# nus-benchmark

Groundwork for generating **machine-learning training data** that compares
**non-uniform sampling (NUS) schedules** for 2D HSQC NMR against a
**fully-sampled reference**.

The core idea: take one fully-sampled HSQC FID as ground truth, retrospectively
undersample it with many different schedules, reconstruct each, integrate the
peaks over **fixed footprints**, and score every reconstruction against the
reference. Each `(schedule, score)` pair is a labelled training example.

> **Status:** fully functional end-to-end on both synthetic and real Bruker
> data. Reconstruction uses `PythonIST` (pure-numpy IST, no external tools)
> by default. `nmrglue` is required only for reading real Bruker `ser` files.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py
```

This generates a synthetic HSQC, benchmarks Poisson-gap / uniform-random /
quantile-biased schedules at several sampling densities, and writes
`data/results/benchmark_results.csv`.

```bash
python scripts/plot_results.py
```

Generates `data/results/plots/benchmark_summary.png` — line plots, box plots,
and heatmaps comparing schedules across sampling densities.

## Using your own real fully-sampled HSQC

```bash
pip install nmrglue
python scripts/run_benchmark_real.py /path/to/experiment/11
```

The path should be the experiment-number folder containing `ser`, `acqus`, and
`acqu2s`. Use the **fully-sampled** experiment (check that `TD` in `acqu2s`
equals `NusTD` — i.e. 100% sampled). The script picks peaks once on the
reference spectrum, then retrospectively undersamples and scores each schedule.

**Why a dedicated loader (`load_real_reference.py`):** a NUS schedule selects
complex *t1 increments*, but a raw `ser` file stores t1 as quadrature-encoded
*rows* (2 rows per increment for States / States-TPPI / echo-antiecho). The
loader collapses those to one complex row per increment so a schedule index
unambiguously means "the i-th t1 increment" and undersampling never splits a
real/imaginary partner pair.

## Reconstruction backends

Three backends are available, in order of preference:

| Backend | Requires | Notes |
|---|---|---|
| `NmrPipeIST` | NMRPipe + hmsIST + nmrglue | Production-grade; hmsIST currently unavailable (site offline) |
| `PythonIST` | numpy only | Default — pure-numpy IST, no external tools needed |
| `ZeroFillFFT` | numpy only | Baseline; shows raw sampling artifacts, useful for testing |

The benchmark script picks the best available one automatically. With hmsIST
unavailable, `PythonIST` is the active backend and produces meaningful results.

### Installing NMRPipe (optional, for NmrPipeIST)

**1. NMRPipe** (free, from NIH/NIST — registration required):
```bash
# Fill out the form at https://www.ibbr.umd.edu/nmrpipe/install.html
# Download install.com, binval.com, s.tZ into the same folder, then:
chmod +x install.com binval.com
./install.com
echo 'export PATH="/path/to/nmrbin.mac11_64:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

**2. hmsIST** — the Hoch lab download site (comdnmr.uconn.edu) is currently
offline. Alternatives: request a copy from your NMR facility, or access it
via [NMRbox](https://nmrbox.org) (free cloud platform with NMR software
pre-installed).

**3. nmrglue** (also needed for reading real Bruker data):
```bash
pip install nmrglue
```

## Pipeline

```
schedules.py          generate a sampling schedule (which t1 increments to keep)
synthetic.py          build a synthetic fully-sampled HSQC FID with known peaks
undersample.py        apply a schedule to a full FID (retrospective undersampling)
reconstruct.py        FID -> spectrum  (PythonIST by default; NmrPipeIST if available)
peaks.py              pick peaks ONCE on the reference; integrate fixed footprints
scoring.py            compare NUS integrals to reference -> metrics + combined score
io_bruker.py          read real Bruker/TopSpin data (needs nmrglue)
scripts/plot_results.py   visualise results as line plots, box plots, heatmaps
```

## Design choices worth knowing

- **Pick peaks once, on the reference.** Every reconstruction is integrated
  over the *same* footprints, so a weakened peak shows up as a smaller
  integral, not a missing entry. Integrals stay directly comparable.
- **Normalize to a reference peak before scoring.** We measure *relative*
  quantification, not absolute scaling between reconstructions.
- **Reconstruction is pluggable.** `PythonIST` works out of the box with no
  external dependencies. `NmrPipeIST` shells out to hmsIST for production
  use. The benchmark auto-selects whichever is available.
- **Weak peaks are tracked separately.** NUS error hits small peaks first, so
  the score reports a weak-peak term explicitly — usually the most
  informative signal for an ML model.

## Moving to real data

1. Acquire (or obtain) **one fully-sampled** HSQC in TopSpin.
2. Confirm it is fully sampled: `TD == NusTD` in `acqu2s`.
3. Install `nmrglue` and run `run_benchmark_real.py` pointing at that experiment folder.
4. Scores and plots are written to `data/results/`.

## Tests

```bash
pytest -q
```

Tests run entirely on the synthetic path — no external dependencies.

## Roadmap / good next steps

- Add 2D lineshape fitting (`lmfit`) for overlapping peaks instead of box sums.
- Export schedules in NMRPipe `.nus` format for real reconstruction.
- Add a small ML baseline (e.g. predict `combined_score` from schedule
  features) once enough rows are generated.
- Wire up `NmrPipeIST` once hmsIST is accessible (via NMRbox or direct install).

## License

MIT — see `LICENSE`.
