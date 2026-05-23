# nus-benchmark

Groundwork for generating **machine-learning training data** that compares
**non-uniform sampling (NUS) schedules** for 2D HSQC NMR against a
**fully-sampled reference**.

The core idea: take one fully-sampled HSQC FID as ground truth, retrospectively
undersample it with many different schedules, reconstruct each, integrate the
peaks over **fixed footprints**, and score every reconstruction against the
reference. Each `(schedule, score)` pair is a labelled training example.

> **Status:** runs end-to-end *today* on synthetic data with only `numpy`
> (and optionally `scipy`). No spectrometer, NMRPipe, or TopSpin needed to
> develop the pipeline. Real-data and real-reconstruction hooks are included
> as clearly-marked plug-in points. The benchmark script automatically uses
> `NmrPipeIST` when hmsIST is installed, and falls back to `ZeroFillFFT`
> with a message when it is not.

## Why retrospective undersampling?

For comparing *schedules*, retrospective undersampling of a single full FID is
the right design: every undersampled example shares the exact same underlying
signal and noise, so the schedule is the *only* variable. That gives clean ML
labels and lets one real experiment yield thousands of training pairs — ideal
when spectrometer access is limited.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py
```

This generates a synthetic HSQC, benchmarks Poisson-gap / uniform-random /
quantile-biased schedules at several sampling densities, and writes
`data/results/benchmark_results.csv`. Expected: Poisson-gap scores highest,
the naive quantile scheme lowest — matching the NUS literature.

## Installing NMRPipe + hmsIST (for real reconstruction)

The benchmark script automatically detects whether hmsIST is available and
falls back to `ZeroFillFFT` if not. To enable real IST reconstruction:

**1. NMRPipe** (free, from NIH/NIST — registration required):
```bash
# Fill out the form at https://www.ibbr.umd.edu/nmrpipe/install.html
# They email you a download link. Then:
chmod +x install.com
./install.com
```

**2. hmsIST** (Hoch lab, UConn):
```bash
# Download from http://comdnmr.uconn.edu/software
tar -xzf hmsist*.tar.gz
cd hmsist*/
make
echo 'export PATH="/path/to/hmsist/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
hmsIST --help   # verify it works
```

**3. nmrglue** (Python, needed for reading/writing NMRPipe files):
```bash
pip install nmrglue
```

Once all three are installed, `python scripts/run_benchmark.py` will
automatically switch to `NmrPipeIST`. NMRPipe registration can take a day
or two — the pipeline works fully with `ZeroFillFFT` in the meantime.

## Using your own real fully-sampled HSQC

This is the intended production setup: a **real** fully-sampled `ser` file as
the ground-truth reference, undersampled by **synthetic** schedules.

```bash
pip install nmrglue          # needed to read Bruker data
python scripts/run_benchmark_real.py /path/to/sample/10
```

where the path is the experiment-number folder containing `ser`, `acqus`, and
`acqu2s`. The script reads your FID, picks peaks once on its reference
spectrum, then undersamples and scores exactly as the synthetic demo does.

**Why a dedicated loader (`load_real_reference.py`):** a NUS schedule selects
complex *t1 increments*, but a raw `ser` file stores t1 as quadrature-encoded
*rows* (2 rows per increment for States / States-TPPI / echo-antiecho). The
loader collapses those to one complex row per increment, so a schedule index
unambiguously means "the i-th t1 increment" and undersampling never splits a
real/imaginary partner pair. It also refuses a `ser` that is already NUS
(presence of a `nuslist`), since the reference must be fully sampled.

## Pipeline

```
schedules.py    generate a sampling schedule (which t1 increments to keep)
synthetic.py    build a synthetic fully-sampled HSQC FID with known peaks
undersample.py  apply a schedule to a full FID (retrospective undersampling)
reconstruct.py  FID -> spectrum  (pluggable: baseline FFT now, IST/MDD later)
peaks.py        pick peaks ONCE on the reference; integrate fixed footprints
scoring.py      compare NUS integrals to reference -> metrics + combined score
io_bruker.py    read real Bruker/TopSpin data when you have it (needs nmrglue)
```

## Design choices worth knowing

- **Pick peaks once, on the reference.** Every reconstruction is integrated
  over the *same* footprints, so a weakened peak shows up as a smaller
  integral, not a missing entry. Integrals stay directly comparable.
- **Normalize to a reference peak before scoring.** We measure *relative*
  quantification, not absolute scaling between reconstructions.
- **Reconstruction is pluggable.** `ZeroFillFFT` needs nothing external and is
  great for development. `NmrPipeIST` shells out to hmsIST and is the
  scalable path for generating large datasets headlessly. The benchmark
  script auto-selects whichever is available.
- **Weak peaks are tracked separately.** NUS error hits small peaks first, so
  the score reports a weak-peak term explicitly — usually the most
  informative signal for an ML model.

## Moving to real data

1. Acquire (or obtain) **one fully-sampled** HSQC in TopSpin.
2. Read its raw FID with `io_bruker.read_fid()` (install `nmrglue`).
3. Use that array everywhere the demo uses `synth_hsqc_fid()`.
4. Install NMRPipe + hmsIST (see above) — the benchmark script will
   automatically use `NmrPipeIST` and loop over thousands of schedules
   unattended.

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

## License

MIT — see `LICENSE`.
