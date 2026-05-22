"""
load_real_reference.py
----------------------
Load ONE fully-sampled real Bruker/TopSpin HSQC `ser` file and adapt it into
the array shape the rest of the pipeline expects, so synthetic schedules can
retrospectively undersample REAL data.

This is the bridge between your spectrometer data and the existing pipeline.
After loading, everything downstream (undersample -> reconstruct -> peaks ->
score) is identical to the synthetic demo.

The one subtlety this module exists to handle:

    A NUS schedule selects COMPLEX t1 INCREMENTS.
    A raw Bruker `ser` file stores t1 as quadrature-encoded ROWS, where each
    complex increment occupies a small group of consecutive rows (2 rows for
    States / States-TPPI / echo-antiecho hypercomplex acquisition).

If you undersample raw rows directly, you'll split real/imaginary partners of
the same increment and corrupt the data. So we convert the raw `ser` into a
clean complex array indexed by INCREMENT (one row per complex t1 point), which
is exactly what undersample.apply_schedule() and the schedules in schedules.py
assume. The schedule index i then unambiguously means "the i-th t1 increment".

Requires nmrglue:  pip install nmrglue
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RealReference:
    """A loaded, increment-indexed real HSQC FID ready for the pipeline."""

    fid: np.ndarray          # complex, shape (n_increments_t1, n_complex_t2)
    n1: int                  # number of complex t1 increments
    n2: int                  # number of complex t2 points
    sw1: float               # F1 spectral width (Hz)
    sw2: float               # F2 spectral width (Hz)
    fnmode: int              # Bruker FnMODE of the indirect dim (encoding)
    dic: dict                # full Bruker parameter dict (for reference)

    def summary(self) -> str:
        enc = _FNMODE_NAMES.get(self.fnmode, f"code {self.fnmode}")
        return (
            f"RealReference: t1 increments={self.n1}, t2 complex pts={self.n2}, "
            f"SW1={self.sw1:.1f} Hz, SW2={self.sw2:.1f} Hz, encoding={enc}"
        )


# Bruker FnMODE codes for the indirect dimension.
_FNMODE_NAMES = {
    0: "undefined",
    1: "QF (single)",
    2: "QSEC",
    3: "TPPI",
    4: "States",
    5: "States-TPPI",
    6: "echo-antiecho",
}
# Encodings where each complex increment = 2 hypercomplex rows on disk.
_HYPERCOMPLEX = {4, 5, 6}


def load_real_reference(expdir: str, expect_full: bool = True) -> RealReference:
    """Load a fully-sampled 2D Bruker `ser` file as an increment-indexed FID.

    Parameters
    ----------
    expdir : str
        Path to the experiment-number folder containing `ser`, `acqus`,
        `acqu2s`, e.g. "/path/to/sample/10".
    expect_full : bool
        If True, verify the experiment is fully sampled (NusAMOUNT == 100,
        or no on-disk schedule shorter than TD1) and raise if it is not.
        A fully-sampled reference is required for retrospective undersampling.

    Returns
    -------
    RealReference
    """
    ng = _require_nmrglue()

    dic, raw = ng.bruker.read(expdir)          # raw, t2 interleaved real/imag

    # Now that we have the parameters, verify the experiment is fully sampled.
    _check_not_nus(dic, expdir, expect_full)

    # nmrglue returns t2 already combined into complex if it can; ensure complex.
    data = np.asarray(raw)
    if not np.iscomplexobj(data):
        # Fallback: combine interleaved real/imag along the last axis.
        data = data[..., ::2] + 1j * data[..., 1::2]

    fnmode = _get_fnmode(dic)
    sw1, sw2 = _get_spectral_widths(dic)

    # Collapse hypercomplex t1 rows into one complex value per INCREMENT.
    fid_by_increment = _to_increment_indexed(data, fnmode)

    n1, n2 = fid_by_increment.shape
    ref = RealReference(
        fid=fid_by_increment,
        n1=n1,
        n2=n2,
        sw1=sw1,
        sw2=sw2,
        fnmode=fnmode,
        dic=dic,
    )
    return ref


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _require_nmrglue():
    try:
        import nmrglue as ng
        return ng
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "nmrglue is required to read real Bruker data. "
            "Install with `pip install nmrglue`."
        ) from e


def _check_not_nus(dic: dict, expdir: str, expect_full: bool) -> None:
    """Verify the experiment is fully sampled before using it as a reference.

    On Bruker, an experiment set up through the NUS framework is still FULLY
    sampled when NusAMOUNT == 100 (every increment acquired). A genuinely
    undersampled experiment has NusAMOUNT < 100 and/or an explicit nuslist with
    fewer points than TD. We check NusAMOUNT directly — that's the parameter
    that actually decides it — and fall back to looking for a populated
    nuslist file. A bare `NUSLIST= <automatic>` in acqus is NOT undersampling
    on its own; only the amount matters.
    """
    import os

    nus_amount = _first(dic, ("acqus", "acqu"), "NusAMOUNT", default=None)
    if nus_amount is not None:
        try:
            amt = float(nus_amount)
        except (TypeError, ValueError):
            amt = None
        if amt is not None and amt < 100.0 and expect_full:
            raise ValueError(
                f"NusAMOUNT={amt} in {expdir} — this experiment is only "
                f"{amt}% sampled (non-uniform). A FULLY-SAMPLED (100%) "
                "reference is required. Pick a 100% experiment, or pass "
                "expect_full=False to override."
            )

    # Secondary guard: an explicit on-disk schedule file shorter than TD.
    for fname in ("nuslist", "vclist", "nusxlist"):
        for sub in ("", "lists"):
            p = os.path.join(expdir, sub, fname) if sub else os.path.join(expdir, fname)
            if os.path.exists(p) and expect_full:
                # Only flag if it actually encodes fewer points than the grid.
                td1 = _first(dic, ("acqu2s", "acqu2"), "TD", default=None)
                n_sched = _count_schedule_points(p)
                if td1 and n_sched and n_sched < int(td1):
                    raise ValueError(
                        f"Found schedule {p} with {n_sched} points but TD1="
                        f"{int(td1)} — this is non-uniformly sampled. A "
                        "fully-sampled reference is required (or pass "
                        "expect_full=False)."
                    )


def _count_schedule_points(path: str) -> int | None:
    """Count non-empty lines in a Bruker schedule file, or None if unreadable."""
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return None


def _get_fnmode(dic: dict) -> int:
    """Read FnMODE from the indirect-dim acquisition parameters (acqu2s)."""
    for key in ("acqu2s", "acqu2"):
        if key in dic and "FnMODE" in dic[key]:
            return int(dic[key]["FnMODE"])
    return 4  # default to States; most common for HSQC


def _get_spectral_widths(dic: dict) -> tuple[float, float]:
    """Return (SW1, SW2) in Hz. SW_h is in Hz in Bruker dicts."""
    sw2 = _first(dic, ("acqus", "acqu"), "SW_h", default=5000.0)
    sw1 = _first(dic, ("acqu2s", "acqu2"), "SW_h", default=2000.0)
    return float(sw1), float(sw2)


def _first(dic: dict, keys: tuple, field: str, default):
    for k in keys:
        if k in dic and field in dic[k]:
            return dic[k][field]
    return default


def _to_increment_indexed(data: np.ndarray, fnmode: int) -> np.ndarray:
    """Convert raw t1 rows to one complex row per t1 increment.

    For hypercomplex encodings (States / States-TPPI / echo-antiecho), the raw
    array has 2 rows per increment (the cos- and sin-modulated parts). For
    quantification/benchmarking we form one representative complex series per
    increment so that a schedule index maps to a single physical increment.

    We keep the cos-modulated (first) component of each pair as the complex t1
    series. This is sufficient and consistent for retrospective undersampling
    + the baseline reconstructor; a full hypercomplex reconstruction backend
    (e.g. NMRPipe/hmsIST) would instead consume both components and the
    schedule together. The mapping below keeps increment indexing unambiguous.
    """
    if fnmode in _HYPERCOMPLEX and data.shape[0] % 2 == 0:
        # rows: [inc0_A, inc0_B, inc1_A, inc1_B, ...] -> take A rows.
        return data[0::2, :].copy()
    # QF / TPPI / single-component: rows already correspond to increments.
    return data.copy()
