"""
reconstruct.py
--------------
Pluggable reconstruction layer.

The rest of the pipeline only ever calls `Reconstructor.reconstruct(fid)` and
gets back a real-valued 2D spectrum. That abstraction is the whole point: you
can develop today with the built-in ZeroFillFFT (no external software needed),
then drop in a scripted NMRPipe/hmsIST backend later without touching the
scoring or schedule code.

Implement a new backend by subclassing Reconstructor and overriding
reconstruct(). Keep the contract:

    input : complex FID (n1, n2), unsampled t1 rows zeroed ("masked" mode)
    output: real spectrum (m1, m2), fftshifted so DC is centered
"""

from __future__ import annotations

import abc
import os
import subprocess
import tempfile

import numpy as np


class Reconstructor(abc.ABC):
    """Abstract reconstruction backend."""

    @abc.abstractmethod
    def reconstruct(self, fid: np.ndarray) -> np.ndarray:
        """FID (complex, t1-zeroed where unsampled) -> real spectrum."""
        raise NotImplementedError

    # Convenience so backends are callable.
    def __call__(self, fid: np.ndarray) -> np.ndarray:
        return self.reconstruct(fid)


class ZeroFillFFT(Reconstructor):
    """Baseline 'reconstruction': window, zero-fill, FFT, magnitude.

    This is NOT a real NUS reconstruction algorithm — it does no gap filling,
    so undersampled spectra will show sampling artifacts (that's the point;
    it lets you *see* the cost of a schedule with zero dependencies). It's
    ideal for developing and unit-testing the rest of the pipeline.

    For real work, swap in an IST/MDD backend (see NmrPipeIST below).
    """

    def __init__(self, zerofill: int = 1, apodize: bool = True,
                 mode: str = "real"):
        # zerofill = number of times to double the points in each dim.
        # mode: "real" (phase-sensitive; gaps create schedule-dependent
        #       artifacts you can measure) or "magnitude" (smoother, hides
        #       artifacts). "real" makes the schedule comparison meaningful.
        self.zerofill = zerofill
        self.apodize = apodize
        self.mode = mode

    def reconstruct(self, fid: np.ndarray) -> np.ndarray:
        data = fid.copy()
        if self.apodize:
            data = _apodize_2d(data)
        data = _zerofill_2d(data, self.zerofill)
        spec = np.fft.fftshift(np.fft.fft2(data))
        if self.mode == "magnitude":
            return np.abs(spec)
        # Real part: unfilled NUS gaps scatter sampling artifacts across F1,
        # and the artifact pattern depends on the schedule -> schedules now
        # score differently, which is what we want to demonstrate.
        return spec.real


class NmrPipeIST(Reconstructor):
    """NMRPipe + hmsIST reconstruction backend for NUS data.

    Requires:
    - hmsIST on PATH  (Hoch lab: http://comdnmr.uconn.edu/software)
    - nmrglue         (`pip install nmrglue`)

    The spectral parameters (sw1, sw2, obs1, obs2, car1_ppm, car2_ppm) are
    needed to write a valid NMRPipe file header. For real Bruker data, use
    from_bruker_dic() to extract them automatically from the acqus dict
    returned by io_bruker.read_fid(). The defaults are reasonable values for
    a 15N-HSQC at 800 MHz.

    Flow
    ----
    1. Infer the NUS schedule from non-zero rows of the masked FID.
    2. Write only the sampled rows + schedule to a temp directory in
       NMRPipe format (nmrglue.pipe.write + plain-text nuslist).
    3. Shell out to hmsIST; it fills in the missing t1 increments via IST.
    4. Read the reconstructed full FID back with nmrglue.pipe.read.
    5. Apodize, zero-fill, and FFT identically to ZeroFillFFT so scores
       from both backends are directly comparable.
    """

    def __init__(
        self,
        nmrpipe_bin: str = "nmrPipe",
        hmsist_bin: str = "hmsIST",
        workdir: str | None = None,
        niter: int = 500,
        sw1: float = 3030.0,     # F1 (indirect) spectral width, Hz
        sw2: float = 8993.0,     # F2 (direct) spectral width, Hz
        obs1: float = 81.076,    # F1 observe frequency, MHz  (15N at 800 MHz)
        obs2: float = 800.130,   # F2 observe frequency, MHz  (1H at 800 MHz)
        car1_ppm: float = 118.0, # F1 carrier position, ppm
        car2_ppm: float = 4.7,   # F2 carrier position, ppm
        label1: str = "15N",
        label2: str = "1H",
    ):
        self.nmrpipe_bin = nmrpipe_bin
        self.hmsist_bin = hmsist_bin
        self.workdir = workdir
        self.niter = niter
        self.sw1 = sw1
        self.sw2 = sw2
        self.obs1 = obs1
        self.obs2 = obs2
        self.car1_ppm = car1_ppm
        self.car2_ppm = car2_ppm
        self.label1 = label1
        self.label2 = label2

    @classmethod
    def from_bruker_dic(cls, bruker_dic: dict, **kwargs) -> "NmrPipeIST":
        """Construct with spectral parameters from io_bruker.read_fid() output.

        Example
        -------
        dic, fid = io_bruker.read_fid("/path/to/expdir")
        recon = NmrPipeIST.from_bruker_dic(dic)
        """
        acq = bruker_dic.get("acqus", {})
        acq2 = bruker_dic.get("acqu2s", {})
        sfo2 = float(acq.get("SFO1", 800.130))
        sfo1 = float(acq2.get("SFO1", 81.076))
        return cls(
            sw2=float(acq.get("SW_h", 8993.0)),
            sw1=float(acq2.get("SW_h", 3030.0)),
            obs2=sfo2,
            obs1=sfo1,
            car2_ppm=float(acq.get("O1", 4.7 * sfo2)) / sfo2,
            car1_ppm=float(acq2.get("O1", 118.0 * sfo1)) / sfo1,
            label2=str(acq.get("NUC1", "1H")),
            label1=str(acq2.get("NUC1", "15N")),
            **kwargs,
        )

    def reconstruct(self, fid: np.ndarray) -> np.ndarray:  # pragma: no cover
        ng = _require_nmrglue()
        n1, n2 = fid.shape

        # Step 1: infer which t1 rows were actually sampled.
        schedule = np.where(np.any(fid != 0, axis=1))[0]
        if len(schedule) == 0:
            raise ValueError("FID has no non-zero rows; cannot infer NUS schedule.")

        with tempfile.TemporaryDirectory(dir=self.workdir) as tmpdir:
            fid_in_path = os.path.join(tmpdir, "fid_in.fid")
            fid_out_path = os.path.join(tmpdir, "fid_out.fid")
            sch_path = os.path.join(tmpdir, "nuslist")

            # Step 2a: write the sampled rows only in NMRPipe format.
            dic = self._build_pipe_dic(n1, n2, len(schedule))
            compact_fid = np.ascontiguousarray(fid[schedule], dtype=np.complex64)
            ng.pipe.write(fid_in_path, dic, compact_fid, overwrite=True)

            # Step 2b: write the NUS schedule (0-based indices, one per line).
            with open(sch_path, "w") as f:
                f.writelines(f"{int(idx)}\n" for idx in schedule)

            # Step 3: run hmsIST to reconstruct the full t1 grid.
            _run_hmsist(
                self.hmsist_bin, fid_in_path, fid_out_path,
                sch_path, self.niter, tmpdir,
            )

            # Step 4: read back the reconstructed FID (full n1 rows).
            _, rec_fid = ng.pipe.read(fid_out_path)

        # Step 5: identical post-processing to ZeroFillFFT for fair comparison.
        rec_fid = rec_fid.astype(np.complex128)
        rec_fid = _apodize_2d(rec_fid)
        rec_fid = _zerofill_2d(rec_fid, doublings=1)
        spec = np.fft.fftshift(np.fft.fft2(rec_fid))
        return spec.real

    def _build_pipe_dic(self, n1: int, n2: int, n_sampled: int) -> dict:
        """Minimal NMRPipe header for a compact 2D NUS FID (sampled rows only)."""
        ng = _require_nmrglue()
        dic = getattr(ng.pipe, "create_dic", dict)()
        dic.update({
            "FDDIMCOUNT": 2.0,
            "FDSIZE": float(n2),            # direct-dim complex points per row
            "FDSPECNUM": float(n_sampled),  # number of rows in this file
            # F2 — direct dimension (1H)
            "FDF2SW": float(self.sw2),
            "FDF2OBS": float(self.obs2),
            "FDF2CAR": float(self.car2_ppm),
            "FDF2LABEL": self.label2,
            "FDF2QUADFLAG": 1.0,
            "FDF2APOD": float(n2),
            "FDF2TDSIZE": float(n2),
            # F1 — indirect dimension (15N)
            "FDF1SW": float(self.sw1),
            "FDF1OBS": float(self.obs1),
            "FDF1CAR": float(self.car1_ppm),
            "FDF1LABEL": self.label1,
            "FDF1QUADFLAG": 1.0,
            "FDF1APOD": float(n1),
            "FDF1TDSIZE": float(n1),
        })
        return dic


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _require_nmrglue():
    try:
        import nmrglue as ng
        return ng
    except ImportError as e:
        raise ImportError(
            "nmrglue is required for NmrPipeIST. Install with `pip install nmrglue`."
        ) from e


def _run_hmsist(
    hmsist_bin: str,
    fid_in: str,
    fid_out: str,
    sch_file: str,
    niter: int,
    cwd: str,
) -> None:
    """Shell out to hmsIST; raises RuntimeError on non-zero exit."""
    cmd = [hmsist_bin, fid_in, fid_out, "-sch", sch_file, "-niter", str(niter), "-ist"]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(
            f"hmsIST exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )



def _apodize_2d(data: np.ndarray) -> np.ndarray:
    """Apply a cosine (sine-bell-ish) window in both dims to reduce truncation
    wiggles. Real processing would use parameters matched to the acquisition."""
    n1, n2 = data.shape
    w1 = np.cos(np.linspace(0, np.pi / 2, n1)) ** 1
    w2 = np.cos(np.linspace(0, np.pi / 2, n2)) ** 1
    return data * w1[:, None] * w2[None, :]


def _zerofill_2d(data: np.ndarray, doublings: int) -> np.ndarray:
    if doublings <= 0:
        return data
    n1, n2 = data.shape
    m1 = n1 * (2 ** doublings)
    m2 = n2 * (2 ** doublings)
    out = np.zeros((m1, m2), dtype=data.dtype)
    out[:n1, :n2] = data
    return out
