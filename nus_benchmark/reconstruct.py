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
    """STUB: scripted NMRPipe + hmsIST reconstruction.

    This is where large-scale ML data generation will happen, because it runs
    headless from the command line. Filling this in requires NMRPipe and the
    hmsIST tools installed and on PATH. The intended flow:

        1. Write the masked FID + sampling schedule to NMRPipe format
           (nmrglue.pipe.write, plus a .nus schedule file).
        2. Shell out to the IST reconstruction script (subprocess.run).
        3. Read the reconstructed spectrum back (nmrglue.pipe.read).
        4. Return the real part as a 2D array.

    Left as a stub deliberately: it depends on your local install, and the
    pipeline is fully testable without it via ZeroFillFFT.
    """

    def __init__(self, nmrpipe_bin: str = "nmrPipe", workdir: str | None = None):
        self.nmrpipe_bin = nmrpipe_bin
        self.workdir = workdir

    def reconstruct(self, fid: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError(
            "NmrPipeIST is a stub. Implement the NMRPipe/hmsIST shell-out here "
            "once you have those tools installed. See the docstring for the "
            "intended four-step flow. Until then, use ZeroFillFFT."
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
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
