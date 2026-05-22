"""
io_bruker.py
------------
Read fully-sampled Bruker/TopSpin data so that, once you have real spectrometer
access, the same pipeline runs on actual HSQC FIDs.

Two entry points:
- read_fid()   : raw time-domain (the 'ser' file) for retrospective
                 undersampling. This is what you want for schedule benchmarking.
- read_pdata() : an already-processed spectrum (pdata/N) if you only want to
                 read what TopSpin produced.

Requires nmrglue (pip install nmrglue). Imported lazily so the rest of the
package works without it for synthetic-data development.
"""

from __future__ import annotations

import numpy as np


def _require_nmrglue():
    try:
        import nmrglue as ng  # noqa: F401
        return ng
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "nmrglue is required to read Bruker data. Install with "
            "`pip install nmrglue`. (Synthetic-data development needs no "
            "external NMR libraries.)"
        ) from e


def read_fid(expdir: str) -> tuple[dict, np.ndarray]:
    """Read a raw 2D Bruker FID (the 'ser' file) from an experiment folder.

    Parameters
    ----------
    expdir : str
        Path to the experiment number folder, e.g. ".../sample/10".

    Returns
    -------
    dic : dict   Bruker parameter dictionary.
    fid : np.ndarray, complex, shape (n1, n2)
        Time-domain data with t1 along axis 0. NOTE: for a true NUS dataset
        on disk the 'ser' file is already non-uniform; for retrospective
        benchmarking you want a FULLY sampled experiment here.
    """
    ng = _require_nmrglue()
    dic, data = ng.bruker.read(expdir)
    return dic, data


def read_pdata(pdata_dir: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Read an already-processed 2D spectrum from a pdata/N folder.

    Returns
    -------
    dic, spec, ppm_f1, ppm_f2
        spec is the real 2D spectrum; ppm axes are descending (Bruker order).
    """
    ng = _require_nmrglue()
    dic, data = ng.bruker.read_pdata(pdata_dir)

    udic = ng.bruker.guess_udic(dic, data)
    uc_f1 = ng.fileiobase.uc_from_udic(udic, dim=0)
    uc_f2 = ng.fileiobase.uc_from_udic(udic, dim=1)
    ppm_f1 = uc_f1.ppm_scale()
    ppm_f2 = uc_f2.ppm_scale()
    return dic, np.asarray(data), ppm_f1, ppm_f2
