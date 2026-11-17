#!/usr/bin/env python3
"""
ixchel_tiff.py — read and write TIFF files in the layout of ixchel

ixchel stores every 2D array (layer potential atompot<label>.tif, transmission
function trans_<label>.tif, wave function) as a TIFF with three pages:

    page 0   RGB preview (uint8), for image viewers only
    page 1   the data (float32), rows = y, columns = x; potentials in 1/Å
             (complex arrays: every row holds the real parts followed by the
             imaginary parts, see check_transmission.py)
    page 2   two uint32 checksums [crystal-structure cfg, simulation cfg];
             ixchel warns about a foreign crystal structure if the first one
             is not the POSIX cksum of its cfg (tiff2ixchel.py sets it)

This module is imported by the scripts that read or write such files
(vcoul2tiff_ulm.py, tiff2ixchel.py, check_transmission.py):

    from ixchel_tiff import read_ixchel_tiff, write_ixchel_tiff

Dependencies: numpy, tifffile
"""

from pathlib import Path

import numpy as np
import tifffile


def read_ixchel_tiff(path):
    """The data page of an ixchel TIFF (page 1, float32) as float64, shape [ny, nx]."""
    return tifffile.imread(str(path), key=1).astype(np.float64)


def write_ixchel_tiff(filename, data_yx, cksum_cry=0):
    """Write data_yx, shape [ny, nx], as an ixchel TIFF.  cksum_cry is the POSIX cksum of the
    crystal-structure cfg (0: not set).  Returns (vmin, vmax), the range of the stored data."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    data = data_yx.astype(np.float32)
    vmin, vmax = float(data.min()), float(data.max())
    if vmax == vmin:
        vmax = vmin + 1e-6                               # flat data: no division by zero in the preview
    preview = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)
    meta = np.array([[cksum_cry, 0]], dtype=np.uint32)   # [crystal_struct, simulation]
    with tifffile.TiffWriter(filename) as tfw:
        tfw.write(preview_rgb, photometric='rgb',        contiguous=False)
        tfw.write(data,        photometric='minisblack', contiguous=False)
        tfw.write(meta,        photometric='minisblack', contiguous=False)
    return vmin, vmax
