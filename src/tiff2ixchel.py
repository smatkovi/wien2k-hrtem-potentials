#!/usr/bin/env python3
"""
tiff2ixchel.py — put single-unit-cell potential TIFFs on the ixchel simulation grid

vcoul2tiff_ulm.py writes the projected potential of ONE unit cell per layer,
sampled at the resolution of the WIEN2k grid.
`ixchel main` instead expects every layer file to cover the whole supercell
(NCELL[0] x NCELL[1] unit cells) on exactly `dim` pixels and aborts with
"ERROR: Dimension mismatch" otherwise (ixchel src/ixchelfun.cpp).

This tool tiles each unit cell NCELL times and resamples it onto the `dim`
grid by Fourier interpolation.  Because the potential is periodic, this is
exact for all spatial frequencies below the Nyquist limit of both grids;
higher frequencies are dropped (ixchel band-limits the potential to 2/3 of
its Nyquist frequency anyway before building the transmission function).
Pixel (0, 0) stays at the unit-cell origin, as in ixchel's own atompot.

Page 2 of the output carries the POSIX cksum of the crystal-structure cfg,
the same value `ixchel atompot` writes, so ixchel does not warn about a
foreign crystal structure.  The unit is unchanged (Kirkland units, 1/Å).

Usage:
    python3 tiff2ixchel.py --cfg Graphene_ixchel_001.cfg -i ulm_DFT_001 -o sim_DFT
    (then run `ixchel main -c Graphene_ixchel_001.cfg -s simulation.cfg`
     in a directory where the layer files from -o are found)

Arguments and where they come from:
    --cfg        the crystal-structure cfg that `ixchel main -c` will get: written
                 by crystruct --ixchel, with dim and NCELL set to the values of
                 the run.  Read from it: dim, NCELL and the file name of every
                 layer (atompot<label>.tif); its checksum goes into the output.
    -i           the directory with the one-unit-cell layer TIFFs, i.e. the -o
                 directory of vcoul2tiff_ulm.py (ulm_DFT_001 in the README).  The
                 name is free; the files in it carry the layer file names of the cfg.
    -o           the directory for the ixchel-ready TIFFs (same file names).  Use
                 the directory in which `ixchel main` runs, ixchel looks for the
                 layer files there.

Layout of this file: the CORE comes first, tile_fourier, which turns one unit
cell into the supercell on the ixchel grid.  After the core: the ixchel cfg,
its checksum and the command line.

Dependencies: numpy, tifffile; the command `cksum` (coreutils)
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ixchel_tiff import read_ixchel_tiff, write_ixchel_tiff     # noqa: E402


# ═══════════════════════════════════════════════════════════════
# CORE: one unit cell -> the ixchel supercell grid
#
# The potential is periodic, so it is a sum of plane waves.  tile_fourier takes
# the waves of the unit-cell image and evaluates them on the pixels of the
# supercell (NCELL cells on dim pixels): tiling and resampling in one step.
# ═══════════════════════════════════════════════════════════════

def tile_fourier(uc, ncell, dim):
    """
    uc    : [ny_uc, nx_uc] one period, pixel (0,0) at the cell origin
    ncell : (NCx, NCy) unit cells in the ixchel supercell
    dim   : (Nx, Ny) ixchel grid
    returns the supercell sampled on the ixchel grid, [Ny, Nx], and the fraction of the
    spectral power that had to be dropped because it lies at or above a Nyquist limit
    """
    ny_uc, nx_uc = uc.shape
    ncx, ncy = ncell
    Nx, Ny = dim

    # Fourier coefficients of one unit cell.  kx, ky count the periods per unit cell; the same wave has
    # kx * ncx and ky * ncy periods across the supercell of ncx x ncy cells.
    coef = np.fft.fft2(uc) / uc.size                       # c[ky, kx]
    kx = np.fft.fftfreq(nx_uc, d=1.0 / nx_uc).astype(int)  # integer UC harmonics
    ky = np.fft.fftfreq(ny_uc, d=1.0 / ny_uc).astype(int)
    KX, KY = np.meshgrid(kx * ncx, ky * ncy)               # supercell harmonics

    # keep strictly below the Nyquist frequency of both grids so that +k and -k
    # are always kept or dropped together (result stays real)
    keep = ((2 * np.abs(kx)[None, :] < nx_uc) & (2 * np.abs(ky)[:, None] < ny_uc) &
            (2 * np.abs(KX) < Nx) & (2 * np.abs(KY) < Ny))

    # every kept coefficient goes to its own place in the spectrum of the supercell grid; transform back
    target = np.zeros((Ny, Nx), dtype=complex)
    target[KY[keep] % Ny, KX[keep] % Nx] = coef[keep]
    out = np.fft.ifft2(target) * (Nx * Ny)
    imag = float(np.max(np.abs(out.imag)))
    if imag > 1e-6 * max(float(np.max(np.abs(out.real))), 1e-30):
        print(f"    WARNING: imaginary residue {imag:.3g} after resampling")
    dropped = 1.0 - float(np.sum(np.abs(coef[keep])**2) / np.sum(np.abs(coef)**2))
    return out.real, dropped


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: the ixchel cfg, its checksum, command line
# ═══════════════════════════════════════════════════════════════

# ── file format: the ixchel cfg (only what the tiling needs) and its checksum ──

def parse_cfg(cfg_path):
    """Return dim, NCELL (ixchel default 1×1) and the unique layer filenames
    of an ixchel cfg (libconfig: '=' or ':', [...] or (...))."""
    text = re.sub(r'#[^\n]*', '', Path(cfg_path).read_text())          # the cfg without its comments

    def pair(key):
        """The first two integers of an entry like 'dim = [ 1008, 1008];' or 'NCELL = [ 14, 8, 1];'
        (None if the cfg has no such entry)."""
        m = re.search(rf'\b{key}\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', text)
        return (int(m.group(1)), int(m.group(2))) if m else None

    dim = pair('dim')
    if dim is None:
        sys.exit(f"Error: no 'dim' in {cfg_path} (ixchel main needs dim)")
    ncell = pair('NCELL') or (1, 1)

    # the entries 'filename = "atompota.tif";' of the layers: every name once, in the order of the file
    filenames = []
    for name in re.findall(r'\bfilename\s*[=:]\s*"([^"]+)"', text):
        if name not in filenames:
            filenames.append(name)
    if not filenames:
        sys.exit(f"Error: no layer 'filename' entries in {cfg_path}")
    return dim, ncell, filenames


def posix_cksum(path):
    """The checksum of a file as the command `cksum` prints it (POSIX CRC-32).  ixchel computes the
    same number for its cfg (Checksums::cksum) and compares it with the one stored in a layer file."""
    result = subprocess.run(['cksum', str(path)], capture_output=True, text=True, check=True)
    return int(result.stdout.split()[0])            # the output is "<checksum> <size in bytes> <file name>"


# ── command line ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Tile single-unit-cell potential TIFFs onto the ixchel dim grid')
    ap.add_argument('--cfg', required=True, help='ixchel crystal-structure cfg (dim, NCELL, layer filenames)')
    ap.add_argument('-i', '--input', required=True, help='directory with one-unit-cell layer TIFFs')
    ap.add_argument('-o', '--output', required=True, help='output directory for ixchel-ready TIFFs')
    args = ap.parse_args()

    dim, ncell, filenames = parse_cfg(args.cfg)
    cksum = posix_cksum(args.cfg)
    print(f"cfg {args.cfg}: dim = {dim[0]} x {dim[1]}, NCELL = {ncell[0]} x {ncell[1]}, cksum = {cksum}")
    if Path(args.input).resolve() == Path(args.output).resolve():
        sys.exit("Error: input and output directory must differ")

    for fn in filenames:
        src = Path(args.input) / fn
        if not src.exists():
            print(f"  {fn}: missing in {args.input}, skipped")
            continue
        uc = read_ixchel_tiff(src)                      # one unit cell, [ny, nx]
        out, dropped = tile_fourier(uc, ncell, dim)
        dst = Path(args.output) / fn
        write_ixchel_tiff(dst, out, cksum)
        print(f"  {fn}: unit cell {uc.shape[1]} x {uc.shape[0]} px -> {dim[0]} x {dim[1]} px, "
              f"range [{out.min():.4g}, {out.max():.4g}] 1/Å, "
              f"power above Nyquist dropped {100*dropped:.2e} %  -> {dst}")


if __name__ == '__main__':
    main()
