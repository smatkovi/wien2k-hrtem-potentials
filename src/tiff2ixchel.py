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
    --dim        NX NY, replaces the dim of the cfg; it must equal dim in the
                 simulation cfg given to `ixchel main -s`
    --drop-edges only for foreign TIFFs that repeat their first row and column;
                 the scripts of this repository never write such files

Layout of this file: the CORE comes first, tile_fourier, which turns one unit
cell into the supercell on the ixchel grid.  After the core: the ixchel cfg
and its checksum, a check of the input and the command line.

Dependencies: numpy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ixchel_tiff import read_ixchel_tiff, write_ixchel_tiff     # noqa: E402

read_potential_tiff = read_ixchel_tiff      # former name, still imported by check scripts


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

    # every kept coefficient goes to its place in the spectrum of the supercell grid; transform back
    target = np.zeros((Ny, Nx), dtype=complex)
    np.add.at(target, (KY[keep] % Ny, KX[keep] % Nx), coef[keep])
    out = np.fft.ifft2(target) * (Nx * Ny)
    imag = float(np.max(np.abs(out.imag)))
    if imag > 1e-6 * max(float(np.max(np.abs(out.real))), 1e-30):
        print(f"    WARNING: imaginary residue {imag:.3g} after resampling")
    dropped = 1.0 - float(np.sum(np.abs(coef[keep])**2) / np.sum(np.abs(coef)**2))
    return out.real, dropped


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: the ixchel cfg and its checksum, a check, command line
# ═══════════════════════════════════════════════════════════════

# ── file format: the ixchel cfg (only what the tiling needs) and its checksum ──

def parse_cfg(cfg_path):
    """Return dim (None if absent), NCELL (ixchel default 1×1) and the unique
    layer filenames of an ixchel cfg (libconfig: '=' or ':', [...] or (...))."""
    text = re.sub(r'#[^\n]*', '', Path(cfg_path).read_text())          # the cfg without its comments

    def pair(key):
        """The first two integers of an entry like 'dim = [ 1008, 1008];' or 'NCELL = [ 14, 8, 1];'
        (None if the cfg has no such entry)."""
        m = re.search(rf'\b{key}\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', text)
        return (int(m.group(1)), int(m.group(2))) if m else None

    dim = pair('dim')
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
    """POSIX cksum (CRC-32, polynomial 0x04C11DB7, length appended) — identical
    to Checksums::cksum in ixchel and to the coreutils `cksum` command."""
    data = Path(path).read_bytes()
    crc = 0
    for b in data:                                  # 1. every byte of the file
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CKSUM_TABLE[((crc >> 24) ^ b) & 0xFF]
    n = len(data)
    while n > 0:                                    # 2. then the length of the file, lowest byte first
        crc = ((crc << 8) & 0xFFFFFFFF) ^ _CKSUM_TABLE[((crc >> 24) ^ (n & 0xFF)) & 0xFF]
        n >>= 8
    return (~crc) & 0xFFFFFFFF                      # 3. all 32 bits inverted


def _make_cksum_table():
    """The table of the CRC above: one 32-bit entry for each of the 256 byte values."""
    table = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if (c & 0x80000000) else (c << 1)
        table.append(c & 0xFFFFFFFF)
    return table


_CKSUM_TABLE = _make_cksum_table()


# ── check of the input ─────────────────────────────────────────

def has_duplicate_edges(uc):
    """True if the last row and column repeat the first ones (XSF general-grid
    convention: n points span the period inclusively)."""
    tol = 1e-6 * max(float(np.ptp(uc)), 1e-30)
    return (np.max(np.abs(uc[0, :] - uc[-1, :])) <= tol and
            np.max(np.abs(uc[:, 0] - uc[:, -1])) <= tol)


# ── command line ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Tile single-unit-cell potential TIFFs onto the ixchel dim grid')
    ap.add_argument('--cfg', required=True, help='ixchel crystal-structure cfg (dim, NCELL, layer filenames)')
    ap.add_argument('-i', '--input', required=True, help='directory with one-unit-cell layer TIFFs')
    ap.add_argument('-o', '--output', required=True, help='output directory for ixchel-ready TIFFs')
    ap.add_argument('--dim', type=int, nargs=2, metavar=('NX', 'NY'),
                    help='override dim (must equal dim in simulation.cfg)')
    ap.add_argument('--drop-edges', action='store_true',
                    help='drop the last row and column; only for TIFFs that repeat their first '
                         'row/column (XSF general grid) — the scripts in this repository write '
                         'periodic grids')
    args = ap.parse_args()

    dim, ncell, filenames = parse_cfg(args.cfg)
    if args.dim:
        dim = tuple(args.dim)
    if dim is None:
        sys.exit(f"Error: no 'dim' in {args.cfg} and no --dim given (ixchel main needs dim)")
    cksum = posix_cksum(args.cfg)
    print(f"cfg {args.cfg}: dim = {dim[0]} x {dim[1]}, NCELL = {ncell[0]} x {ncell[1]}, cksum = {cksum}")
    if Path(args.input).resolve() == Path(args.output).resolve():
        sys.exit("Error: input and output directory must differ")

    for fn in filenames:
        src = Path(args.input) / fn
        if not src.exists():
            print(f"  {fn}: missing in {args.input}, skipped")
            continue
        uc = read_ixchel_tiff(src)
        note = ''
        if uc.shape == (dim[1], dim[0]) and ncell != (1, 1):
            print(f"  WARNING: {fn} already has the ixchel dim shape {uc.shape}; "
                  f"it is still treated as ONE unit cell")
        if args.drop_edges:
            if not has_duplicate_edges(uc):
                print(f"  WARNING: {fn}: --drop-edges given, but the last row/column do not "
                      f"repeat the first ones")
            uc = uc[:-1, :-1]
            note = ' (last row/column dropped)'
        elif has_duplicate_edges(uc):
            print(f"  NOTE: {fn}: last row/column equal the first ones (flat edges, or an XSF "
                  f"general grid); kept — pass --drop-edges only for general grids")
        out, dropped = tile_fourier(uc, ncell, dim)
        dst = Path(args.output) / fn
        write_ixchel_tiff(dst, out, cksum)
        print(f"  {fn}: unit cell {uc.shape[1]} x {uc.shape[0]} px{note} -> {dim[0]} x {dim[1]} px, "
              f"range [{out.min():.4g}, {out.max():.4g}] 1/Å, "
              f"power above Nyquist dropped {100*dropped:.2e} %  -> {dst}")


if __name__ == '__main__':
    main()
