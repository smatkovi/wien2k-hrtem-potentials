#!/usr/bin/env python3
"""
tiff2ixchel.py — put single-unit-cell potential TIFFs on the ixchel simulation grid

vcoul2tiff.py, vcoul2tiff_ulm.py and xsf2tiff_dwFactor.py write the projected
potential of ONE unit cell per layer, sampled at their own resolution.
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
    python3 tiff2ixchel.py --cfg Graphene_ixchel_001.cfg -i wien2k_001 -o ixchel_wien2k_001
    (then run `ixchel main -c Graphene_ixchel_001.cfg -s simulation.cfg`
     in a directory where the layer files from -o are found)

Dependencies: numpy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile


# ═══════════════════════════════════════════════════════════════
# cfg parsing (only what the tiling needs)
# ═══════════════════════════════════════════════════════════════

def parse_cfg(cfg_path):
    """Return dim (None if absent), NCELL (ixchel default 1×1) and the unique
    layer filenames of an ixchel cfg (libconfig: '=' or ':', [...] or (...))."""
    text = Path(cfg_path).read_text()
    text_nc = re.sub(r'#[^\n]*', '', text)          # drop comments

    dim_m = re.search(r'\bdim\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', text_nc)
    ncell_m = re.search(r'\bNCELL\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', text_nc)
    dim = (int(dim_m.group(1)), int(dim_m.group(2))) if dim_m else None
    ncell = (int(ncell_m.group(1)), int(ncell_m.group(2))) if ncell_m else (1, 1)

    filenames = []
    for m in re.finditer(r'\bfilename\s*[=:]\s*"([^"]+)"', text_nc):
        if m.group(1) not in filenames:
            filenames.append(m.group(1))
    if not filenames:
        sys.exit(f"Error: no layer 'filename' entries in {cfg_path}")
    return dim, ncell, filenames


def posix_cksum(path):
    """POSIX cksum (CRC-32, polynomial 0x04C11DB7, length appended) — identical
    to Checksums::cksum in ixchel and to the coreutils `cksum` command."""
    data = Path(path).read_bytes()
    crc = 0
    table = _CKSUM_TABLE
    for b in data:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ b) & 0xFF]
    n = len(data)
    while n > 0:
        crc = ((crc << 8) & 0xFFFFFFFF) ^ table[((crc >> 24) ^ (n & 0xFF)) & 0xFF]
        n >>= 8
    return (~crc) & 0xFFFFFFFF


def _make_cksum_table():
    table = []
    for i in range(256):
        c = i << 24
        for _ in range(8):
            c = ((c << 1) ^ 0x04C11DB7) if (c & 0x80000000) else (c << 1)
        table.append(c & 0xFFFFFFFF)
    return table


_CKSUM_TABLE = _make_cksum_table()


# ═══════════════════════════════════════════════════════════════
# TIFF I/O (ixchel layout: page 0 RGB preview, page 1 float32, page 2 checksums)
# ═══════════════════════════════════════════════════════════════

def read_potential_tiff(path):
    with tifffile.TiffFile(str(path)) as tif:
        for page in tif.pages:
            if page.dtype == np.float32 and page.ndim == 2:
                return page.asarray().astype(np.float64)
    sys.exit(f"Error: no float32 data page in {path}")


def write_ixchel_tiff(filename, data_yx, cksum_cry):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    data = data_yx.astype(np.float32)
    vmin, vmax = float(data.min()), float(data.max())
    if vmax == vmin:
        vmax = vmin + 1e-6
    preview = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)
    meta = np.array([[cksum_cry, 0]], dtype=np.uint32)   # [crystal_struct, simulation]
    with tifffile.TiffWriter(filename) as tfw:
        tfw.write(preview_rgb, photometric='rgb',        contiguous=False)
        tfw.write(data,        photometric='minisblack', contiguous=False)
        tfw.write(meta,        photometric='minisblack', contiguous=False)


# ═══════════════════════════════════════════════════════════════
# Tiling + Fourier resampling
# ═══════════════════════════════════════════════════════════════

def has_duplicate_edges(uc):
    """True if the last row and column repeat the first ones (XSF general-grid
    convention: n points span the period inclusively)."""
    tol = 1e-6 * max(float(np.ptp(uc)), 1e-30)
    return (np.max(np.abs(uc[0, :] - uc[-1, :])) <= tol and
            np.max(np.abs(uc[:, 0] - uc[:, -1])) <= tol)


def tile_fourier(uc, ncell, dim):
    """
    uc    : [ny_uc, nx_uc] one period, pixel (0,0) at the cell origin
    ncell : (NCx, NCy) unit cells in the ixchel supercell
    dim   : (Nx, Ny) ixchel grid
    returns [Ny, Nx] supercell sampled on the ixchel grid
    """
    ny_uc, nx_uc = uc.shape
    ncx, ncy = ncell
    Nx, Ny = dim

    coef = np.fft.fft2(uc) / uc.size                       # c[ky, kx]
    kx = np.fft.fftfreq(nx_uc, d=1.0 / nx_uc).astype(int)  # integer UC harmonics
    ky = np.fft.fftfreq(ny_uc, d=1.0 / ny_uc).astype(int)
    KX, KY = np.meshgrid(kx * ncx, ky * ncy)               # supercell harmonics

    # keep strictly below the Nyquist frequency of both grids so that +k and -k
    # are always kept or dropped together (result stays real)
    keep = ((2 * np.abs(kx)[None, :] < nx_uc) & (2 * np.abs(ky)[:, None] < ny_uc) &
            (2 * np.abs(KX) < Nx) & (2 * np.abs(KY) < Ny))

    target = np.zeros((Ny, Nx), dtype=complex)
    np.add.at(target, (KY[keep] % Ny, KX[keep] % Nx), coef[keep])
    out = np.fft.ifft2(target) * (Nx * Ny)
    imag = float(np.max(np.abs(out.imag)))
    if imag > 1e-6 * max(float(np.max(np.abs(out.real))), 1e-30):
        print(f"    WARNING: imaginary residue {imag:.3g} after resampling")
    dropped = 1.0 - float(np.sum(np.abs(coef[keep])**2) / np.sum(np.abs(coef)**2))
    return out.real, dropped


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Tile single-unit-cell potential TIFFs onto the ixchel dim grid')
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
    print(f"cfg {args.cfg}: dim = {dim[0]} x {dim[1]}, NCELL = {ncell[0]} x {ncell[1]}, "
          f"cksum = {cksum}")
    if Path(args.input).resolve() == Path(args.output).resolve():
        sys.exit("Error: input and output directory must differ")

    for fn in filenames:
        src = Path(args.input) / fn
        if not src.exists():
            print(f"  {fn}: missing in {args.input}, skipped")
            continue
        uc = read_potential_tiff(src)
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
