#!/usr/bin/env python3
"""
tiff2ixchel.py — repeat single-unit-cell potential TIFFs over the ixchel supercell

vcoul2tiff_ulm.py writes the projected potential of ONE unit cell per layer,
sampled on the WIEN2k grid.  `ixchel main` instead expects every layer file to
cover the whole supercell (NCELL[0] x NCELL[1] unit cells) on exactly `dim`
pixels and aborts with "ERROR: Dimension mismatch" otherwise (ixchel
src/ixchelfun.cpp).

This tool copies the unit cell NCELL times side by side.  No value is changed:
one ixchel pixel is one point of the WIEN2k grid.  The grids therefore have to
fit,

    dim = NCELL x (points per unit cell),    graphene: 1008 = 14 x 72 = 8 x 126

Choose the grid of lapw5grid.py (--grid N+1 for N points per unit cell), NCELL
and dim accordingly.  If they do not fit, the output has the wrong size and
`ixchel main` stops with "Dimension mismatch".
Pixel (0, 0) stays at the unit-cell origin, as in ixchel's own atompot.

The checksum page of the output is left at zero.  `ixchel main` therefore
prints, for every layer, the warning that the file "was created with a
different crystal structure config file"; the calculation is not affected.
The unit is unchanged (Kirkland units, 1/Å).

Usage:
    python3 tiff2ixchel.py --cfg Graphene_ixchel_001.cfg -i ulm_DFT_001 -o sim_DFT
    (then run `ixchel main -c Graphene_ixchel_001.cfg -s simulation.cfg`
     in a directory where the layer files from -o are found)

Arguments and where they come from:
    --cfg        the crystal-structure cfg that `ixchel main -c` will get: written
                 by crystruct --ixchel, with dim and NCELL set to the values of
                 the run.  Read from it: NCELL and the file name of every layer
                 (atompot<label>.tif).
    -i           the directory with the one-unit-cell layer TIFFs, i.e. the -o
                 directory of vcoul2tiff_ulm.py (ulm_DFT_001 in the README).  The
                 name is free; the files in it carry the layer file names of the cfg.
    -o           the directory for the ixchel-ready TIFFs (same file names).  Use
                 the directory in which `ixchel main` runs, ixchel looks for the
                 layer files there.

Dependencies: numpy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ixchel_tiff import read_ixchel_tiff, write_ixchel_tiff     # noqa: E402


# ── file format: the ixchel cfg (only what is needed here) ──

def parse_cfg(cfg_path):
    """Return NCELL (ixchel default 1×1) and the layer file names of an ixchel cfg
    (libconfig: '=' or ':', [...] or (...))."""
    text = re.sub(r'#[^\n]*', '', Path(cfg_path).read_text())          # the cfg without its comments

    # the entry 'NCELL = [ 14, 8, 1];': its first two numbers
    m = re.search(r'\bNCELL\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', text)
    ncell = (int(m.group(1)), int(m.group(2))) if m else (1, 1)

    # the entries 'filename = "atompota.tif";', one per layer
    filenames = re.findall(r'\bfilename\s*[=:]\s*"([^"]+)"', text)
    if not filenames:
        sys.exit(f"Error: no layer 'filename' entries in {cfg_path}")
    return ncell, filenames


# ── command line ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Repeat single-unit-cell potential TIFFs over the ixchel supercell')
    ap.add_argument('--cfg', required=True, help='ixchel crystal-structure cfg (NCELL, layer filenames)')
    ap.add_argument('-i', '--input', required=True, help='directory with one-unit-cell layer TIFFs')
    ap.add_argument('-o', '--output', required=True, help='output directory for ixchel-ready TIFFs')
    args = ap.parse_args()

    ncell, filenames = parse_cfg(args.cfg)
    if Path(args.input).resolve() == Path(args.output).resolve():
        sys.exit("Error: input and output directory must differ")

    for fn in filenames:
        uc = read_ixchel_tiff(Path(args.input) / fn)    # one unit cell, rows = y, columns = x
        out = np.tile(uc, (ncell[1], ncell[0]))         # NCELL_y copies downwards, NCELL_x copies across
        dst = Path(args.output) / fn
        write_ixchel_tiff(dst, out)
        print(f"  {fn}: unit cell {uc.shape[1]} x {uc.shape[0]} px, {ncell[0]} x {ncell[1]} copies -> "
              f"{out.shape[1]} x {out.shape[0]} px, range [{out.min():.4g}, {out.max():.4g}] 1/Å  -> {dst}")


if __name__ == '__main__':
    main()
