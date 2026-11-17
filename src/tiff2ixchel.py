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

and the script stops if they do not.  Choose the grid of lapw5grid.py
(--grid N+1 for N points per unit cell), NCELL and dim accordingly.
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


# ── file format: the ixchel cfg (only what is needed here) and its checksum ──

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
    ap = argparse.ArgumentParser(description='Repeat single-unit-cell potential TIFFs over the ixchel supercell')
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
        uc = read_ixchel_tiff(src)                      # one unit cell, rows = y, columns = x
        ny, nx = uc.shape

        # the copies must fill the ixchel grid exactly
        if (nx * ncell[0], ny * ncell[1]) != dim:
            sys.exit(f"Error: {fn}: {nx} x {ny} points per unit cell times NCELL = {ncell[0]} x {ncell[1]} "
                     f"gives {nx * ncell[0]} x {ny * ncell[1]} pixels, but dim = {dim[0]} x {dim[1]}.\n"
                     f"  Choose the lapw5grid.py grid, NCELL and dim so that points x NCELL = dim.")

        out = np.tile(uc, (ncell[1], ncell[0]))         # NCELL_y copies downwards, NCELL_x copies across
        dst = Path(args.output) / fn
        write_ixchel_tiff(dst, out, cksum)
        print(f"  {fn}: unit cell {nx} x {ny} px, {ncell[0]} x {ncell[1]} copies -> {dim[0]} x {dim[1]} px, "
              f"range [{out.min():.4g}, {out.max():.4g}] 1/Å  -> {dst}")


if __name__ == '__main__':
    main()
