#!/usr/bin/env python3
"""
lapw5grid.py — evaluate WIEN2k clm-type files on regular 3D grids with lapw5

WIEN2k stores densities and potentials (case.clmsum, case.vcoul, ...) as
lattice harmonics inside the muffin-tin spheres plus a star expansion in the
interstitial.  Evaluating that representation correctly needs the stars, the
local rotation matrices, the symmetry operations and the lattice centring;
lapw5 does all of this.  This module drives the standard 2D mode of lapw5c
plane by plane (like wien2venus.py of Kurasch et al. 2011) and assembles the
planes into a 3D array.  The complex build lapw5c is always used: it reads
both columns of the plane-wave coefficients, which structures without
inversion need (the real lapw5 silently drops the imaginary parts), and it
gives bit-identical results for centrosymmetric structures.

Why not 3ddens: WIEN2k 24.1 3ddens adds the centring translations only for
F, B, CXZ and R lattices (SRC_3ddens/3ddens.F:403), so for CXY and CYZ
lattices (e.g. graphene) the sphere contributions of the centring atoms are
missing.  lapw5 has no such restriction.

Values are written by lapw5 with 9 significant digits (5ES16.8).  With
`sub` the second file is subtracted inside lapw5 in double precision before
output, so for vcoul_DFT - vcoul_IAM the -2Z/r nuclear terms cancel exactly
and the difference is finite everywhere.

Units: ATU, i.e. Ry for potentials and e/bohr^3 for densities.  norm='val'
for potentials and valence densities, norm='tot' for case.clmsum (lapw5 then
divides the L=0 component by sqrt(4 pi), as 3ddens does for *sum files).

Command line (XSF in the 3ddens general-grid convention: n points per axis
spanning the conventional cell including both end planes, x fastest, Å;
one period therefore has n-1 points, choose n-1 so that the nuclei lie on
grid points, e.g. 75 127 451 = 74 x 126 x 450 periodic for graphene):

    python3 lapw5grid.py --struct case.struct --clm case.vcoul --grid 75 127 451 -o case_vcoul.xsf
    python3 lapw5grid.py --struct case.struct --clm case.vcoul_DFT --sub case.vcoul_IAM --grid ... -o dV.xsf
    python3 lapw5grid.py --struct case.struct --clm case.clmsum --norm tot --grid ... -o rho.xsf

Other cells (--edges, --origin): by default the grid spans the conventional
cell.  --edges gives the three grid edge vectors and --origin the grid origin
(default 0 0 0; fractions like 1/2 are accepted), both in fractional
coordinates of the conventional cell, so that e.g. a hexagonal (H) case can be
sampled on its rectangular (orthohexagonal) cell a1, a1 + 2 a2, the cell that
crystruct --ixchel "1 0 0  1 2 0  0 0 1" writes:

    python3 lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul --grid 75 127 451 \\
        --edges 1 0 0  1 2 0  0 0 1 -o Graphene_vcoul_ongrid.xsf

The edges must be lattice translations, otherwise the last planes are not
periodic images of the first (a warning is printed).  The XSF then lists the
grid cell as the crystal cell.

Requires lapw5c on PATH ($WIENROOT) and numpy.
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np

BOHR = 0.529177210903   # Bohr -> Å

# centring translations of the WIEN2k lattice types (fractions of the conventional cell)
CENTRING = {'P': [(0, 0, 0)], 'H': [(0, 0, 0)], 'B': [(0, 0, 0), (0.5, 0.5, 0.5)],
            'F': [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
            'CXY': [(0, 0, 0), (0.5, 0.5, 0)], 'CXZ': [(0, 0, 0), (0.5, 0, 0.5)], 'CYZ': [(0, 0, 0), (0, 0.5, 0.5)]}

# file list of one lapw5 run (Fortran unit, file): 5 input, 6 log, 8 struct, 9 the clm file,
# 11 the clm file to subtract, 21 the values on the plane
DEF = """\
5 ,'{tag}.in5','old','formatted',0
6 ,'{tag}.output5','unknown','formatted',0
8 ,'case.struct','old','formatted',0
9 ,'clm_a','old','formatted',0
10,'{tag}.tmp','unknown','unformatted',0
11,'{unit11}','unknown','formatted',0
12,'{tag}.sigma','unknown','formatted',0
20,'{tag}.rho_onedim','unknown','formatted',0
21,'{tag}.rho','unknown','formatted',0
"""


# ── case.struct ────────────────────────────────────────────────

def read_struct(path):
    """Lattice type, lattice constants [bohr], angles [deg] and all atoms of
    the conventional cell as listed in case.struct (centring copies are NOT
    included; see centring_translations)."""
    lines = Path(path).read_text().splitlines()
    abc = [float(lines[3][i:i + 10]) for i in (0, 10, 20)]
    ang = []
    for i in (30, 40, 50):
        s = lines[3][i:i + 10].strip()
        ang.append(float(s) if s and float(s) != 0.0 else 90.0)      # blank or 0 means 90°
    atoms, i = [], 4
    while i < len(lines):
        if not lines[i].startswith('ATOM'):
            i += 1
            continue
        mult = int(lines[i + 1].split('MULT=')[1].split()[0])        # line i + 1: MULT= and ISPLIT=
        rows = [i, *range(i + 2, i + 1 + mult)]                      # first position, then the equivalent ones
        el = lines[i + 1 + mult]                                     # element line, after the positions
        atoms.append({'name': el[:10].strip(), 'Z': float(el.split('Z:')[1].split()[0]),
                      'positions': [_xyz(lines[j]) for j in rows]})
        i += 2 + mult
    return {'lattice': lines[1][:4].strip(), 'abc_bohr': abc, 'angles': ang, 'atoms': atoms}


def _xyz(line):
    s = line.split('X=')[1]
    return (float(s[:10]), float(s.split('Y=')[1][:10]), float(s.split('Z=')[1][:10]))


def centring_translations(lattice):
    for key, shifts in CENTRING.items():
        if lattice.upper().startswith(key):
            return list(shifts)
    raise NotImplementedError(f"lattice type '{lattice}' not supported")


def conventional_vectors_ang(struct):
    """Cartesian conventional lattice vectors [Å] (rows) in the WIEN2k/3ddens
    orientation.  Orthogonal cells and the hexagonal H lattice only."""
    a, b, c = (x * BOHR for x in struct['abc_bohr'])
    if struct['lattice'].upper().startswith('H'):
        return np.array([[a * math.sqrt(3) / 2, -a / 2, 0], [0, a, 0], [0, 0, c]])
    if all(abs(x - 90.0) < 1e-6 for x in struct['angles']):
        return np.diag([a, b, c])
    raise NotImplementedError("non-orthogonal cells other than H are not supported")


# ── lapw5 driver ───────────────────────────────────────────────

def _int_vector(frac3, max_den=10**6):
    """(IX, IY, IZ, IDV) with IX/IDV = frac3[0] etc. (exact when possible)."""
    fr = [x if isinstance(x, Fraction) else Fraction(x).limit_denominator(max_den) for x in frac3]
    den = math.lcm(*(f.denominator for f in fr))
    if den > 10**8:                                    # keep integers well inside int32
        den = max_den
        fr = [Fraction(round(float(f) * den), den) for f in fr]
    nums = [int(f * den) for f in fr]
    if max(abs(n) for n in nums + [den]) >= 2**31:
        raise ValueError(f"cannot represent {frac3} as lapw5 integers")
    return (*nums, den)


def check_clm_header(path):
    """Refuse vcoul files written without the r^2 factor (NONORM patch of lapw0);
    lapw5 assumes the standard normalisation."""
    with open(path) as f:
        f.readline()
        second = f.readline()
    if 'NORM OF' in second and '*r**2' not in second.replace(' ', ''):
        sys.exit(f"Error: {path} was written without the r^2 normalisation (NONORM); "
                 f"lapw5 cannot evaluate it. Re-run lapw0 without NONORM.")


class Lapw5Runner:
    """Evaluate clm_a (optionally clm_a - clm_b) on planes with lapw5."""

    def __init__(self, struct_path, clm_a, clm_b=None, norm='val', nshell=(2, 2, 2),
                 workdir=None, nproc=None, keep=False):
        self.norm = norm.upper()
        if self.norm not in ('VAL', 'TOT'):
            raise ValueError("norm must be 'val' or 'tot'")
        for p in [clm_a] + ([clm_b] if clm_b else []):
            if not Path(p).is_file() or Path(p).stat().st_size == 0:
                sys.exit(f"Error: {p} missing or empty (for case.vcoul: R2V must be set in case.in0)")
            check_clm_header(p)
        if shutil.which('lapw5c') is None:
            sys.exit("Error: lapw5c not found on PATH (source the WIEN2k environment)")
        self.sub = clm_b is not None
        self.nshell = nshell            # minimum; raised per plane if needed
        self.nproc = nproc or max(1, (os.cpu_count() or 2) - 2)
        self.keep = keep
        self._tmp = tempfile.mkdtemp(prefix='lapw5grid_') if workdir is None else None
        self.wd = Path(self._tmp or workdir)            # temporary unless a workdir is given
        self.wd.mkdir(parents=True, exist_ok=True)
        for name, src in (('case.struct', struct_path), ('clm_a', clm_a), ('clm_b', clm_b)):
            (self.wd / name).unlink(missing_ok=True)    # stale link of an earlier run
            if src:
                (self.wd / name).symlink_to(Path(src).resolve())
        self._counter = 0
        self._grid_calls = 0

    def close(self):
        if self._tmp and not self.keep:
            shutil.rmtree(self._tmp, ignore_errors=True)

    def plane(self, origin, xend, yend, npx, npy, tag=None):
        """Values on the npx x npy grid origin + i/(npx-1)(xend-origin) + j/(npy-1)(yend-origin)
        (fractional coordinates of the conventional cell).  Returns [npx, npy]."""
        if tag is None:
            self._counter += 1
            tag = f"p{self._counter:05d}"
        wd = self.wd
        # lapw5 looks for muffin-tin spheres only among the cell origins
        # -NSHELL..+NSHELL, so every corner of the plane must lie well inside
        o, x, y = (np.asarray(v, float) for v in (origin, xend, yend))
        reach = np.abs([o, x, y, x + y - o]).max(axis=0)            # farthest corner along each cell axis
        nshell = [max(int(s), math.ceil(r - 1e-9) + 1) for s, r in zip(self.nshell, reach)]
        in5 = ["%d %d %d %d" % _int_vector(v) for v in (origin, xend, yend)]
        in5 += ["%d %d %d" % tuple(nshell), "%d %d" % (npx, npy), "RHO SUB " if self.sub else "RHO     ",
                f"ATU {self.norm} NODEBUG", "NONORTHO"]
        (wd / f"{tag}.in5").write_text("\n".join(in5) + "\n")
        (wd / f"{tag}.def").write_text(DEF.format(tag=tag, unit11='clm_b' if self.sub else f"{tag}.none"))
        res = subprocess.run(['lapw5c', f"{tag}.def"], cwd=wd, capture_output=True, text=True,
                             env=dict(os.environ, OMP_NUM_THREADS='1'))
        rho = wd / f"{tag}.rho"
        if (res.returncode != 0 or not re.search(r'STOP\s+LAPW5C?\s+END', res.stdout + res.stderr)
                or not rho.exists()):
            out5 = wd / f"{tag}.output5"
            tail = out5.read_text(errors='replace')[-1500:] if out5.exists() else ''
            raise RuntimeError(f"lapw5c failed for plane {tag}:\n{res.stdout[-800:]}{res.stderr[-800:]}\n"
                               f"--- end of {tag}.output5 ---\n{tail}")
        with open(rho) as f:
            head = f.readline().split()
            vals = np.array(f.read().split(), dtype=np.float64)
        if int(head[0]) != npx or int(head[1]) != npy or vals.size != npx * npy:
            raise RuntimeError(f"lapw5 output {rho} has unexpected size {head[:2]} / {vals.size}")
        if not self.keep:
            for ext in ('in5', 'def', 'output5', 'tmp', 'sigma', 'rho_onedim', 'rho', 'none'):
                (wd / f"{tag}.{ext}").unlink(missing_ok=True)
        return vals.reshape((npx, npy))            # prtrh.f: x outer, y inner

    def grid3d(self, origin, edges, counts, progress=True):
        """Values at origin + i/(n0-1) e0 + j/(n1-1) e1 + k/(n2-1) e2 for all
        i < n0, j < n1, k < n2 (end points included).  Planes are taken normal to
        the axis with the fewest points and run in parallel.  Returns [n0, n1, n2]."""
        origin = np.asarray(origin, float)
        edges = [np.asarray(e, float) for e in edges]
        counts = [int(n) for n in counts]
        if min(counts) < 2:
            raise ValueError("every axis needs at least 2 points")
        k = int(np.argmin(counts))
        p, q = [ax for ax in range(3) if ax != k]
        nk = counts[k]
        self._grid_calls += 1
        call = self._grid_calls                 # file names unique across calls (--workdir)

        def job(m):
            o = origin + edges[k] * (m / (nk - 1))
            return self.plane(o, o + edges[p], o + edges[q], counts[p], counts[q], tag=f"g{call:03d}_{m:05d}")

        planes = []
        with ThreadPoolExecutor(max_workers=self.nproc) as ex:
            for plane in ex.map(job, range(nk)):                    # results arrive in the order of m
                planes.append(plane)
                if progress and (len(planes) % max(1, nk // 10) == 0 or len(planes) == nk):
                    print(f"    lapw5 planes {len(planes)}/{nk}", flush=True)
        return np.stack(planes, axis=k)


# ── XSF output (3ddens-compatible general grid) ────────────────

def is_lattice_translation(vec, lattice, tol=1e-6):
    """True if vec (fractional, conventional cell) is a translation of the lattice."""
    v = np.asarray(vec, float)
    return any(np.all(np.abs((v - np.array(t)) - np.round(v - np.array(t))) < tol)
               for t in centring_translations(lattice))


def atoms_in_cell(struct, origin, edges, tol=1e-6):
    """(Z, fractional position in the conventional cell) of all atoms inside the
    cell origin + u e0 + v e1 + w e2, 0 <= u, v, w < 1."""
    origin = np.asarray(origin, float)
    E = np.array(edges, float)
    Einv = np.linalg.inv(E)
    corners = np.array([origin + E.T @ np.array(c) for c in np.ndindex(2, 2, 2)])
    lo = np.floor(corners.min(axis=0)).astype(int) - 1
    hi = np.ceil(corners.max(axis=0)).astype(int) + 1
    out = []
    for at in struct['atoms']:
        for pos in at['positions']:
            for t in centring_translations(struct['lattice']):
                f0 = (np.array(pos) + np.array(t)) % 1.0
                for n in np.ndindex(*(hi - lo + 1)):
                    f = f0 + lo + np.array(n)
                    g = (f - origin) @ Einv
                    if np.all(g > -tol) and np.all(g < 1 - tol):
                        out.append((int(round(at['Z'])), f))
    return out


def write_xsf(path, data, struct, comment, origin=(0, 0, 0), edges=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    """origin, edges: grid origin and edge vectors in fractional coordinates of
    the conventional cell (default: the conventional cell itself)."""
    conv = conventional_vectors_ang(struct)
    cell = np.array(edges, float) @ conv               # Cartesian grid vectors [Å] (rows)
    o_cart = np.asarray(origin, float) @ conv
    atoms = [(z, f @ conv) for z, f in atoms_in_cell(struct, origin, edges)]
    nx, ny, nz = data.shape
    vec = "%16.8E%16.8E%16.8E\n"
    cell_txt = ''.join(vec % tuple(v) for v in cell)
    with open(path, 'w') as f:
        f.write(f"# XSF-File prepared by lapw5grid.py (values from lapw5)\n# {comment}\n")
        f.write(f"CRYSTAL\nPRIMVEC\n{cell_txt}CONVVEC\n{cell_txt}")
        f.write("PRIMCOORD\n%5d    1\n" % len(atoms))
        for z, r in atoms:
            f.write("%3d" % z + vec % tuple(r))
        f.write("BEGIN_BLOCK_DATAGRID3D\n lapw5grid\nDATAGRID_3D_GRID\n")
        f.write(f"{nx} {ny} {nz}\n" + vec % tuple(o_cart) + cell_txt)
        flat = data.reshape(-1, order='F')             # x fastest, as 3ddens
        for s in range(0, flat.size, 5):
            f.write(''.join("%16.8E" % v for v in flat[s:s + 5]) + "\n")
        f.write("END_DATAGRID_3D\nEND_BLOCK_DATAGRID3D\n")


def main():
    ap = argparse.ArgumentParser(description='Evaluate a WIEN2k clm-type file on a 3D grid with lapw5 and write XSF')
    ap.add_argument('--struct', required=True)
    ap.add_argument('--clm', required=True, help='case.vcoul, case.clmsum, ... (lapw5 unit 9)')
    ap.add_argument('--sub', help='second file subtracted inside lapw5 (unit 11), e.g. case.vcoul_IAM')
    ap.add_argument('--norm', choices=['val', 'tot'], default='val',
                    help="'val' for potentials/valence densities, 'tot' for case.clmsum (default val)")
    ap.add_argument('--grid', type=int, nargs=3, required=True, metavar=('NX', 'NY', 'NZ'),
                    help='points per axis including both end planes (as the first line of case.in3d)')
    ap.add_argument('--origin', type=Fraction, nargs=3, default=[Fraction(0)] * 3, metavar=('X', 'Y', 'Z'),
                    help='grid origin, fractional coordinates of the conventional cell (default 0 0 0; '
                         'fractions like 1/2 are accepted)')
    ap.add_argument('--edges', type=Fraction, nargs=9, default=[Fraction(x) for x in (1, 0, 0, 0, 1, 0, 0, 0, 1)],
                    metavar='E', help='the three grid edge vectors, fractional coordinates of the conventional '
                                      'cell (default: the conventional cell, 1 0 0  0 1 0  0 0 1)')
    ap.add_argument('--nshell', type=int, nargs=3, default=[2, 2, 2])
    ap.add_argument('--nproc', type=int, default=None)
    ap.add_argument('--workdir', default=None, help='keep lapw5 files here (default: temporary)')
    ap.add_argument('-o', '--output', required=True)
    args = ap.parse_args()

    struct = read_struct(args.struct)
    print(f"struct {args.struct}: lattice {struct['lattice']}, a,b,c = "
          f"{', '.join(f'{x*BOHR:.4f}' for x in struct['abc_bohr'])} Å")
    runner = Lapw5Runner(args.struct, args.clm, args.sub, args.norm, tuple(args.nshell),
                         args.workdir, args.nproc, keep=args.workdir is not None)
    origin = [float(x) for x in args.origin]
    edges = [[float(x) for x in args.edges[3 * i:3 * i + 3]] for i in range(3)]
    if abs(np.linalg.det(np.array(edges))) < 1e-9:
        sys.exit("Error: --edges are linearly dependent")
    for e in edges:
        if not is_lattice_translation(e, struct['lattice']):
            print(f"WARNING: edge {e} is not a lattice translation; the grid is not periodic along it")
    try:
        data = runner.grid3d(origin, edges, args.grid)
    finally:
        runner.close()
    what = f"{Path(args.clm).name}" + (f" - {Path(args.sub).name}" if args.sub else '')
    write_xsf(args.output, data, struct, f"{what}, norm {args.norm}, units ATU", origin, edges)
    print(f"wrote {args.output}: {what}, grid {tuple(args.grid)}, range [{data.min():.6g}, {data.max():.6g}]")


if __name__ == '__main__':
    main()
