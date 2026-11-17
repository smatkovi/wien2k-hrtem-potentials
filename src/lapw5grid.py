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

Values are written by lapw5 with 9 significant digits (5ES16.8).

Units: ATU, i.e. Ry for potentials.

Command line (XSF in the 3ddens general-grid convention: n points per axis
spanning the conventional cell including both end planes, x fastest, Å;
one period therefore has n-1 points, choose n-1 so that the nuclei lie on
grid points, e.g. 75 127 451 = 74 x 126 x 450 periodic for graphene):

    python3 lapw5grid.py --struct case.struct --clm case.vcoul --grid 75 127 451 -o case_vcoul.xsf

Other cells (--edges, --origin): by default the grid spans the conventional
cell.  --edges gives the three grid edge vectors and --origin the grid origin
(default 0 0 0; fractions like 1/2 are accepted), both in fractional
coordinates of the conventional cell, so that e.g. a hexagonal (H) case can be
sampled on its rectangular (orthohexagonal) cell a1, a1 + 2 a2, the cell that
crystruct --ixchel "1 0 0  1 2 0  0 0 1" writes:

    python3 lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul --grid 75 127 451 \\
        --edges 1 0 0  1 2 0  0 0 1 -o Graphene_vcoul_ongrid.xsf

The edges must be lattice translations, otherwise the last planes are not
periodic images of the first (a warning is printed).

Arguments and where they come from:
    --struct     case.struct of the WIEN2k case.  lapw5 reads lattice, atoms and
                 symmetry from it; this script reads the lattice type and the cell.
    --clm        the file to evaluate: case.vcoul, the Coulomb potential, written
                 by `x lapw0` when line 2 of case.in0 says R2V (not NR2V).
                 In the DFT-vs-IAM workflow it is a copy made by hand (README,
                 steps 2 and 3): case.vcoul_DFT after the SCF, case.vcoul_IAM after
                 `x dstart` + `x lapw0`.
    --grid       NX NY NZ, the number of points along the three grid edges, both
                 end points included.  Chosen by the user: (N-1) / edge length is
                 the sampling rate (the paper uses 30 points per Å), and N-1
                 should have small prime factors and put the nuclei on grid points.
    --edges      nine numbers, the three edge vectors of the grid box (default: the
                 conventional cell).  Choose the cell the ixchel cfg describes,
                 i.e. the same triplet as given to crystruct --ixchel.
    --origin     where the box starts (default 0 0 0)
    --nproc      number of lapw5 runs at the same time (default: all cores but two)
    -o           the XSF file to write.  For a potential it is the --xsf input of
                 vcoul2tiff_ulm.py.

Layout of this file: the CORE comes first, in two parts.  Part 1 is the lapw5
driver; it is the same for graphene and for every other crystal, because
lapw5 reads case.struct itself.  Part 2 holds the lattice types.  After the
core: the file formats (case.struct in, XSF out), the checks with their error
messages, and the command line.

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


# ═══════════════════════════════════════════════════════════════
# CORE, part 1 of 2: the lapw5 driver
#
# Turns a WIEN2k file into values on a 3D grid, plane by plane.  Nothing here
# depends on the crystal: lapw5 reads case.struct itself, so graphene and
# every other lattice type take exactly this path.
# ═══════════════════════════════════════════════════════════════

# The file list ("def file") that lapw5c gets as its only argument, one line per Fortran unit:
#     5  the input file, written in plane()         9  the file to evaluate
#     6  the log of lapw5                          21  the result: the values on the plane
#     8  the struct file
#    10, 11, 12, 20  further files that lapw5 opens; this script does not read them
# {tag} becomes the name of the plane, so that several lapw5 runs can work side by side.
DEF = """\
5 ,'{tag}.in5','old','formatted',0
6 ,'{tag}.output5','unknown','formatted',0
8 ,'case.struct','old','formatted',0
9 ,'case.clm','old','formatted',0
10,'{tag}.tmp','unknown','unformatted',0
11,'{tag}.none','unknown','formatted',0
12,'{tag}.sigma','unknown','formatted',0
20,'{tag}.rho_onedim','unknown','formatted',0
21,'{tag}.rho','unknown','formatted',0
"""


class Lapw5Runner:
    """Evaluate a WIEN2k file on planes with lapw5.

        runner = Lapw5Runner('case.struct', 'case.vcoul')      # prepares a temporary working directory
        data = runner.grid3d(origin, edges, (nx, ny, nz))      # one lapw5c run per plane
        runner.close()                                         # removes the working directory
    """

    def __init__(self, struct_path, clm, nproc=None):
        """struct_path  the case.struct file
        clm          the file to evaluate (case.vcoul)
        nproc        number of lapw5 runs at the same time (default: all cores but two)"""
        check_clm_header(clm)                           # stops if the file was written with NONORM
        self.nproc = nproc or max(1, (os.cpu_count() or 2) - 2)

        # the working directory: lapw5 runs here, nothing is written into the case directory.
        # Links with fixed names, so that every def file can say 'case.struct' and 'case.clm'.
        self.wd = Path(tempfile.mkdtemp(prefix='lapw5grid_'))
        (self.wd / 'case.struct').symlink_to(Path(struct_path).resolve())
        (self.wd / 'case.clm').symlink_to(Path(clm).resolve())

        self._counter = 0               # numbers the planes of direct plane() calls

    def grid3d(self, origin, edges, counts):
        """Values at origin + i/(n0-1) e0 + j/(n1-1) e1 + k/(n2-1) e2 for all
        i < n0, j < n1, k < n2 (end points included).  The grid is computed plane by
        plane, several planes at the same time.  Returns an array [n0, n1, n2]."""
        origin = np.asarray(origin, float)
        edges = [np.asarray(e, float) for e in edges]
        counts = [int(n) for n in counts]

        # lapw5 computes one plane per run.  The planes are stacked along the axis with the fewest
        # points (fewest runs); the other two axes become the x and y edge of every plane.
        stack_axis = int(np.argmin(counts))
        x_axis, y_axis = [ax for ax in range(3) if ax != stack_axis]
        n_planes = counts[stack_axis]

        def one_plane(m):
            """Plane number m: its origin lies m/(n_planes-1) of the way along the stacking edge."""
            o = origin + edges[stack_axis] * (m / (n_planes - 1))
            return self.plane(o, o + edges[x_axis], o + edges[y_axis], counts[x_axis], counts[y_axis],
                              tag=f"g{m:05d}")

        # nproc planes at a time.  Threads are enough: each one only waits for its lapw5c process.
        with ThreadPoolExecutor(max_workers=self.nproc) as pool:
            planes = list(pool.map(one_plane, range(n_planes)))         # results arrive in the order of m
        return np.stack(planes, axis=stack_axis)

    def plane(self, origin, xend, yend, npx, npy, tag=None):
        """Values on the npx x npy grid origin + i/(npx-1)(xend-origin) + j/(npy-1)(yend-origin)
        (fractional coordinates of the conventional cell).  Returns [npx, npy]."""
        if tag is None:                             # called directly, not from grid3d: number the plane here
            self._counter += 1
            tag = f"p{self._counter:05d}"
        wd = self.wd

        # 1. NSHELL: lapw5 looks for muffin-tin spheres only among the cell origins
        #    -NSHELL..+NSHELL, so every corner of the plane must lie well inside (at least 2 cells)
        o, x, y = (np.asarray(v, float) for v in (origin, xend, yend))
        corners = [o, x, y, x + y - o]                              # the fourth one lies opposite the origin
        reach = np.abs(corners).max(axis=0)                         # farthest corner along each cell axis
        nshell = [max(2, math.ceil(far - 1e-9) + 1) for far in reach]

        # 2. the input file of lapw5, one list entry per line of the file
        in5 = ["%d %d %d %d" % _int_vector(origin),        # origin of the plane, as IX IY IZ IDV (see _int_vector)
               "%d %d %d %d" % _int_vector(xend),          # end point of its x edge
               "%d %d %d %d" % _int_vector(yend),          # end point of its y edge
               "%d %d %d" % tuple(nshell),                 # NSHELL along the three cell axes
               "%d %d" % (npx, npy),                       # number of points along the x and the y edge
               "RHO     ",                                 # evaluate the file as it is
               "ATU VAL NODEBUG",                          # atomic units; VAL: the normalisation of potentials
               "NONORTHO"]                                 # the two edges need not be perpendicular
        (wd / f"{tag}.in5").write_text("\n".join(in5) + "\n")

        # 3. the file list of this plane, from the template DEF
        (wd / f"{tag}.def").write_text(DEF.format(tag=tag))

        # 4. run lapw5c in the working directory (one core per run, since several run side by side)
        res = subprocess.run(['lapw5c', f"{tag}.def"], cwd=wd, capture_output=True, text=True,
                             env=dict(os.environ, OMP_NUM_THREADS='1'))
        rho = wd / f"{tag}.rho"
        check_lapw5_run(res, rho, wd / f"{tag}.output5", tag)       # raises with the end of the lapw5 log

        # 5. the result file: a header line (npx, npy, the two edge lengths), then the values
        with open(rho) as f:
            f.readline()                                # the header line is not needed
            vals = np.array(f.read().split(), dtype=np.float64)

        # 6. remove the files of this plane
        for ext in ('in5', 'def', 'output5', 'tmp', 'sigma', 'rho_onedim', 'rho', 'none'):
            (wd / f"{tag}.{ext}").unlink(missing_ok=True)
        return vals.reshape((npx, npy))            # lapw5 (prtrh.f) writes x in the outer loop, y in the inner

    def close(self):
        """Remove the working directory."""
        shutil.rmtree(self.wd, ignore_errors=True)


def _int_vector(frac3, max_den=10**6):
    """(IX, IY, IZ, IDV) with IX/IDV = frac3[0] etc.  lapw5 reads a point as
    three integers over a common denominator: (0.5, 0, 0) -> (1, 0, 0, 2), (1.5, 2, 0) -> (3, 4, 0, 2)."""
    # every coordinate as a fraction; a float becomes the nearest fraction with a denominator <= max_den
    fracs = [x if isinstance(x, Fraction) else Fraction(x).limit_denominator(max_den) for x in frac3]
    den = math.lcm(*(f.denominator for f in fracs))    # the common denominator
    numerators = [int(f * den) for f in fracs]
    return (*numerators, den)


# ═══════════════════════════════════════════════════════════════
# CORE, part 2 of 2: lattice types
#
# The geometry of the cell, needed to give the grid in Å and to recognise
# lattice vectors.  H is the hexagonal lattice of graphene; the other entries
# cover the other crystals (orthogonal cells, see conventional_vectors_ang).
# ═══════════════════════════════════════════════════════════════

# Centring translations of the WIEN2k lattice types, in fractions of the conventional cell.  Every atom
# listed in case.struct exists once at each of them.
CENTRING = {
    'H':   [(0, 0, 0)],                                                # hexagonal (graphene)
    'P':   [(0, 0, 0)],                                                # primitive
    'B':   [(0, 0, 0), (0.5, 0.5, 0.5)],                               # body-centred
    'F':   [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],   # face-centred
    'CXY': [(0, 0, 0), (0.5, 0.5, 0)],                                 # base-centred, xy face
    'CXZ': [(0, 0, 0), (0.5, 0, 0.5)],                                 # base-centred, xz face
    'CYZ': [(0, 0, 0), (0, 0.5, 0.5)],                                 # base-centred, yz face
}


def conventional_vectors_ang(struct):
    """Cartesian conventional lattice vectors [Å] (rows) in the WIEN2k/3ddens
    orientation.  Orthogonal cells and the hexagonal H lattice only."""
    a, b, c = (x * BOHR for x in struct['abc_bohr'])
    if struct['lattice'].upper().startswith('H'):                        # graphene
        return np.array([[a * math.sqrt(3) / 2, -a / 2, 0],             # a1
                         [0, a, 0],                                     # a2, at 120° to a1
                         [0, 0, c]])                                    # c
    if all(abs(x - 90.0) < 1e-6 for x in struct['angles']):              # the other crystals: orthogonal cells
        return np.diag([a, b, c])
    raise NotImplementedError("non-orthogonal cells other than H are not supported")


def centring_translations(lattice):
    """The centring translations of a lattice type as it is named in case.struct ('H', 'CXY', ...)."""
    for key, shifts in CENTRING.items():
        if lattice.upper().startswith(key):
            return list(shifts)
    raise NotImplementedError(f"lattice type '{lattice}' not supported")


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: file formats, checks, command line
# ═══════════════════════════════════════════════════════════════

# ── file formats: case.struct in, XSF out (3ddens-compatible general grid) ──

def read_struct(path):
    """Lattice type, lattice constants [bohr], angles [deg] and all atoms of
    the conventional cell as listed in case.struct (centring copies are NOT
    included; see centring_translations)."""
    lines = Path(path).read_text().splitlines()

    # line 2 starts with the lattice type; line 4 holds a, b, c, alpha, beta, gamma, 10 characters each
    lattice = lines[1][:4].strip()
    abc = [float(lines[3][i:i + 10]) for i in (0, 10, 20)]
    ang = []
    for i in (30, 40, 50):
        s = lines[3][i:i + 10].strip()
        ang.append(float(s) if s and float(s) != 0.0 else 90.0)      # blank or 0 means 90°

    # then one block per inequivalent atom, here the one of graphene (MULT = 2 equivalent positions):
    #     ATOM  -1: X=0.33333333 Y=0.66666667 Z=0.00000000                   line i: the first position
    #               MULT= 2          ISPLIT= 4                               line i + 1
    #           -1: X=0.66666667 Y=0.33333333 Z=0.00000000                   MULT - 1 further positions
    #     C 1        NPT=  781  R0=0.00010000 RMT=    1.3300   Z:  6.0       element line: name and Z
    atoms, i = [], 4
    while i < len(lines):
        if lines[i].startswith('ATOM'):
            mult = int(lines[i + 1].split('MULT=')[1].split()[0])
            position_lines = [i] + list(range(i + 2, i + 1 + mult))     # line i, then the lines after MULT
            el = lines[i + 1 + mult]                                    # the element line
            atoms.append({'name': el[:10].strip(), 'Z': float(el.split('Z:')[1].split()[0]),
                          'positions': [_xyz(lines[j]) for j in position_lines]})
            i += 2 + mult                            # continue after the element line
        else:
            i += 1
    return {'lattice': lattice, 'abc_bohr': abc, 'angles': ang, 'atoms': atoms}


def _xyz(line):
    """The three numbers of a position line '... X=0.33333333 Y=0.66666667 Z=0.00000000'."""
    s = line.split('X=')[1]
    return (float(s[:10]), float(s.split('Y=')[1][:10]), float(s.split('Z=')[1][:10]))


def write_xsf(path, data, struct, comment, origin=(0, 0, 0), edges=((1, 0, 0), (0, 1, 0), (0, 0, 1))):
    """Write data [nx, ny, nz] as the data grid of an XSF file.  origin, edges: grid origin and edge vectors
    in fractional coordinates of the conventional cell (default: the conventional cell itself)."""
    conv = conventional_vectors_ang(struct)
    cell = np.array(edges, float) @ conv               # Cartesian grid vectors [Å] (rows)
    o_cart = np.asarray(origin, float) @ conv
    nx, ny, nz = data.shape
    vec = "%16.8E%16.8E%16.8E\n"                       # one vector per line
    with open(path, 'w') as f:
        f.write(f"# XSF-File prepared by lapw5grid.py (values from lapw5)\n# {comment}\n")
        # the grid: points per axis, origin, the three vectors that span it, then the values
        f.write("BEGIN_BLOCK_DATAGRID3D\n lapw5grid\nDATAGRID_3D_GRID\n")
        f.write(f"{nx} {ny} {nz}\n" + vec % tuple(o_cart) + ''.join(vec % tuple(v) for v in cell))
        flat = data.reshape(-1, order='F')             # x fastest, as 3ddens
        for s in range(0, flat.size, 5):               # five values per line
            f.write(''.join("%16.8E" % v for v in flat[s:s + 5]) + "\n")
        f.write("END_DATAGRID_3D\nEND_BLOCK_DATAGRID3D\n")


# ── checks and error messages ──────────────────────────────────

def check_clm_header(path):
    """Refuse vcoul files written without the r^2 factor (NONORM patch of lapw0);
    lapw5 assumes the standard normalisation."""
    with open(path) as f:
        f.readline()
        second = f.readline()
    if 'NORM OF' in second and '*r**2' not in second.replace(' ', ''):
        sys.exit(f"Error: {path} was written without the r^2 normalisation (NONORM); "
                 f"lapw5 cannot evaluate it. Re-run lapw0 without NONORM.")


def check_lapw5_run(res, rho, out5, tag):
    """After the lapw5 run of one plane: raise with the end of its output if it failed."""
    if (res.returncode != 0 or not re.search(r'STOP\s+LAPW5C?\s+END', res.stdout + res.stderr)
            or not rho.exists()):
        tail = out5.read_text(errors='replace')[-1500:] if out5.exists() else ''
        raise RuntimeError(f"lapw5c failed for plane {tag}:\n{res.stdout[-800:]}{res.stderr[-800:]}\n"
                           f"--- end of {tag}.output5 ---\n{tail}")


def is_lattice_translation(vec, lattice, tol=1e-6):
    """True if vec (fractional, conventional cell) is a translation of the lattice."""
    v = np.asarray(vec, float)
    return any(np.all(np.abs((v - np.array(t)) - np.round(v - np.array(t))) < tol)
               for t in centring_translations(lattice))


# ── command line ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Evaluate a WIEN2k clm-type file on a 3D grid with lapw5 and write XSF')
    ap.add_argument('--struct', required=True)
    ap.add_argument('--clm', required=True, help='the file to evaluate, case.vcoul (lapw5 unit 9)')
    ap.add_argument('--grid', type=int, nargs=3, required=True, metavar=('NX', 'NY', 'NZ'),
                    help='points per axis including both end planes (as the first line of case.in3d)')
    ap.add_argument('--origin', type=Fraction, nargs=3, default=[Fraction(0)] * 3, metavar=('X', 'Y', 'Z'),
                    help='grid origin, fractional coordinates of the conventional cell (default 0 0 0; '
                         'fractions like 1/2 are accepted)')
    ap.add_argument('--edges', type=Fraction, nargs=9, default=[Fraction(x) for x in (1, 0, 0, 0, 1, 0, 0, 0, 1)],
                    metavar='E', help='the three grid edge vectors, fractional coordinates of the conventional '
                                      'cell (default: the conventional cell, 1 0 0  0 1 0  0 0 1)')
    ap.add_argument('--nproc', type=int, default=None)
    ap.add_argument('-o', '--output', required=True)
    args = ap.parse_args()

    struct = read_struct(args.struct)
    print(f"struct {args.struct}: lattice {struct['lattice']}, a,b,c = "
          f"{', '.join(f'{x*BOHR:.4f}' for x in struct['abc_bohr'])} Å")
    origin = [float(x) for x in args.origin]
    edges = [[float(x) for x in args.edges[3 * i:3 * i + 3]] for i in range(3)]
    for e in edges:
        if not is_lattice_translation(e, struct['lattice']):
            print(f"WARNING: edge {e} is not a lattice translation; the grid is not periodic along it")
    runner = Lapw5Runner(args.struct, args.clm, args.nproc)
    try:
        data = runner.grid3d(origin, edges, args.grid)
    finally:
        runner.close()
    what = Path(args.clm).name
    write_xsf(args.output, data, struct, f"{what}, units ATU (Ry)", origin, edges)
    print(f"wrote {args.output}: {what}, grid {tuple(args.grid)}, range [{data.min():.6g}, {data.max():.6g}]")


if __name__ == '__main__':
    main()
