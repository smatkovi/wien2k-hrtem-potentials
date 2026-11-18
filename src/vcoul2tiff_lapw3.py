#!/usr/bin/env python3
"""
vcoul2tiff_lapw3.py — projected potential for ixchel from the Fourier coefficients of case.vcoul

A second route beside vcoul2tiff_ulm.py.  No 3D grid and no cutoff: WIEN2k's
lapw3 Fourier-transforms the potential, and the projected potential is built
from the coefficients that belong to the beam direction.

Method (derivation: notes "projektion.pdf", sections 2 and 5):
  1. lapw3, run on case.vcoul instead of a density, prints for every reflection
     G = (h k l) the integral F(G) = ∫_cell V_C(r) exp(-i G·r) d³r  [Ry·bohr³].
  2. The beam runs along a lattice vector t = [u v w].  The projection of the
     potential over one period |t| has only the coefficients with G ⟂ t
     (h u + k v + l w = 0, the zeroth Laue zone):
         V_t(r) = |t| / Ω · Σ_{G ⟂ t} F(G) exp(i G·r)
  3. These G form a 2D lattice; the image repeats with the projection of the
     crystal lattice along t.  ixchel needs a rectangular cell of that projected
     lattice; the script searches the smallest one and stops if there is none.
  4. The coefficients are put on the ixchel grid (NCELL cells on dim pixels) and
     transformed back.  The result is the layer file, in 1/Å, and a crystal cfg
     with ONE layer of thickness |t|.

What "any direction" means here:
  * the beam must be a lattice direction [u v w] (a zone axis);
  * the projected lattice must have a rectangular cell (every zone axis of a
    cubic lattice has one; in other lattices many do, the script tells);
  * one period along the beam is ONE slice.  A long period is a thick slice.
Limits of this script (it stops otherwise): lattice types H and P; a structure
with an inversion centre at the origin (the real lapw3; lapw3c is untested).

Zero of the potential: the mean value is taken as lapw0 defines it.  The cutoff
route shifts to the vacuum level instead.  The two differ by a constant, which
is only a global phase of the electron wave.

Usage:
    python3 vcoul2tiff_lapw3.py --struct Graphene.struct --vcoul Graphene.vcoul_DFT \\
        --in2 Graphene.in2 --beam 0 0 1 --ncell 14 8 --dim 1008 1008 -o sim_lapw3
    (then, in sim_lapw3:  ixchel main -c crystal.cfg -s sim.cfg)

Arguments and where they come from:
    --struct     case.struct of the WIEN2k case
    --vcoul      the Coulomb potential file: case.vcoul_DFT (after the SCF) or
                 case.vcoul_IAM (after x dstart + x lapw0), see the README
    --in2        case.in2 of the case; lapw3 reads it.  The script gives lapw3 a
                 copy with a larger GMAX, because lapw3 takes the range of the
                 indices h k l from that number.
    --beam       U V W, the beam direction as a lattice vector of the
                 conventional cell
    --ncell      NX NY, how often the rectangular image cell is repeated
    --dim        NX NY, the ixchel grid; must equal dim in the simulation cfg
    -o           output directory: atompota.tif (ready for ixchel main) and
                 crystal.cfg (one layer, the atoms of the image cell)

Output on the terminal: the image cell, the lattice vectors that span it (they
can be given to lapw5grid.py --edges together with the beam), the resolution.

Layout of this file: the CORE comes first (image cell, symmetry, Fourier sum),
then the lapw3 run, the two file formats, and main(), which calls them in order.

Dependencies: numpy, tifffile; lapw3 on PATH ($WIENROOT)
"""

import argparse
import itertools
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ixchel_tiff import write_ixchel_tiff                               # noqa: E402
from lapw5grid import BOHR, conventional_vectors_ang, read_struct       # noqa: E402
from vcoul2tiff_ulm import H2_2PI_ME, RY_TO_V                           # noqa: E402

TWO_PI = 2 * math.pi


# ═══════════════════════════════════════════════════════════════
# CORE: from the coefficients of lapw3 to the layer on the ixchel grid
#
#   image_cell        the rectangular cell of the lattice projected along the beam
#   all_reflections   F(G) for every reflection, from the one per symmetry class that lapw3 lists
#   layer_on_grid     the Fourier sum on the ixchel grid
# ═══════════════════════════════════════════════════════════════

def simplest_first(n):
    """Sort key for lattice vectors (integer triples) that are otherwise equally good:
    small indices before large ones, then positive before negative ones."""
    return (sum(abs(i) for i in n), tuple(-i for i in n))


def image_cell(cell, beam):
    """
    The rectangular cell of the crystal lattice projected along the beam.

    cell : conventional lattice vectors [Å], rows;  beam : (u, v, w) integers
    Returns (c1, c2, T1, T2, multiplicity):
        c1, c2  the two edges of the image cell [Å], perpendicular to the beam and to
                each other, (c1, c2, beam) right-handed
        T1, T2  lattice vectors (integer triples) whose projections are c1 and c2
        multiplicity  area of the image cell / area of the primitive projected cell
    """
    t = np.array(beam, float) @ cell
    t_hat = t / np.linalg.norm(t)

    # 1. Project the lattice vectors of a box of cells onto the plane perpendicular to the beam.
    #    Lattice vectors that differ by a multiple of the beam vector have the same projection;
    #    of those, the one closest to the plane is kept.
    projected = {}                                    # rounded projection -> (rank, lattice vector, projection)
    for n in itertools.product(range(-6, 7), repeat=3):
        T = np.array(n, float) @ cell
        along = T @ t_hat                             # the part along the beam
        perp = T - along * t_hat                      # the projection
        key = tuple(np.round(perp, 6) + 0.0)
        rank = (round(abs(along), 6),) + simplest_first(n)
        if key not in projected or rank < projected[key][0]:
            projected[key] = (rank, n, perp)
    del projected[(0.0, 0.0, 0.0)]                    # the multiples of the beam vector itself

    # 2. Sort the projections, the shortest first.
    points = sorted(projected.values(), key=lambda p: (round(np.linalg.norm(p[2]), 6), p[0]))

    # 3. The primitive cell of the projected lattice is spanned by the shortest projection and
    #    the shortest one that is not parallel to it.  Its area is needed for the multiplicity.
    p1 = points[0][2]
    p2 = next(p[2] for p in points if np.linalg.norm(np.cross(p1, p[2])) > 1e-6)
    primitive_area = np.linalg.norm(np.cross(p1, p2))

    # 4. The image cell: of all pairs of perpendicular projections the pair with the smallest
    #    area, then with the shortest first edge, then with the simplest lattice vectors.
    best = None
    for (_, n1, v1), (_, n2, v2) in itertools.permutations(points[:120], 2):
        l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
        perpendicular = abs(v1 @ v2) <= 1e-6 * l1 * l2
        right_handed = np.cross(v1, v2) @ t_hat > 0
        if perpendicular and right_handed:
            rank = (round(l1 * l2, 6), round(l1, 6)) + simplest_first(n1) + simplest_first(n2)
            if best is None or rank < best[0]:
                best = (rank, v1, v2, n1, n2)
    if best is None:
        angle = math.degrees(math.acos(np.clip(p1 @ p2 / np.linalg.norm(p1) / np.linalg.norm(p2), -1, 1)))
        sys.exit(f"Error: the lattice projected along {list(beam)} has no rectangular cell "
                 f"(its primitive cell: {np.linalg.norm(p1):.4f} x {np.linalg.norm(p2):.4f} Å, {angle:.2f}°). "
                 f"ixchel needs a rectangular image cell.")
    _, c1, c2, T1, T2 = best
    return c1, c2, T1, T2, round(np.linalg.norm(c1) * np.linalg.norm(c2) / primitive_area)


def all_reflections(listed, operations):
    """
    Every reflection that the symmetry operations make out of the list of lapw3, with its F.

    lapw3 lists one reflection per symmetry class.  A symmetry operation x' = M x + tau turns the
    reflection L into N = L M (row vector times matrix), and F(N) = F(L) * exp(2 pi i L·tau).
    With an inversion centre at the origin all F are real, so this factor is +1 or -1.  If two
    operations give the same N with different signs, that reflection is forbidden by symmetry: F = 0.
    listed : {(h, k, l): F};  operations : list of (M, tau).   Returns {(h, k, l): F}.
    """
    result = {}
    for M, tau in operations:
        for L, F in listed.items():
            N = tuple(L[0] * M[0][j] + L[1] * M[1][j] + L[2] * M[2][j] for j in range(3))
            angle = TWO_PI * (L[0] * tau[0] + L[1] * tau[1] + L[2] * tau[2])
            if abs(math.sin(angle)) > 1e-6:
                sys.exit(f"Error: complex symmetry factor for reflection {L}; is the inversion centre at the origin?")
            value = F * round(math.cos(angle))                            # the factor is +1 or -1
            if N not in result:
                result[N] = value
            elif abs(result[N] - value) > 1e-6 * max(1.0, abs(value)):    # two operations disagree
                if abs(value) > 1e-4:
                    sys.exit(f"Error: symmetry operations give different values for reflection {N}")
                result[N] = 0.0
    return result


def layer_on_grid(coefficients, ncell, dim):
    """
    The Fourier sum on the ixchel grid.

    coefficients : {(H, K): value}, H and K count the periods along the two edges of ONE image cell
    Returns the layer [ny, nx]: ncell image cells on dim pixels, pixel (0, 0) at the cell origin.
    """
    nx, ny = dim
    spectrum = np.zeros((ny, nx), dtype=complex)
    for (H, K), value in coefficients.items():
        spectrum[(K * ncell[1]) % ny, (H * ncell[0]) % nx] = value       # H periods per cell = H * ncell per supercell
    layer = np.fft.ifft2(spectrum) * (nx * ny)
    return layer.real


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: the lapw3 run, file formats, the cfg, command line
# ═══════════════════════════════════════════════════════════════

# ── the lapw3 run ──────────────────────────────────────────────

def reflection_plane(beam, operations):
    """
    'HK', 'KL' or 'LH' if lapw3 can be asked for the reflections of one plane only, else None.

    A beam along the cell axis c needs only the reflections with l = 0 (lapw3 calls that plane
    HK), a beam along a those with h = 0 (KL), a beam along b those with k = 0 (LH).  That saves
    most of the run time.  But lapw3 lists ONE reflection per symmetry class and leaves the class
    out if that one is not in the plane.  The plane is therefore only asked for if every symmetry
    operation turns reflections of the plane into reflections of the plane.
    """
    if beam[0] == 0 and beam[1] == 0:
        plane, zero = 'HK', 2                 # index 2 (l) is zero in the plane
    elif beam[1] == 0 and beam[2] == 0:
        plane, zero = 'KL', 0                 # h is zero
    elif beam[0] == 0 and beam[2] == 0:
        plane, zero = 'LH', 1                 # k is zero
    else:
        return None
    # N = L M.  Its index `zero` is L[0] M[0][zero] + L[1] M[1][zero] + L[2] M[2][zero]; for every L
    # of the plane it stays zero only if the two entries M[i][zero] with i != zero are zero.
    for M, tau in operations:
        if any(M[i][zero] != 0 for i in range(3) if i != zero):
            return None
    return plane


def run_lapw3(struct_path, vcoul_path, in2_path, n_atoms, s_max, plane):
    """Run lapw3 on the potential file in a temporary directory.  Returns {(h, k, l): F [Ry bohr^3]}
    for one reflection of every symmetry class up to sin(theta)/lambda = s_max [1/Å].
    plane : 'HK', 'KL' or 'LH' for the reflections of that plane only (see reflection_plane),
            None for all reflections."""
    if shutil.which('lapw3') is None:
        sys.exit("Error: lapw3 not found on PATH (source the WIEN2k environment)")
    wd = Path(tempfile.mkdtemp(prefix='lapw3proj_'))
    try:
        (wd / 'case.struct').symlink_to(Path(struct_path).resolve())
        # a copy, not a link: lapw3 ends with ENDFILE on this file, which would touch the original
        shutil.copy(vcoul_path, wd / 'case.vcoul')

        # lapw3 takes the range of h, k, l from max(20, GMAX of case.in2) [1/bohr], not from s_max.
        # GMAX stands in the line after three header lines and one line per inequivalent atom.
        lines = Path(in2_path).read_text().splitlines()
        gmax = max(20.0, 1.1 * 4 * math.pi * s_max * BOHR + 1.0)
        lines[3 + n_atoms] = f" {gmax:.1f}          GMAX (enlarged for lapw3 by vcoul2tiff_lapw3.py)"
        (wd / 'case.in2').write_text("\n".join(lines) + "\n")

        # the file list: unit 8 is normally the density, here the potential
        (wd / 'lapw3.def').write_text("66,'case.output3','unknown','formatted',0\n"
                                      "4 ,'case.in2','old','formatted',0\n"
                                      "8 ,'case.vcoul','old','formatted',0\n"
                                      "15,'case.tmp3','unknown','unformatted',0\n"
                                      "20,'case.struct','old','formatted',0\n")

        # the answers to the questions of lapw3: the resolution, then the normalisation (VAL for a
        # potential file).  A negative resolution makes lapw3 ask for the plane in between.
        if plane is None:
            answers = f"{s_max:.4f}\nVAL\n"
        else:
            answers = f"-{s_max:.4f}\n{plane}\nVAL\n"

        # ONE thread.  lapw3 of WIEN2k 24.1 multiplies the atom positions by 2 pi at the first call of
        # its routine USTPHX, and that first call happens inside a parallel loop (fourir.frc).  With
        # several threads the positions can be multiplied more than once, and then every coefficient
        # is wrong (measured on graphene with the full list: 2 of 5 runs at s_max = 6).
        res = subprocess.run(['lapw3', 'lapw3.def'], cwd=wd, input=answers, capture_output=True, text=True,
                             env=dict(os.environ, OMP_NUM_THREADS='1'))
        out = wd / 'case.output3'
        text = out.read_text(errors='replace') if out.exists() else ''
        if res.returncode != 0 or 'X - RAY STRUCTURFACTORS' not in text:
            sys.exit(f"Error: lapw3 failed:\n{res.stdout[-600:]}{res.stderr[-600:]}")

        # the list at the end of case.output3:   h k l   sin(theta)/lambda   F
        #        0    0   -1   0.0333333      -98.6663026687
        listed = {}
        for line in text[text.rindex('X - RAY STRUCTURFACTORS'):].splitlines():
            p = line.split()
            if len(p) == 5 and re.fullmatch(r'-?\d+', p[0]):
                listed[(int(p[0]), int(p[1]), int(p[2]))] = float(p[4])
        return listed
    finally:
        shutil.rmtree(wd, ignore_errors=True)


# ── file format: the symmetry operations of case.struct ────────

def read_operations(struct_path):
    """The symmetry operations x' = M x + tau of case.struct as a list of (M, tau), M as three rows
    of integers, and whether the inversion at the origin (M = -1, tau = 0) is among them."""
    lines = Path(struct_path).read_text().splitlines()
    i0 = next(i for i, line in enumerate(lines) if 'NUMBER OF SYMMETRY OPERATIONS' in line)
    # one operation is three lines "M_r1 M_r2 M_r3 tau_r" (format 3I2,F10.8) and a line with its number:
    #      0-1 0 0.00000000
    #      1-1 0 0.00000000
    #      0 0 1 0.00000000
    #            2
    operations, has_inversion = [], False
    for m in range(int(lines[i0].split()[0])):
        rows = lines[i0 + 1 + 4 * m:i0 + 4 + 4 * m]
        M = [[int(r[0:2]), int(r[2:4]), int(r[4:6])] for r in rows]
        tau = [float(r[6:16]) for r in rows]
        operations.append((M, tau))
        if M == [[-1, 0, 0], [0, -1, 0], [0, 0, -1]] and all(abs(x - round(x)) < 1e-6 for x in tau):
            has_inversion = True
    return operations, has_inversion


# ── file format: the ixchel crystal cfg ────────────────────────

def atoms_of_image_cell(struct, cell, c1, c2, t):
    """
    All atoms above one image cell: (name, Z, x, y, depth), x and y in fractions of the image
    cell, depth in fractions of the period along the beam.

    c1, c2 and the beam vector t are perpendicular to each other, so an atom at r has
    x = r·c1 / |c1|²,  y = r·c2 / |c2|²,  depth = r·t / |t|².  Of every atom and its copies in the
    other cells exactly one has all three between 0 and 1.
    """
    atoms = []
    for at in struct['atoms']:
        for pos in at['positions']:
            for n in itertools.product(range(-8, 9), repeat=3):
                r = (np.array(pos) + np.array(n)) @ cell
                x = r @ c1 / (c1 @ c1)
                y = r @ c2 / (c2 @ c2)
                depth = r @ t / (t @ t)
                if all(-1e-6 < f < 1 - 1e-6 for f in (x, y, depth)):
                    # 10 decimals as written to the cfg; "% 1.0" turns a value just below zero into one just below 1
                    atoms.append((at['name'], int(round(at['Z'])),
                                  round(x, 10) % 1.0, round(y, 10) % 1.0, round(depth, 10) % 1.0))
    return sorted(atoms)


def write_cfg(path, atoms, ax, by, cz, ncell, dim):
    """One layer 'a' of thickness cz with the atoms of the image cell, in the layout of crystruct."""
    by_name = {}
    for name, Z, x, y, depth in atoms:
        by_name.setdefault((name, Z), []).append((x, y, depth))
    blocks = []
    for (name, Z), positions in by_name.items():
        rows = ",\n".join(f"\t\t\t\t\t[ {x:.10f}, {y:.10f}, {d:.10f} ]" for x, y, d in positions)
        blocks.append(f'\t\t\t{{\n\t\t\t\tname = "{name}";\n\t\t\t\tZ = {Z};\n'
                      f'\t\t\t\tpositions = (\n{rows}\n\t\t\t\t);\n\t\t\t\tuid = -1;\n\t\t\t}}')
    Path(path).write_text(
        'stack_seq = "1(a)";\n'
        f'dim = [ {dim[0]}, {dim[1]}];\n'
        f'NCELL = [ {ncell[0]}, {ncell[1]}, 1];\n'
        f'# written by vcoul2tiff_lapw3.py: one layer = one period along the beam ({cz:.4f} Ang)\n'
        'layers = {\n\ta = {\n'
        f'\t\tsize_uc = {{ x = {ax:.10f}; y = {by:.10f}; z = {cz:.10f}; }};\n'
        '\t\tfilename = "atompota.tif";\n'
        '\t\tatoms = (\n' + ",\n".join(blocks) + '\n\t\t);\n\t};\n};\n')


# ── command line ───────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Projected potential for ixchel from the Fourier coefficients of case.vcoul (lapw3)')
    ap.add_argument('--struct', required=True, help='case.struct')
    ap.add_argument('--vcoul', required=True, help='case.vcoul_DFT or case.vcoul_IAM')
    ap.add_argument('--in2', required=True, help='case.in2')
    ap.add_argument('--beam', type=int, nargs=3, required=True, metavar=('U', 'V', 'W'),
                    help='beam direction, a lattice vector of the conventional cell')
    ap.add_argument('--ncell', type=int, nargs=2, required=True, metavar=('NX', 'NY'), help='image cells in x and y')
    ap.add_argument('--dim', type=int, nargs=2, required=True, metavar=('NX', 'NY'), help='ixchel grid')
    ap.add_argument('-o', '--output', required=True, help='output directory')
    args = ap.parse_args()

    # 1. the crystal
    struct = read_struct(args.struct)
    if struct['lattice'][:1].upper() not in ('H', 'P'):
        sys.exit(f"Error: lattice type '{struct['lattice']}' is not supported (only H and P)")
    operations, has_inversion = read_operations(args.struct)
    if not has_inversion:
        sys.exit("Error: the structure has no inversion centre at the origin (lapw3c would be needed; not supported)")
    cell = conventional_vectors_ang(struct)                                   # rows a1, a2, a3 [Å]
    volume_bohr = abs(np.linalg.det(cell)) / BOHR ** 3

    # 2. the beam and the image cell
    beam = tuple(i // math.gcd(*args.beam) for i in args.beam)
    t = np.array(beam, float) @ cell                                          # one period along the beam [Å]
    period = np.linalg.norm(t)
    c1, c2, T1, T2, multiplicity = image_cell(cell, beam)
    ax, by = np.linalg.norm(c1), np.linalg.norm(c2)
    nx, ny = args.dim
    k_max = min(nx / (2 * args.ncell[0] * ax), ny / (2 * args.ncell[1] * by))  # Nyquist limit of the grid [1/Å]
    print(f"beam {list(beam)}: period {period:.4f} Å = thickness of the one layer")
    print(f"image cell: {ax:.4f} x {by:.4f} Å ({multiplicity} primitive cell(s) of the projected lattice), "
          f"spanned by the projections of the lattice vectors {list(T1)} and {list(T2)}")
    print(f"grid: {args.ncell[0]} x {args.ncell[1]} cells on {nx} x {ny} px, pixel {args.ncell[0] * ax / nx:.4f} x "
          f"{args.ncell[1] * by / ny:.4f} Å; coefficients up to 1/d = {k_max:.3f} 1/Å (ixchel uses 2/3 of that)")

    # 3. lapw3: F for one reflection per symmetry class, then for all reflections
    listed = run_lapw3(args.struct, args.vcoul, args.in2, len(struct['atoms']), 1.01 * k_max / 2,
                       reflection_plane(beam, operations))
    F_of = all_reflections(listed, operations)

    # 4. the coefficients of the layer.  The image has H periods along c1 and K along c2 per image
    #    cell; that is the reflection G = 2 pi (H c1 / ax² + K c2 / by²), if its h k l are integers.
    scale = -period * RY_TO_V / volume_bohr / H2_2PI_ME       # F [Ry bohr^3] -> coefficient of V_z [1/Å]; V_s = -V_C
    coefficients = {}
    for H in range(-int(k_max * ax) - 1, int(k_max * ax) + 2):
        for K in range(-int(k_max * by) - 1, int(k_max * by) + 2):
            G = TWO_PI * (H * c1 / ax ** 2 + K * c2 / by ** 2)                # [1/Å]
            hkl = cell @ G / TWO_PI
            if np.linalg.norm(G) / TWO_PI <= k_max and np.allclose(hkl, np.round(hkl), atol=1e-6):
                hkl = tuple(int(round(x)) for x in hkl)
                if hkl not in F_of:
                    sys.exit(f"Error: reflection {hkl} has no symmetry image in the lapw3 list ({len(listed)} entries)")
                coefficients[(H, K)] = scale * F_of[hkl]
    print(f"lapw3: {len(listed)} reflections listed, {len(coefficients)} coefficients used")

    # 5. the layer and the cfg
    layer = layer_on_grid(coefficients, args.ncell, args.dim)
    out_dir = Path(args.output)
    vmin, vmax = write_ixchel_tiff(out_dir / 'atompota.tif', layer)
    atoms = atoms_of_image_cell(struct, cell, c1, c2, t)
    write_cfg(out_dir / 'crystal.cfg', atoms, ax, by, period, args.ncell, args.dim)
    print(f"wrote {out_dir / 'atompota.tif'}: V_z from {vmin:.4g} to {vmax:.4g} 1/Å, mean {layer.mean():.5f} 1/Å")
    print(f"wrote {out_dir / 'crystal.cfg'}: one layer, {len(atoms)} atoms in the image cell")


if __name__ == '__main__':
    main()
