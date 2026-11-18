#!/usr/bin/env python3
"""
plot_kurasch_fig2_lapw3.py — the paper's Fig. 2 for a crystal, from the Fourier coefficients of case.vcoul

Kurasch et al. 2011, Fig. 2, is a line scan of the WIEN2k starting potential
from the nucleus of a single carbon atom outwards, compared with the Kirkland
and Doyle-Turner atomic potentials and with the bare nucleus.  This script
draws the same figure for a crystal (graphene), and it takes the WIEN2k curve
from the coefficients of lapw3, the numbers that vcoul2tiff_lapw3.py also
uses, instead of a lapw5 line scan.

Method:
  1. lapw3, run on case.vcoul, prints F(G) = ∫_cell V_C(r) exp(-i G·r) d³r
     [Ry·bohr³] for one reflection of every symmetry class up to
     sin(theta)/lambda = s_max.  The symmetry operations of case.struct give
     the other reflections.
  2. The potential is the Fourier sum  V_C(r) = 1/Ω · Σ_G F(G) exp(i G·r).
     At a nucleus V_C = -2Z/r, and no finite sum gives that.  Therefore a
     reference is taken out first: point nuclei, each with a Gaussian cloud of
     charge -Z around it.  Its coefficients are known in closed form and are
     subtracted from F(G); what remains is smooth and its sum converges.  The
     reference itself, -2Z erfc(alpha d)/d for a nucleus at distance d, is
     added again in real space.
  3. The line runs from the first atom of case.struct along the lattice vector
     [u v w] to half of it, i.e. to the middle between the atom and its next
     copy, with --ppa points per Å.
  4. As in the paper: V_s = -V_C in volts, smallest value set to zero.

The comparison curves are Kirkland and Doyle-Turner atoms placed on ALL atom
positions of the crystal and summed, because in a crystal the neighbours
contribute along the line.  The blue curve is the bare nucleus of the scanned
atom alone, as in the paper.

Accuracy, measured on graphene against a lapw5c line scan of the same file
along the same line (s_max = 6): 3 V of 21 000 V at 0.004 Å, 0.1 V between 0.1
and 0.4 Å, 0.05 V up to 1 Å, and a ripple of 0.006 V beyond (the sum is cut at
s_max).  The figure therefore ends at 0.01 V; the paper's axis goes down to
1e-6 V.

Usage:
    python3 plot_kurasch_fig2_lapw3.py --struct Graphene.struct --in2 Graphene.in2 \\
        --vcoul Graphene.vcoul_IAM Graphene.vcoul_DFT --direction 0 0 1 \\
        -o kurasch_fig2_graphene_lapw3.png

Arguments and where they come from:
    --struct     case.struct of the WIEN2k case
    --in2        case.in2 of the case; lapw3 reads it (it gets a copy with a
                 larger GMAX, see vcoul2tiff_lapw3.py)
    --vcoul      one or more Coulomb potential files.  The first one is drawn as
                 the paper draws its curve (red, dashed); for the paper's figure
                 that is the starting potential, case.vcoul_IAM (x dstart +
                 x lapw0).  case.vcoul_DFT (after the SCF) can follow.
    --direction  U V W, the lattice vector of the conventional cell along which
                 the line runs; graphene: 0 0 1 = from the nucleus
                 perpendicular to the sheet into the middle of the vacuum
    --smax       largest sin(theta)/lambda of the reflections [1/Å] (default 6;
                 lapw3 then takes about a minute per file for graphene, on one
                 thread, see run_lapw3 in vcoul2tiff_lapw3.py)
    --ppa        points per Å of the line (default 250, the paper's value)
    -o           the figure to write (default kurasch_fig2_lapw3.png)

Dependencies: numpy, scipy, matplotlib; lapw3 on PATH ($WIENROOT)
"""

import argparse
import itertools
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.special import erfc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lapw5grid import BOHR, conventional_vectors_ang, read_struct                    # noqa: E402
from plot_kurasch_fig2 import (E2, _DT, _KIRK, doyle_turner_potential,               # noqa: E402
                               kirkland_potential, normalize_like_paper)
from vcoul2tiff_lapw3 import all_reflections, read_operations, run_lapw3             # noqa: E402

TWO_PI = 2 * math.pi


# ═══════════════════════════════════════════════════════════════
# CORE: from the list of lapw3 to the potential along the line
#
#   reflections_in_sphere   the reflections inside s_max as arrays (all_reflections of
#                           vcoul2tiff_lapw3.py makes them from the one per symmetry class)
#   smooth_part             F(G) minus the reference, summed along the line
#   reference_on_line       the reference (nuclei with Gaussian clouds) in real space
# ═══════════════════════════════════════════════════════════════

def reflections_in_sphere(F_of, cell, g_cut):
    """
    The reflections with |G| <= g_cut as arrays, and a check that none is missing.

    F_of : {(h, k, l): F} from all_reflections (vcoul2tiff_lapw3.py);  cell : lattice vectors [bohr], rows
    Returns hkl [n, 3] (integers), F [n], |G| [n] in 1/bohr.
    """
    recip = np.linalg.inv(cell).T                                 # rows b_i / 2 pi [1/bohr]
    hkl = np.array(list(F_of.keys()))
    F = np.array(list(F_of.values()))
    G = TWO_PI * np.linalg.norm(hkl @ recip, axis=1)
    inside = G <= g_cut
    hkl, F, G = hkl[inside], F[inside], G[inside]

    # nothing may be missing: count the points of the reciprocal lattice inside the sphere.
    # The index along a_i is G·a_i / 2 pi, so it cannot exceed g_cut |a_i| / 2 pi.
    reach = [int(g_cut / TWO_PI * np.linalg.norm(a)) + 1 for a in cell]
    box = np.array(list(itertools.product(*[range(-n, n + 1) for n in reach])))
    expected = int((TWO_PI * np.linalg.norm(box @ recip, axis=1) <= g_cut).sum())
    if len(hkl) != expected:
        sys.exit(f"Error: {expected} reflections lie inside s_max, but the lapw3 list gives {len(hkl)}")
    return hkl, F, G


def smooth_part(hkl, F, G, atoms, alpha, volume, start, direction, t):
    """
    The Fourier sum of F(G) minus the reference, at the points start + t * direction.

    Reference = every nucleus (charge Z) with a Gaussian cloud of charge -Z.  One of them at x_a has
        F_ref(G) = -8 pi Z / G² * (1 - exp(-G² / 4 alpha²)) * exp(-i G·x_a),   F_ref(0) = -2 pi Z / alpha².
    start, direction : fractional coordinates;  t : fractions of the lattice vector `direction`
    Returns the sum [Ry] at every t.
    """
    F_ref = np.zeros(len(F))
    not_zero = G > 1e-9
    for Z, x_a in atoms:
        # exp(-i G·x_a): the imaginary parts of G and -G cancel in the sum over all atoms and reflections
        structure = np.cos(TWO_PI * (hkl @ x_a))
        F_ref[not_zero] += (-8 * np.pi * Z / G[not_zero] ** 2
                            * (1 - np.exp(-G[not_zero] ** 2 / (4 * alpha ** 2))) * structure[not_zero])
        F_ref[~not_zero] += -2 * np.pi * Z / alpha ** 2
    rest = F - F_ref

    # along the line, exp(i G·r) = exp(2 pi i hkl·start) * exp(2 pi i m t) with the integer m = hkl·direction.
    # All reflections with the same m are added first; what is left is a Fourier series in t.
    m = hkl @ np.array(direction)
    series = np.zeros(m.max() - m.min() + 1, dtype=complex)
    np.add.at(series, m - m.min(), rest * np.exp(TWO_PI * 1j * (hkl @ start)))
    orders = np.arange(m.min(), m.max() + 1)
    total = np.exp(TWO_PI * 1j * np.outer(t, orders)) @ series / volume
    return total.real


def reference_on_line(distances, alpha):
    """The reference in real space [Ry]: -2Z erfc(alpha d)/d for every nucleus near the line (d in bohr)."""
    total = np.zeros(len(distances[0][2]))
    for Z, _, d in distances:
        with np.errstate(divide='ignore', invalid='ignore'):
            total += np.where(d > 0, -2 * Z * erfc(alpha * d) / d, -np.inf)
    return total


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: the atoms around the line, the comparison curves, figure, command line
# ═══════════════════════════════════════════════════════════════

def atoms_near_line(struct, cell, points, reach):
    """(Z, element symbol, distance [same unit as cell] of every line point to the atom) for every
    atom of the crystal that is closer than about `reach` to the cell of the line."""
    recip = np.linalg.inv(cell).T
    cells = [int(np.ceil(reach * np.linalg.norm(row))) + 1 for row in recip]       # cell copies per axis
    result = []
    for at in struct['atoms']:
        symbol = re.match(r'[A-Za-z]+', at['name']).group(0)
        for pos in at['positions']:
            for n in itertools.product(*[range(-c, c + 1) for c in cells]):
                centre = (np.array(pos) + np.array(n)) @ cell
                result.append((at['Z'], symbol, np.linalg.norm(points - centre, axis=1)))
    return result


def atom_sum(potential, distances):
    """Sum of isolated-atom potentials [V] over all atoms near the line (distances in Å)."""
    total = np.zeros(len(distances[0][2]))
    for _, symbol, d in distances:
        with np.errstate(divide='ignore', invalid='ignore'):
            total += np.where(d < 12.0, potential(d, symbol), 0.0)
    return total


def main():
    ap = argparse.ArgumentParser(description="Fig. 2 of Kurasch et al. 2011 for a crystal, from the lapw3 coefficients")
    ap.add_argument('--struct', required=True, help='case.struct')
    ap.add_argument('--in2', required=True, help='case.in2')
    ap.add_argument('--vcoul', required=True, nargs='+', help='case.vcoul_IAM [case.vcoul_DFT ...]')
    ap.add_argument('--direction', type=int, nargs=3, required=True, metavar=('U', 'V', 'W'),
                    help='lattice vector along which the line runs')
    ap.add_argument('--smax', type=float, default=6.0, help='largest sin(theta)/lambda [1/Å]')
    ap.add_argument('--ppa', type=float, default=250.0, help='points per Å (paper: 250)')
    ap.add_argument('-o', default='kurasch_fig2_lapw3.png')
    args = ap.parse_args()

    # the crystal
    struct = read_struct(args.struct)
    if struct['lattice'][:1].upper() not in ('H', 'P'):
        sys.exit(f"Error: lattice type '{struct['lattice']}' is not supported (only H and P)")
    operations, has_inversion = read_operations(args.struct)
    if not has_inversion:
        sys.exit("Error: the structure has no inversion centre at the origin (lapw3c would be needed; not supported)")
    cell = conventional_vectors_ang(struct)                       # rows a1, a2, a3 [Å]
    cell_bohr = cell / BOHR
    volume = abs(np.linalg.det(cell_bohr))                        # [bohr³]
    atoms = [(at['Z'], np.array(pos)) for at in struct['atoms'] for pos in at['positions']]
    symbols = {re.match(r'[A-Za-z]+', at['name']).group(0) for at in struct['atoms']}
    if not symbols <= set(_KIRK) or not symbols <= set(_DT):
        sys.exit(f"Error: no Kirkland / Doyle-Turner parameters for {sorted(symbols)} in plot_kurasch_fig2.py")

    # the line: from the first atom along the lattice vector, half of its length
    start = atoms[0][1]
    direction = np.array(args.direction, float)
    length = np.linalg.norm(direction @ cell) / 2                 # [Å]
    n = int(round(length * args.ppa)) + 1
    t = np.linspace(0.0, 0.5, n)                                  # fractions of the lattice vector
    r = t * 2 * length                                            # distance from the nucleus [Å]
    points = (start + np.outer(t, direction)) @ cell              # [Å]
    print(f"line: from atom {start} along {args.direction}, {n} points over 0..{length:.3f} Å")

    # the sum is cut at g_cut; the width of the Gaussian clouds is tied to it so that
    # exp(-g_cut² / 4 alpha²) = e^-25: the reference is then smooth on the scale of the cut
    g_cut = 4 * np.pi * args.smax * BOHR                          # [1/bohr]
    alpha = g_cut / 10.0                                          # [1/bohr]
    near_bohr = atoms_near_line(struct, cell_bohr, points / BOHR, reach=3.0)

    # the WIEN2k curves
    curves = []
    for vcoul in args.vcoul:
        listed = run_lapw3(args.struct, vcoul, args.in2, len(struct['atoms']), 1.01 * args.smax, plane=None)
        hkl, F, G = reflections_in_sphere(all_reflections(listed, operations), cell_bohr, g_cut)
        v_ry = smooth_part(hkl, F, G, atoms, alpha, volume, start, args.direction, t) + reference_on_line(near_bohr, alpha)
        v = normalize_like_paper(v_ry[1:])                        # without the point on the nucleus
        curves.append((Path(vcoul).name, v))
        print(f"{Path(vcoul).name}: lapw3 listed {len(listed)} reflections, {len(hkl)} used; "
              f"V_C from {v_ry[1]:.1f} Ry at {r[1]:.3f} Å to {v_ry[-1]:.4f} Ry at the end; "
              f"shift {-(-v_ry[1:] * 13.605693122994).min():+.4f} V")
    r = r[1:]

    # the comparison curves
    near_ang = [(Z, symbol, d[1:]) for Z, symbol, d in atoms_near_line(struct, cell, points, reach=12.0)]
    v_kirk = atom_sum(kirkland_potential, near_ang)
    v_dt = atom_sum(doyle_turner_potential, near_ang)
    v_coul = atoms[0][0] * E2 / r

    # the layout of the paper's Fig. 2: log-log main panel, linear inset 0-0.4 Å
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.loglog(r, v_kirk, 'g-', lw=1.6, label='Kirkland atoms, summed')
    for i, (name, v) in enumerate(curves):
        style = dict(color='r', ls='--') if i == 0 else dict(color='tab:orange', ls=':')
        ax.loglog(r, np.where(v > 0, v, np.nan), lw=1.6, label=f'WIEN2k {name}', **style)
    ax.loglog(r, v_dt, 'k-', lw=1.0, label='Doyle–Turner atoms, summed')
    ax.loglog(r, v_coul, 'b-', lw=0.7, label='Coulomb of nucleus')
    ax.set_xlim(2e-3, 10)
    ax.set_ylim(1e-2, 1e5)
    ax.set_xlabel('distance in Å')
    ax.set_ylabel('potential in V')
    ax.set_title(f"{Path(args.struct).stem}: line from a nucleus along {args.direction}, "
                 f"potential from the lapw3 coefficients\nlayout of Kurasch et al. 2011, Fig. 2", fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    inset = ax.inset_axes([0.09, 0.08, 0.36, 0.5])
    close = r <= 0.4
    inset.plot(r[close], v_kirk[close], 'g-', lw=1.4)
    for i, (name, v) in enumerate(curves):
        inset.plot(r[close], v[close], lw=1.4, **(dict(color='r', ls='--') if i == 0 else dict(color='tab:orange', ls=':')))
    inset.plot(r[close], v_dt[close], 'k-', lw=0.9)
    inset.plot(r[close], v_coul[close], 'b-', lw=0.7)
    inset.set_xlim(0, 0.4)
    inset.set_ylim(0, 3000)
    inset.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(args.o, dpi=150, bbox_inches='tight')
    print(f"saved {args.o}")

    # the numbers at some distances
    width = max(len(name) for name, _ in curves) + 2
    print("\n   r [Å]  " + "".join(f"{name:>{width}s}" for name, _ in curves) + "      Kirkland  Doyle-Turner   [V]")
    for distance in [0.004, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        j = int(np.argmin(np.abs(r - distance)))
        print(f"  {r[j]:6.3f}  " + "".join(f"{v[j]:{width}.4f}" for _, v in curves) + f"{v_kirk[j]:14.4f}{v_dt[j]:14.4f}")


if __name__ == '__main__':
    main()
