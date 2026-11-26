#!/usr/bin/env python3
"""
line_integral_lapw3.py — the potential integrated along a straight line, from the Fourier coefficients of case.vcoul

An electron enters the crystal at one point and leaves it at another.  This
script integrates the Coulomb potential along the straight line between the
two points:

    ∫ V_s ds   from --entry to --exit        [V·Å],  V_s = -V_C

The line may have any direction and any length.  It need not be a lattice
direction, and nothing has to repeat along it.

Method:
  1. lapw3, run on case.vcoul, prints F(G) = ∫_cell V_C(r) exp(-i G·r) d³r
     [Ry·bohr³] for the reflections up to sin(theta)/lambda = s_max (as in
     plot_kurasch_fig2_lapw3.py).  The potential is the Fourier sum
         V_C(r) = 1/Ω · Σ_G F(G) exp(i G·r).
  2. Every term of the sum is integrated along the line in closed form.  With
     M the middle of the line, D the vector from entry to exit and S = |D|:
         ∫ exp(i G·r) ds = S · exp(i G·M) · sin(G·D/2) / (G·D/2)
     No points on the line and no step width are needed.  If D is a lattice
     vector, only the reflections with G·D = 0 remain: the projection of
     vcoul2tiff_lapw3.py.
  3. At a nucleus V_C = -2Z/r.  The integral along a line that passes a nucleus
     at the distance rho grows like 4Z ln(1/rho) and is infinite for a line
     through the nucleus; a finite sum cannot show that.  As in
     plot_kurasch_fig2_lapw3.py a reference is therefore taken out of the
     coefficients first (point nuclei, each with a Gaussian cloud of charge
     -Z), and the line integral of the reference, -2Z erfc(alpha d)/d for a
     nucleus at the distance d, is added again in real space:
         ∫ -2Z erfc(alpha d)/d ds = -2Z [asinh((s - s0)/rho)] + 2Z ∫ erf(alpha d)/d ds
     with d² = rho² + (s - s0)², s0 the point of the line closest to the
     nucleus.  The first part is the bare nucleus in closed form, the second
     is smooth and is integrated numerically.

  4. With --radius R the line becomes a cylinder: the two points are the centres
     of its end faces, and the result is the mean of the line integral over
     the cross-section (the integral over the cylinder divided by pi R²).  In
     the sum that is one more factor for every term,
         2 J1(G⊥ R) / (G⊥ R),   G⊥ = the part of G perpendicular to the line,
     and for the reference a mean over the circular face (reference_on_line).
     The mean is finite even if the axis passes through a nucleus, but it then
     depends on R: it grows like 4Z ln(1/R).

The script prints two numbers for every potential file: the plain sum of step
2 (always finite, but cut at s_max), and the sum with the nuclei taken out
(step 3), which is the result.  Far from the nuclei the two agree.

Zero of the potential: as lapw0 defines it (the mean value F(0)/Ω is printed).
A different zero changes the integral by a constant times the length.

Limits (the script stops otherwise): lattice types H and P; a structure with
an inversion centre at the origin (as vcoul2tiff_lapw3.py).

Usage (graphene, along c through the middle of a hexagon, and an oblique line):
    python3 line_integral_lapw3.py --struct Graphene.struct --in2 Graphene.in2 \\
        --vcoul Graphene.vcoul_DFT --entry 0 0 -0.5 --exit 0 0 0.5
    python3 line_integral_lapw3.py --struct Graphene.struct --in2 Graphene.in2 \\
        --vcoul Graphene.vcoul_DFT --entry 0.1 0.2 -0.3 --exit 0.45 0.3 0.3

Arguments and where they come from:
    --struct     case.struct of the WIEN2k case
    --in2        case.in2 of the case; lapw3 reads it (it gets a copy with a
                 larger GMAX, see vcoul2tiff_lapw3.py)
    --vcoul      one or more Coulomb potential files (case.vcoul_DFT,
                 case.vcoul_IAM, see the README)
    --entry      X Y Z, where the electron enters: fractional coordinates of the
                 conventional cell, like the atom positions in case.struct
                 (fractions such as 1/3 are accepted)
    --exit       X Y Z, where it leaves
    --radius     R [Å]: mean over a cylinder of this radius around the line
                 (default 0, the line itself)
    --smax       largest sin(theta)/lambda of the reflections [1/Å] (default 6;
                 lapw3 then takes about a minute per file for graphene)

Dependencies: numpy, scipy; lapw3 on PATH ($WIENROOT)
"""

import argparse
import itertools
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import erf, j1

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lapw5grid import BOHR, conventional_vectors_ang, read_struct                    # noqa: E402
from plot_kurasch_fig2_lapw3 import reflections_in_sphere                            # noqa: E402
from vcoul2tiff_lapw3 import all_reflections, read_operations, run_lapw3             # noqa: E402
from vcoul2tiff_ulm import RY_TO_V                                                   # noqa: E402

TWO_PI = 2 * math.pi


def integrated_terms(hkl, G, cell, entry, exit_, radius):
    """
    Step 2: for every reflection the integral of exp(i G·r) from entry to exit, divided by the length.

    In fractional coordinates G·r = 2 pi hkl·x, so with the middle m and the difference d of the two
    points:  exp(i G·M) = exp(2 pi i hkl·m)  and  sin(G·D/2) / (G·D/2) = sin(pi hkl·d) / (pi hkl·d),
    which is numpy's sinc(hkl·d).  The imaginary parts of G and -G cancel, the cosine remains.

    Step 4, radius > 0: the mean over a disc of that radius perpendicular to the line multiplies the
    term by 2 J1(x)/x with x = G⊥ · radius.  Along the line G has the component G·D / S, and
    G⊥² = G² - (G·D / S)².
    G [1/bohr], cell [bohr], radius [bohr].  Returns one number per reflection.
    """
    middle = (entry + exit_) / 2
    d = exit_ - entry
    terms = np.cos(TWO_PI * (hkl @ middle)) * np.sinc(hkl @ d)
    if radius > 0:
        g_along = TWO_PI * (hkl @ d) / np.linalg.norm(d @ cell)
        x = np.sqrt(np.maximum(G ** 2 - g_along ** 2, 0.0)) * radius
        not_zero = x > 1e-9
        terms[not_zero] *= 2 * j1(x[not_zero]) / x[not_zero]             # the factor is 1 for x = 0
    return terms


def reference_coefficients(hkl, G, atoms, alpha):
    """
    Step 3: the coefficients of the reference, as in smooth_part of plot_kurasch_fig2_lapw3.py.

    Reference = every nucleus (charge Z) with a Gaussian cloud of charge -Z.  One of them at x_a has
        F_ref(G) = -8 pi Z / G² * (1 - exp(-G² / 4 alpha²)) * exp(-i G·x_a),   F_ref(0) = -2 pi Z / alpha².
    """
    F_ref = np.zeros(len(G))
    not_zero = G > 1e-9
    for Z, x_a in atoms:
        structure = np.cos(TWO_PI * (hkl @ x_a))        # the imaginary parts cancel in the sum over the atoms
        F_ref[not_zero] += (-8 * np.pi * Z / G[not_zero] ** 2
                            * (1 - np.exp(-G[not_zero] ** 2 / (4 * alpha ** 2))) * structure[not_zero])
        F_ref[~not_zero] += -2 * np.pi * Z / alpha ** 2
    return F_ref


def arc_inside(rho, rho0, radius):
    """
    The angle [rad] of the circle of radius rho around a nucleus that lies inside the disc.

    The disc has the given radius, the nucleus sits at the distance rho0 from its centre.  The whole
    circle (2 pi) lies inside if rho <= radius - rho0, nothing if the two do not meet.
    """
    if rho0 < 1e-9:                                        # the nucleus on the axis
        return TWO_PI if rho < radius else 0.0
    return 2 * math.acos(min(1.0, max(-1.0, (rho ** 2 + rho0 ** 2 - radius ** 2) / (2 * rho * rho0))))


def reference_on_line(struct, cell, entry, exit_, alpha, radius=0.0):
    """
    Step 3: the line integral of the reference in real space [Ry·bohr], and the nucleus closest to the line.

    A nucleus at the distance rho from a line, closest to the line point s0 (s = distance from
    the entry point), contributes  -2Z erfc(alpha d)/d  with d² = rho² + (s - s0)².  Beyond
    d = 6/alpha that is zero, so only the piece s1..s2 of the line inside that range counts:
        g(rho) = -2Z [asinh((s2 - s0)/rho) - asinh((s1 - s0)/rho)]  +  2Z ∫ erf(alpha d)/d ds

    Step 4, radius > 0: the mean of g over the disc around the line.  g depends only on the distance
    rho from the nucleus, so the mean is a sum over circles around the nucleus, each counted with
    the part of it that lies inside the disc (arc_inside); rho0 = distance of the nucleus from the axis:
        1 / (pi R²) · ∫ g(rho) · rho · arc_inside(rho) d rho      from |rho0 - R| or 0 to rho0 + R

    Returns (integral, (distance of the closest nucleus from the line [bohr], its name)).  For the
    line itself (radius 0) the integral is -infinity if the line passes through a nucleus.
    """
    a, b = entry @ cell, exit_ @ cell                       # the two points [bohr]
    length = np.linalg.norm(b - a)
    u = (b - a) / length                                    # unit vector along the line
    reach = 6.0 / alpha

    # the cells whose atoms can lie within `reach` of the line or cylinder, one more on every side
    margin = (reach + radius) * np.linalg.norm(np.linalg.inv(cell).T, axis=1)       # in fractions of the cell
    low = np.floor(np.minimum(entry, exit_) - margin).astype(int) - 2
    high = np.ceil(np.maximum(entry, exit_) + margin).astype(int) + 1
    shifts = np.array(list(itertools.product(*[range(lo, hi + 1) for lo, hi in zip(low, high)])))

    total, closest = 0.0, (np.inf, '')
    for at in struct['atoms']:
        Z = at['Z']
        for pos in at['positions']:
            centres = (np.array(pos) + shifts) @ cell                       # this atom in all those cells
            s0 = (centres - a) @ u                                          # closest point of the infinite line
            rho0 = np.linalg.norm(centres - a - np.outer(s0, u), axis=1)    # distance from the infinite line
            on_line = np.clip(s0, 0.0, length)                              # closest point between entry and exit
            distance = np.linalg.norm(centres - a - np.outer(on_line, u), axis=1)
            if distance.min() < closest[0]:
                closest = (distance.min(), at['name'])
            for i in np.flatnonzero((rho0 < reach + radius) & (s0 > -reach) & (s0 < length + reach)):
                s1, s2, at_s0 = max(0.0, s0[i] - reach), min(length, s0[i] + reach), s0[i]

                def g(rho):
                    """The contribution of this nucleus to the line at the distance rho from it."""
                    r = max(rho, 1e-12)             # a nucleus on the extension of the line: rho = 0, but d > 0 on the line
                    bare = -2 * Z * (math.asinh((s2 - at_s0) / r) - math.asinh((s1 - at_s0) / r))
                    cloud = 2 * Z * quad(lambda s: erf(alpha * math.hypot(r, s - at_s0)) / math.hypot(r, s - at_s0),
                                         s1, s2, epsabs=1e-12, epsrel=1e-12)[0]
                    return bare + cloud

                if radius == 0:
                    if distance[i] < 1e-6:          # the nucleus lies on the line (positions in case.struct have 8 digits)
                        return -np.inf, closest
                    total += g(rho0[i])
                else:
                    inner, outer = max(0.0, rho0[i] - radius), min(reach, rho0[i] + radius)
                    if inner < outer:
                        whole = radius - rho0[i]    # up to here the whole circle lies inside the disc
                        total += quad(lambda rho: g(rho) * rho * arc_inside(rho, rho0[i], radius), inner, outer,
                                      points=[whole] if inner < whole < outer else None,
                                      epsabs=1e-11, epsrel=1e-11, limit=200)[0] / (np.pi * radius ** 2)
    return total, closest


def main():
    ap = argparse.ArgumentParser(description="Line integral of the Coulomb potential between two points, "
                                             "from the lapw3 coefficients")
    ap.add_argument('--struct', required=True, help='case.struct')
    ap.add_argument('--in2', required=True, help='case.in2')
    ap.add_argument('--vcoul', required=True, nargs='+', help='case.vcoul_DFT [case.vcoul_IAM ...]')
    ap.add_argument('--entry', type=Fraction, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
                    help='where the electron enters, fractional coordinates of the cell')
    ap.add_argument('--exit', type=Fraction, nargs=3, required=True, metavar=('X', 'Y', 'Z'),
                    help='where it leaves')
    ap.add_argument('--radius', type=float, default=0.0, help='radius of a cylinder around the line [Å] (default 0)')
    ap.add_argument('--smax', type=float, default=6.0, help='largest sin(theta)/lambda [1/Å]')
    args = ap.parse_args()

    # the crystal
    struct = read_struct(args.struct)
    if struct['lattice'][:1].upper() not in ('H', 'P'):
        sys.exit(f"Error: lattice type '{struct['lattice']}' is not supported (only H and P)")
    operations, has_inversion = read_operations(args.struct)
    if not has_inversion:
        sys.exit("Error: the structure has no inversion centre at the origin (lapw3c would be needed; not supported)")
    cell = conventional_vectors_ang(struct) / BOHR                # rows a1, a2, a3 [bohr]
    volume = abs(np.linalg.det(cell))                             # [bohr³]
    atoms = [(at['Z'], np.array(pos)) for at in struct['atoms'] for pos in at['positions']]

    # the line
    entry = np.array([float(x) for x in args.entry])
    exit_ = np.array([float(x) for x in args.exit])
    length = np.linalg.norm((exit_ - entry) @ cell)               # [bohr]
    if length < 1e-9:
        sys.exit("Error: --entry and --exit are the same point")
    print(f"line: from {entry} to {exit_} (fractions of the cell), length {length * BOHR:.4f} Å")
    radius = args.radius / BOHR                                   # [bohr]
    if radius > 0:
        print(f"cylinder of radius {args.radius:g} Å around the line: mean over its cross-section")

    # the sum is cut at g_cut; the width of the Gaussian clouds is tied to it as in plot_kurasch_fig2_lapw3.py
    g_cut = 4 * np.pi * args.smax * BOHR                          # [1/bohr]
    alpha = g_cut / 10.0                                          # [1/bohr]
    reference, (distance, name) = reference_on_line(struct, cell, entry, exit_, alpha, radius)
    print(f"closest nucleus: {name}, {distance * BOHR:.6f} Å from the line")

    to_volt_ang = -RY_TO_V * BOHR                                 # Ry·bohr -> V·Å, and V_s = -V_C
    for vcoul in args.vcoul:
        listed = run_lapw3(args.struct, vcoul, args.in2, len(struct['atoms']), 1.01 * args.smax, plane=None)
        hkl, F, G = reflections_in_sphere(all_reflections(listed, operations), cell, g_cut)
        terms = integrated_terms(hkl, G, cell, entry, exit_, radius)
        plain = length / volume * np.sum(F * terms)
        smooth = length / volume * np.sum((F - reference_coefficients(hkl, G, atoms, alpha)) * terms)
        print(f"{Path(vcoul).name}: lapw3 listed {len(listed)} reflections, {len(hkl)} used")
        print(f"    plain sum up to s_max = {args.smax:g} 1/Å:   {plain * to_volt_ang:14.4f} V·Å")
        if np.isinf(reference):
            print("    with the nuclei taken out:        infinite, the line passes through a nucleus")
        else:
            print(f"    with the nuclei taken out:      {(smooth + reference) * to_volt_ang:14.4f} V·Å   <- the result")
        print(f"    mean potential of the file:     {-F[G < 1e-9].sum() / volume * RY_TO_V:14.4f} V   (zero as lapw0 defines it)")


if __name__ == '__main__':
    main()
