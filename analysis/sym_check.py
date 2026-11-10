#!/usr/bin/env python3
"""
sym_check.py — three-fold symmetry test of a graphene SCF (CXY cell, Cmmm)

In the orthorhombic CXY description of graphene the vertical bond centre
(0, 0) and the inclined bond centre (1/4, 1/4) are not related by any of the
8 symmetry operations, but they are equivalent in real graphene (6-fold axis).
Their difference therefore measures how much the SCF broke the hexagonal
symmetry (old run with the Lebedev-point reduction of the patched lapw0:
6.9 % in the density).  It also measures a cell that is not exactly hexagonal:
b/a must be sqrt(3), otherwise the vertical bond is longer or shorter than
the inclined ones and the two centres differ already in the IAM.

For the hexagonal (H, P6/mmm) cell with atoms at (1/3, 2/3) and (2/3, 1/3)
the two centres are (1/2, 0) and (1/2, 1/2); they are symmetry equivalent, so
the result is the numerical noise of lapw5:
    python3 sym_check.py <casedir> <clmsum> <vcoul> --p1 1/2 0 --p2 1/2 1/2

Evaluated with lapw5c (via lapw5grid.Lapw5Runner):
  * total density rho at both bond centres in the graphene plane (z = 0)
  * projected scattering potential V_z = sum_z V_s dz through both centres,
    V_s = (V_vac - V_C) * 13.606 V, z at 30 ppÅ (450 planes, as the Kurasch grid)

Usage:
    python3 sym_check.py <casedir> <clmsum> <vcoul> [--p1 X Y] [--p2 X Y] [--nz N]
"""
import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from lapw5grid import BOHR, Lapw5Runner, read_struct   # noqa: E402

RY_TO_V = 13.605693122994

ap = argparse.ArgumentParser()
ap.add_argument('casedir')
ap.add_argument('clmsum')
ap.add_argument('vcoul')
ap.add_argument('--nz', type=int, default=451, help='z points incl. both ends (451 = 30 ppÅ for c = 15 Å)')
ap.add_argument('--p1', type=Fraction, nargs=2, default=[Fraction(0), Fraction(0)], metavar=('X', 'Y'),
                help='vertical bond centre, fractional coordinates of the conventional cell (default 0 0)')
ap.add_argument('--p2', type=Fraction, nargs=2, default=[Fraction(1, 4), Fraction(1, 4)], metavar=('X', 'Y'),
                help='inclined bond centre (default 1/4 1/4)')
args = ap.parse_args()
(x1, y1), (x2, y2) = [float(v) for v in args.p1], [float(v) for v in args.p2]

d = Path(args.casedir)
struct_path = d / 'Graphene.struct'
st = read_struct(struct_path)
c_A = st['abc_bohr'][2] * BOHR

# density at the two bond centres, z = 0 (points: p1, p2; 2nd row unused)
run = Lapw5Runner(struct_path, d / args.clmsum, norm='tot', nproc=1)
try:
    rho = run.plane((x1, y1, 0), (x2, y2, 0), (x1, y1, 0.25), 2, 2)[:, 0]
finally:
    run.close()

# potential columns through both centres: x-axis of the plane = z of the cell
run = Lapw5Runner(struct_path, d / args.vcoul, norm='val', nproc=1)
try:
    col = run.plane((x1, y1, -0.5), (x1, y1, 0.5), (x2, y2, -0.5), args.nz, 2)   # [nz, 2]
finally:
    run.close()
col = col[:-1]                                   # drop the duplicated z end point
dz = c_A / (args.nz - 1)
vac = col.max()                                  # vacuum level (largest V_C) [Ry]
Vz = ((vac - col) * RY_TO_V).sum(axis=0) * dz    # [V*Å]

print(f"case {d}  ({args.clmsum}, {args.vcoul})")
print(f"  rho(z=0)  vertical {rho[0]:.6f}  inclined {rho[1]:.6f} e/bohr^3  "
      f"-> rel. diff {abs(rho[0] - rho[1]) / rho.mean() * 100:.3f} %")
print(f"  V_z       vertical {Vz[0]:.4f}  inclined {Vz[1]:.4f} V*Å  "
      f"(diff {Vz[0] - Vz[1]:+.4f} V*Å = {(Vz[0] - Vz[1]) / 47.88:+.2e} 1/Å, "
      f"rel. {abs(Vz[0] - Vz[1]) / Vz.mean() * 100:.3f} %)  vacuum level {vac:.5f} Ry")
