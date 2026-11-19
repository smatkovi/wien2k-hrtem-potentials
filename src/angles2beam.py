#!/usr/bin/env python3
"""
Beam direction from two angles: which lattice directions [u v w] lie closest to it?

A direction in space needs two angles. Here they are measured from a known zone axis:

  --tilt     angle between the beam and the zone axis [degrees]
  --azimuth  direction of the tilt, as a rotation about the zone axis [degrees];
             0 means: towards the lattice direction given with --x

    z = zone axis,  x = the part of --x perpendicular to z,  y = z × x
    beam = cos(tilt) z + sin(tilt) (cos(azimuth) x + sin(azimuth) y)

Only integer [u v w] repeat along the beam, so the script lists the integer directions
with indices up to --max-index, sorted by their angle to the wanted beam, with the
period along each of them (conventional cell).

Example (rutile, 35 degrees away from c towards a):

    angles2beam.py --struct TiO2.struct --zone 0 0 1 --x 1 0 0 --tilt 35 --azimuth 0
"""
import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lapw5grid import conventional_vectors_ang, read_struct       # noqa: E402


def unit(v):
    return v / np.linalg.norm(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--struct', required=True, help='WIEN2k case.struct')
    ap.add_argument('--zone', type=int, nargs=3, required=True, metavar=('U', 'V', 'W'), help='zone axis at zero tilt')
    ap.add_argument('--x', type=int, nargs=3, required=True, metavar=('U', 'V', 'W'), help='lattice direction of azimuth 0')
    ap.add_argument('--tilt', type=float, required=True, help='angle to the zone axis [degrees]')
    ap.add_argument('--azimuth', type=float, default=0.0, help='rotation about the zone axis [degrees]')
    ap.add_argument('--max-index', type=int, default=6, help='largest index |u|, |v|, |w| in the list')
    ap.add_argument('--lines', type=int, default=10, help='number of directions printed')
    args = ap.parse_args()

    cell = conventional_vectors_ang(read_struct(args.struct))        # rows a1, a2, a3 [Å]

    # 1. The wanted beam as a vector in space.
    z = unit(np.array(args.zone) @ cell)
    x = np.array(args.x) @ cell
    x = x - (x @ z) * z                                              # only the part perpendicular to z
    if np.linalg.norm(x) < 1e-9:
        sys.exit('Error: --x is parallel to --zone')
    x = unit(x)
    y = np.cross(z, x)
    tilt, azimuth = math.radians(args.tilt), math.radians(args.azimuth)
    beam = math.cos(tilt) * z + math.sin(tilt) * (math.cos(azimuth) * x + math.sin(azimuth) * y)

    # 2. The same vector in lattice coordinates; in general these are not integers.
    exact = beam @ np.linalg.inv(cell)
    exact = exact / np.abs(exact).max()
    print(f"wanted beam in lattice coordinates: [{exact[0]:.4f} {exact[1]:.4f} {exact[2]:.4f}]")

    # 3. All integer directions up to max-index, each one once (no common factor).
    n = args.max_index
    found = []
    for uvw in itertools.product(range(-n, n + 1), repeat=3):
        if math.gcd(*uvw) != 1:
            continue
        t = np.array(uvw) @ cell
        period = np.linalg.norm(t)
        angle = math.degrees(math.acos(min(1.0, (t @ beam) / period)))
        found.append((angle, period, uvw))
    found.sort()

    print("   u   v   w   angle to wanted beam [deg]   period [Å]")
    for angle, period, (u, v, w) in found[:args.lines]:
        print(f"{u:4d}{v:4d}{w:4d}   {angle:12.4f}               {period:9.3f}")


if __name__ == '__main__':
    main()
