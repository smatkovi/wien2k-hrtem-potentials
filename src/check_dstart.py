#!/usr/bin/env python3
"""
check_dstart.py — check that `x dstart` produced a usable IAM (superposed-atom) density

WIEN2k 24.1 dstart never initialises the logical `tau_mode` (declared in
SRC_dstart/dstart.F, only ever set to .true. in init.F).  Depending on the
stack contents a run therefore takes the tau/potential code path by accident:
no density smoothing, no renormalisation, and a density that is not neutral
(graphene 2026-03-15: 12.027 instead of 12 electrons, sloped vacuum potential).
A stale case.rsp (lstart run for a different RMT, R0 or NPT) gives a charged
density as well, even when the renormalisation runs.

Checks on case.outputd against case.struct:
  1. the standard path was taken: 'DENSITY SMOOTHENED' (old_dstart) and ':NEC01'
     (charge normalisation) are printed;
  2. the nuclear charge on ':NEC01' equals sum(Z * MULT) of case.struct;
  3. every 'Charge decomposition' ZZ equals the Z of its atom in case.struct
     (otherwise case.rsp does not belong to this struct: run x lstart).
Exit status 0 if everything passes, 1 otherwise.  Rerun `x dstart` until the
check passes; only a stale case.rsp needs `x lstart` first.

Usage (in the case directory):
    python3 check_dstart.py Graphene
    python3 check_dstart.py --outputd Graphene.outputd --struct Graphene.struct

Dependencies: none beyond the standard library
"""

import argparse
import re
import sys
from pathlib import Path

TOL = 1e-3


def struct_atoms(path):
    """(name, Z, MULT) of the inequivalent atoms in case.struct."""
    lines = Path(path).read_text().splitlines()
    atoms, i = [], 4
    while i < len(lines):
        if lines[i].startswith('ATOM'):
            mult = int(lines[i + 1].split('MULT=')[1].split()[0])
            el = lines[i + 1 + mult]
            atoms.append((el[:10].strip(), float(el.split('Z:')[1].split()[0]), mult))
            i += 2 + mult
        else:
            i += 1
    return atoms


def main():
    ap = argparse.ArgumentParser(description='Check a dstart run (IAM density) for the known WIEN2k 24.1 pitfalls')
    ap.add_argument('case', nargs='?', help='case name (uses <case>.outputd and <case>.struct)')
    ap.add_argument('--outputd')
    ap.add_argument('--struct')
    args = ap.parse_args()
    outputd = args.outputd or (f"{args.case}.outputd" if args.case else None)
    struct = args.struct or (f"{args.case}.struct" if args.case else None)
    if not outputd or not struct:
        ap.error('give a case name or both --outputd and --struct')

    text = Path(outputd).read_text(errors='replace')
    atoms = struct_atoms(struct)
    ok = True

    def report(passed, msg):
        nonlocal ok
        ok &= passed
        print(("  ok    " if passed else "  FAIL  ") + msg)

    print(f"{outputd} vs {struct}")
    report('DENSITY SMOOTHENED' in text,
           "standard dstart path (old_dstart: 'DENSITY SMOOTHENED' printed)")
    nec = re.search(r':NEC01:\s*NUCLEAR AND ELECTRONIC CHARGE\s+(\S+)\s+(\S+)', text)
    z_total = sum(z * m for _, z, m in atoms)
    if nec:
        nuc, ele = float(nec.group(1)), float(nec.group(2))
        report(abs(nuc - z_total) < TOL,
               f":NEC01 nuclear charge {nuc:.5f} = sum(Z*MULT) {z_total:.5f} "
               f"(electronic before renormalisation {ele:.5f})")
    else:
        report(False, "':NEC01' missing: the charge was not renormalised")

    zz = [float(m.group(1)) for m in
          re.finditer(r'Charge decomposition:[^\n]*\n\s*([-+0-9.Ee]+)', text)]
    if len(zz) != len(atoms):
        report(False, f"{len(zz)} 'Charge decomposition' blocks for {len(atoms)} atoms")
    for (name, z, _), value in zip(atoms, zz):
        report(abs(value - z) < TOL,
               f"atom {name}: ZZ {value:.5f} = Z {z:.5f}"
               + ("" if abs(value - z) < TOL else " (case.rsp is stale: run x lstart)"))

    print("PASS" if ok else "FAIL — rerun x dstart (and x lstart if ZZ != Z) before x lapw0")
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
