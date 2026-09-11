#!/usr/bin/env python3
"""
vcoul2tiff.py — WIEN2k Coulomb potentials (DFT − IAM via lapw5) → ixchel TIFFs

METHOD:
  V_z = V_z^Kirkland + ΔV_z
  ΔV_s(r) = −[V_C^DFT(r) − V_C^IAM(r)]
      case.vcoul holds the Coulomb potential energy of an electron, V_C [Ry]
      (negative at the nuclei); the scattering potential is V_s = −V_C.
  ΔV_C is evaluated by lapw5 ('RHO SUB' of case.vcoul_DFT and case.vcoul_IAM,
  see lapw5grid.py).  lapw5 applies WIEN2k's own star expansion, local rotation
  matrices, symmetry operations and lattice centring, and it subtracts the two
  files before evaluating them, so the −2Z/r nuclear terms cancel exactly and
  ΔV is finite everywhere (no cutoff).
  ΔV_z(x, y) = ∫ ΔV_s dz over the layer's depth window, trapezoid rule on a
  regular grid (--dz), converted to Kirkland units: V·Å / 47.88 V·Å².
  V_z^Kirkland is ixchel's atompot algorithm (kirkland_layer from
  xsf2tiff_dwFactor.py: B-factors, subslicing, band limit), which reproduces
  `ixchel atompot` to float precision.

GEOMETRY:
  The TIFF frame (x, y, beam) is spanned by the crystal directions v1, v2, v3
  of the conventional cell: either the crystruct triplet the cfg was made with
  (--frame) or v3 = --miller with v1, v2 from xsf2tiff_dwFactor.inplane_vectors.
  The cfg layer sizes must equal |v1| and |v2|.  The cfg atoms are matched to
  the struct atoms (including the centring copies) to find the origin of the
  cfg frame; the run stops if no offset makes them coincide.
  Depth window per layer (dv_window): centred on the layer's atomic plane,
  half way to its neighbours (classical projection) or the layer slab
  [z0, z0+cz) (subslicing), as in xsf2tiff_dwFactor.py.
  Pixel (i, j) sits at (i·ax/nx, j·by/ny), as in ixchel atompot.

CONVERGENCE (graphene, 15 Å cell): dz = 67 mÅ changes ΔV_z by < 1.3e-5 1/Å
  against dz = 2 mÅ; 25 mÅ lateral sampling leaves < 1e-5 1/Å aliasing.

OUTPUT:
  One unit cell per layer, <outdir>/<cfg filename>, float32 in 1/Å, rows = y.
  Put the layers on the ixchel `dim` grid with tiff2ixchel.py before running
  `ixchel main`.

POTENTIAL ZERO:
  lapw0 fixes the zero of each potential by its own convention, so V_C^DFT and
  V_C^IAM have different vacuum levels (graphene: 2.314 V vs 2.814 V) and ΔV
  carries a constant offset.  For image simulation that is a global phase.
  For slab geometries with vacuum, --vacuum-depth D shifts ΔV_s so that its
  mean on the plane at cfg depth D is zero (both potentials referenced to the
  vacuum, as the Kirkland term is); the lateral spread on that plane is
  printed and must be small.  A sloped vacuum means a non-neutral starting
  density: regenerate case.vcoul_IAM and check ':NEC01' in case.outputd.
  Without --vacuum-depth, mean values (⟨DFT − IAM⟩, mean inner potential) are
  not meaningful.

Usage:
    python3 vcoul2tiff.py \\
        --vcoul-dft Graphene.vcoul_DFT --vcoul-iam Graphene.vcoul_IAM \\
        --struct Graphene.struct --cfg Graphene_ixchel_001.cfg \\
        [--miller 0 0 1 | --frame 1 0 0  0 1 0  0 0 1] [--dz 0.05] [--dx 0.025] \\
        [--vacuum-depth 7.5] -o wien2k_001

Dependencies: numpy, scipy, tifffile, pylibconfig2; lapw5 on PATH ($WIENROOT)
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lapw5grid as l5                    # noqa: E402
import xsf2tiff_dwFactor as xsf           # noqa: E402

RY_TO_V   = 13.605693122994   # 1 Ry of electron energy ↔ 13.606 V of potential
H2_2PI_ME = 47.88             # h²/(2π m_e e) [V·Å²], Kirkland/ixchel unit


# ═══════════════════════════════════════════════════════════════
# Geometry
# ═══════════════════════════════════════════════════════════════

def struct_atoms_cart(struct, conv):
    """All atoms of the conventional cell (listed positions × centring) as
    (Z, Cartesian position [Å])."""
    atoms = []
    for at in struct['atoms']:
        for pos in at['positions']:
            for t in l5.centring_translations(struct['lattice']):
                f = (np.array(pos) + np.array(t)) % 1.0
                atoms.append((int(round(at['Z'])), f @ conv))
    return atoms


def frame_offset(atoms, cfg, conv, ex, ey, ez, period):
    """
    Offset t = (t_u, t_v, t_d) [Å] of the cfg frame origin in struct
    coordinates along (ex, ey, ez), such that every struct atom has a cfg atom
    of the same Z at cfg position + t (modulo ax, by and the beam period) and
    every cfg atom lies on a struct atom (modulo the conventional lattice).
    Returns (t, largest mismatch [Å]).
    """
    cat = [(at['Z'], at['proj_x'] * ly['ax'], at['proj_y'] * ly['by'],
            ly['z0'] + at['proj_z'] * ly['cz'], ly['ax'], ly['by'])
           for ly in cfg['layers'] for at in ly['atoms']]
    if not atoms or not cat:
        return np.zeros(3), np.inf
    to_frac = np.linalg.inv(conv.T)
    s_frac = [(Z, to_frac @ r) for Z, r in atoms]

    def wrap(d, L):
        return d - L * np.round(d / L)

    def mismatch(t):
        worst = 0.0
        for Z, r in atoms:                                   # struct -> cfg
            u, v, d = r @ ex - t[0], r @ ey - t[1], r @ ez - t[2]
            dist = [math.sqrt(wrap(u - cu, cax)**2 + wrap(v - cv, cby)**2 +
                              wrap(d - cd, period)**2)
                    for cZ, cu, cv, cd, cax, cby in cat if cZ == Z]
            worst = max(worst, min(dist) if dist else np.inf)
        for cZ, cu, cv, cd, _, _ in cat:                    # cfg -> struct
            f = to_frac @ ((cu + t[0]) * ex + (cv + t[1]) * ey + (cd + t[2]) * ez)
            dist = [np.linalg.norm(conv.T @ (df - np.round(df)))
                    for Z, sf in s_frac if Z == cZ for df in [f - sf]]
            worst = max(worst, min(dist) if dist else np.inf)
        return worst

    Z0, r0 = atoms[0]
    cands = [np.array([r0 @ ex - cu, r0 @ ey - cv, r0 @ ez - cd])
             for cZ, cu, cv, cd, _, _ in cat if cZ == Z0]
    if not cands:
        return np.zeros(3), np.inf
    return min(((t, mismatch(t)) for t in cands), key=lambda x: x[1])


def apply_bfactor_overrides(layers, tokens, cfg_path):
    """--bfactors Elem:B — Elem is the exact cfg atom name ("C 1") or its element
    symbol, i.e. the leading letters of the name ("C 1" -> C, "Si1" -> Si)."""
    override = {}
    for tok in tokens:
        el, val = tok.rsplit(':', 1)
        override[el.strip()] = float(val)
    if not override:
        return
    used, changed = set(), {}
    for ly in layers:
        for at in ly['atoms']:
            sm = re.match(r'[A-Za-z]+', at['name'].strip())
            sym = sm.group(0) if sm else None
            used.update(k for k in (at['name'], sym) if k in override)
            key = at['name'] if at['name'] in override else (sym if sym in override else None)
            if key is None:
                continue
            changed.setdefault((at['name'], at['Z']), (at['B'], key))
            at['B'] = override[key]
    for (name, Z), (B_old, key) in changed.items():
        print(f"    B override {key}: {name}(Z={Z}) B={B_old:.3f} → {override[key]:.3f} Å²")
    unused = [el for el in override if el not in used]
    if unused:
        names = sorted(set(at['name'] for ly in layers for at in ly['atoms']))
        sys.exit(f"Error: --bfactors {', '.join(unused)} matched no atom in {cfg_path} "
                 f"(atom names: {', '.join(repr(n) for n in names)})")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='WIEN2k vcoul (DFT − IAM, evaluated by lapw5) + Kirkland → ixchel TIFFs')
    ap.add_argument('--vcoul-dft', required=True, help='case.vcoul from the SCF density')
    ap.add_argument('--vcoul-iam', required=True, help='case.vcoul from the dstart density')
    ap.add_argument('--struct', required=True, help='case.struct')
    ap.add_argument('--cfg', required=True, help='ixchel crystal-structure cfg (crystruct)')
    ap.add_argument('--miller', type=int, nargs=3, default=[0, 0, 1], metavar=('H', 'K', 'L'),
                    help='beam direction (conventional cell); in-plane axes from '
                         'inplane_vectors unless --frame is given (default 0 0 1)')
    ap.add_argument('--frame', type=int, nargs=9, metavar='N',
                    help='crystruct triplet "x1 x2 x3  y1 y2 y3  z1 z2 z3" the cfg was made with')
    ap.add_argument('--bfactors', nargs='+', default=[], metavar='Elem:B',
                    help='B-factor overrides (ixchel convention exp(-B k²)); Elem = element '
                         'symbol (leading letters of the cfg atom name) or exact cfg name')
    ap.add_argument('--dx', type=float, default=0.025,
                    help='target lateral pixel size of the unit-cell TIFF [Å] (default 0.025)')
    ap.add_argument('--dz', type=float, default=0.05,
                    help='sampling step of ΔV along the beam [Å] (default 0.05)')
    ap.add_argument('--vacuum-depth', type=float, default=None, metavar='D',
                    help='cfg depth [Å] of a plane in the vacuum; ΔV is shifted to zero mean '
                         'there (slabs only; default: no shift)')
    ap.add_argument('--nproc', type=int, default=None, help='parallel lapw5 processes')
    ap.add_argument('--nshell', type=int, nargs=3, default=[2, 2, 2],
                    help='lapw5 neighbour shells (default 2 2 2)')
    ap.add_argument('--workdir', default=None, help='keep the lapw5 files here (default: temporary)')
    ap.add_argument('-o', '--output', default=None)
    args = ap.parse_args()

    if args.frame:
        v1, v2, v3 = args.frame[0:3], args.frame[3:6], args.frame[6:9]
    else:
        v3 = list(args.miller)
        v1, v2 = xsf.inplane_vectors(v3)
    v1, v2, v3 = ([int(x) for x in v] for v in (v1, v2, v3))
    tag = ''.join(str(abs(x)) for x in v3)
    out_dir = args.output or f"wien2k_{tag}"

    # ── struct and potentials ───────────────────────────────────────────────
    print(f"\n[1] struct: {args.struct}")
    struct = l5.read_struct(args.struct)
    try:
        conv = l5.conventional_vectors_ang(struct)
    except NotImplementedError as e:
        sys.exit(f"Error: {e}")
    print(f"    lattice {struct['lattice']}, a, b, c = "
          + ", ".join(f"{np.linalg.norm(v):.4f}" for v in conv) + " Å")

    # ── cfg ─────────────────────────────────────────────────────────────────
    print(f"\n[2] cfg: {args.cfg}")
    try:
        cfg = xsf.parse_cfg(args.cfg)
    except ValueError as e:
        sys.exit(f"Error: {e}")
    layers = cfg['layers']
    print(f"    {len(layers)} layer(s), {len(cfg['slices'])} slice(s), total thickness "
          f"{cfg['total_z']:.3f} Å, dim {cfg['dim'][0]}×{cfg['dim'][1]}, "
          f"NCELL {cfg['ncell'][0]}×{cfg['ncell'][1]}, "
          + ("subslicing" if cfg['subslicing'] else "classical projection"))
    apply_bfactor_overrides(layers, args.bfactors, args.cfg)
    seen = {}
    for ly in layers:
        for at in ly['atoms']:
            seen.setdefault((at['name'], at['Z']), at['B'])
    for (name, Z), B in sorted(seen.items(), key=lambda x: x[1]):
        print(f"    {name}(Z={Z}): B={B:.3f} Å²  σ_DW={np.sqrt(B / (2 * np.pi**2)) * 1000:.1f} mÅ")

    # ── frame ───────────────────────────────────────────────────────────────
    X = [conv.T @ np.array(v, float) for v in (v1, v2, v3)]
    L = [np.linalg.norm(x) for x in X]
    if min(L) == 0:
        sys.exit(f"Error: degenerate frame {v1} {v2} {v3}")
    ex, ey, ez = (x / n for x, n in zip(X, L))
    for (i, j) in ((0, 1), (0, 2), (1, 2)):
        if abs(X[i] @ X[j]) > 1e-6 * L[i] * L[j]:
            sys.exit(f"Error: frame {v1} {v2} {v3} is not orthogonal in this cell")
    period = L[2]
    print(f"\n[3] frame: x = {v1} ({L[0]:.4f} Å), y = {v2} ({L[1]:.4f} Å), "
          f"beam = {v3} (period {period:.4f} Å)")
    for ly in layers:
        if abs(ly['ax'] - L[0]) > 1e-3 * L[0] or abs(ly['by'] - L[1]) > 1e-3 * L[1]:
            sys.exit(f"Error: layer '{ly['label']}' size_uc {ly['ax']:.4f} × {ly['by']:.4f} Å does not "
                     f"match |x| × |y| = {L[0]:.4f} × {L[1]:.4f} Å; pass the crystruct triplet "
                     f"with --frame")
    atoms = struct_atoms_cart(struct, conv)
    t, err = frame_offset(atoms, cfg, conv, ex, ey, ez, period)
    print(f"    cfg origin at struct (x, y, depth) = ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) Å, "
          f"largest atom mismatch {err:.3g} Å")
    if not err < 0.05:
        sys.exit("Error: the cfg atoms do not coincide with the struct atoms in this frame; "
                 "pass the crystruct triplet the cfg was made with (--frame)")
    try:
        windows = {ly['label']: xsf.dv_window(ly, cfg) for ly in layers}
    except ValueError as e:
        sys.exit(f"Error: {e}")

    # ── ΔV with lapw5 ───────────────────────────────────────────────────────
    print(f"\n[4] lapw5: {args.vcoul_dft} − {args.vcoul_iam}")
    runner = l5.Lapw5Runner(args.struct, args.vcoul_dft, args.vcoul_iam, norm='val',
                            nshell=tuple(args.nshell), workdir=args.workdir,
                            nproc=args.nproc, keep=args.workdir is not None)
    to_frac = np.linalg.inv(conv.T)
    outs = []
    try:
        dv_shift = 0.0                                              # [V], subtracted from ΔV_s
        if args.vacuum_depth is not None:
            # planes at D and D ± 1 Å: a charged (non-neutral) starting density gives a
            # parabolic vacuum potential that is flat laterally but not along the depth
            nxv = max(16, round(layers[0]['ax'] / args.dx))
            nyv = max(16, round(layers[0]['by'] / args.dx))
            vac = {}
            for dd in (-1.0, 0.0, 1.0):
                o = to_frac @ (t[0] * ex + t[1] * ey + (args.vacuum_depth + dd + t[2]) * ez)
                plane = runner.plane(o, o + np.array(v1, float), o + np.array(v2, float),
                                     nxv + 1, nyv + 1, tag=f'vacuum{int(dd):+d}')
                vac[dd] = -plane[:nxv, :nyv] * RY_TO_V
            dv_shift = float(vac[0.0].mean())
            spread = max(float(v.max()) for v in vac.values()) - min(float(v.min()) for v in vac.values())
            print(f"    vacuum plane at cfg depth {args.vacuum_depth:.3f} Å: ΔV_s = {dv_shift:+.4f} V → subtracted; "
                  f"spread over the planes at depth ±1 Å: {spread * 1000:.2f} mV")
            if spread > 5e-3:
                print("    WARNING: ΔV is not constant around the vacuum plane — the planes are not in "
                      "the vacuum, or the IAM (dstart) density is not neutral; check ':NEC01' and "
                      "'Normalization of interstitial' in case.outputd and regenerate case.vcoul_IAM")

        for ly in layers:
            ax, by = ly['ax'], ly['by']
            nx = max(16, round(ax / args.dx))
            ny = max(16, round(by / args.dx))
            w0, wlen = windows[ly['label']]
            nz = max(2, math.ceil(wlen / args.dz))
            if nx * cfg['ncell'][0] < cfg['dim'][0] or ny * cfg['ncell'][1] < cfg['dim'][1]:
                print(f"    WARNING: '{ly['label']}': unit-cell grid {nx}×{ny} px is coarser "
                      f"than dim/NCELL — band limit below ixchel's")
            syms = '+'.join(sorted(set(a['name'] for a in ly['atoms'])))
            print(f"\n  layer '{ly['label']}' ({syms}): {ax:.3f} × {by:.3f} Å, {nx}×{ny} px, "
                  f"ΔV window [{w0:.3f}, {w0 + wlen:.3f}) Å (cfg depth), {nz} steps")

            origin = to_frac @ (t[0] * ex + t[1] * ey + (w0 + t[2]) * ez)
            edges = [np.array(v1, float), np.array(v2, float), np.array(v3, float) * (wlen / period)]
            dV = runner.grid3d(origin, edges, (nx + 1, ny + 1, nz + 1))
            Vs = -dV[:nx, :ny, :] * RY_TO_V - dv_shift              # V_s = −V_C [V]
            dpot = np.trapezoid(Vs, dx=wlen / nz, axis=2) / H2_2PI_ME  # [1/Å]

            kirk = xsf.kirkland_layer(ly, cfg, nx, ny, xsf.FPARAMS)
            tot = kirk + dpot
            print(f"    Kirkland [{kirk.min():.3f}, {kirk.max():.3f}]  "
                  f"ΔV [{dpot.min():.4f}, {dpot.max():.4f}] (mean {dpot.mean():+.4f})  "
                  f"total [{tot.min():.3f}, {tot.max():.3f}] 1/Å")
            fname = f"{out_dir}/{ly['filename']}"
            xsf.write_ixchel_tiff(fname, tot.T)                    # rows = y
            outs.append(fname)
    finally:
        runner.close()

    print(f"\nDone — {len(outs)} file(s) in '{out_dir}/' (one unit cell each; "
          f"use tiff2ixchel.py for ixchel main)")


if __name__ == '__main__':
    main()
