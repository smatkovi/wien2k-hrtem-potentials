#!/usr/bin/env python3
"""
plot_kurasch_fig2.py — Reproduction of Kurasch et al. 2011, Fig. 2

Radial line scan of the WIEN2k Coulomb potential through a single atom,
compared with the Doyle-Turner and Kirkland isolated-atom potentials and the
bare Coulomb potential of the nucleus.

Paper (p. 397): "The WIEN2k potential was obtained by putting a single carbon
atom into a 10 Å × 10 Å × 10 Å unit cell and calculating a linescan of the
electrostatic potential with a resolution of 250 points per Angstrom (ppÅ).
Far from the core, the WIEN2k potential approaches a non-zero constant value.
In order to obtain the usual normalization, the potential was shifted (i.e.,
smallest value was set to zero)."  The curve is the WIEN2k *starting*
potential (Fig. 2 caption): case.vcoul from `x lapw0` on the `x dstart`
density, before the first SCF cycle.

Two input modes:
  --vcoul/--struct  (paper): lapw5c line scan at --ppa points/Å (default 250)
                    from the first atom along --axis over half the period
  --xsf             line scans along the grid axes of a 3ddens/lapw5grid XSF
                    (resolution of the grid; axis order detected automatically)
Both are normalised as in the paper: V_s = -V_C, smallest value set to zero.

Usage:
    python3 plot_kurasch_fig2.py --vcoul C_single.vcoul --struct C_single.struct [-o fig2.png]
    python3 plot_kurasch_fig2.py --xsf C_single_vcoul.xsf [--order auto|F|C] [-o fig2.png]
"""
import argparse, itertools, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RY_TO_V = 13.605693122994
E2      = 14.399                 # e^2/(4 pi eps0) [eV*A] for the bare Coulomb potential

A0E     = 0.529177210903 * 14.399645   # a0*e [V*Å^2]

# ── Kirkland isolated-atom potential (ixchel dat/fparams.dat) ───────────
# fparams.dat layout per element: a1 b1 a2 b2 / a3 b3 c1 d1 / c2 d2 c3 d3
_KIRK = {
    'C': [[2.53148417e-01, 2.08761775e-01, 1.66953010e-01, 5.69804854e+00],
          [1.60250013e-01, 2.08774166e-01, 1.41633354e-01, 1.34112859e+00],
          [3.60244465e-01, 3.81665326e+00, 8.66192838e-04, 4.12204523e-02]],
    # further elements: add the three rows from fparams.dat here
}

def kirkland_potential(r, symbol='C'):
    """Isolated Kirkland atomic potential V(r) [V], r in Å (Kirkland, Advanced
    Computing in Electron Microscopy, eq. C.19)."""
    p = np.array(_KIRK[symbol]).ravel()
    a, b, c, d = p[[0, 2, 4]], p[[1, 3, 5]], p[[6, 8, 10]], p[[7, 9, 11]]
    r = np.asarray(r, float)
    return (2 * np.pi**2 * A0E * sum(a[i] / r * np.exp(-2 * np.pi * r * np.sqrt(b[i])) for i in range(3))
            + 2 * np.pi**2.5 * A0E * sum(c[i] * d[i]**-1.5 * np.exp(-np.pi**2 * r**2 / d[i]) for i in range(3)))


# ── Doyle-Turner isolated-atom potential ─────────────────────────────────
# Doyle & Turner, Acta Cryst. A24, 390 (1968): f_e(s) = sum a_i exp(-b_i s^2),
# s = sin(theta)/lambda [1/Å].  V(r) = 2 pi a0 e sum a_i (4 pi/b_i)^(3/2) exp(-4 pi^2 r^2/b_i)
_DT = {
    'C': [[0.7307, 1.1951, 0.4563, 0.1247], [36.9951, 11.2966, 2.8139, 0.3456]],
}

def doyle_turner_potential(r, symbol='C'):
    """Isolated Doyle-Turner atomic potential V(r) [V], r in Å (finite at r = 0)."""
    a, b = (np.array(v) for v in _DT[symbol])
    r = np.asarray(r, float)
    return 2 * np.pi * A0E * sum(a[i] * (4 * np.pi / b[i])**1.5 * np.exp(-4 * np.pi**2 * r**2 / b[i])
                                 for i in range(4))


# ── XSF reader with automatic axis detection + PRIMCOORD ────────────────
def _roughness(d):
    return sum(np.mean(np.abs(np.diff(d, axis=ax))) for ax in range(3))

def read_xsf(path, order='auto'):
    lines = open(path).readlines()
    atoms, vecs, i = [], {}, 0
    data = cell = shape = None
    while i < len(lines):
        tok = lines[i].strip()
        if tok in ('PRIMVEC', 'CONVVEC'):
            vecs[tok] = np.array([[float(x) for x in lines[i+1+k].split()[:3]]
                                  for k in range(3)])
            i += 4; continue
        if tok == 'PRIMCOORD':
            n = int(lines[i+1].split()[0])
            for k in range(n):
                p = lines[i+2+k].split()
                atoms.append({'Z': int(float(p[0])),
                              'cart': np.array([float(x) for x in p[1:4]])})
            i += 2 + n; continue
        if ('DATAGRID_3D' in tok and 'END' not in tok
                and 'BEGIN_BLOCK' not in tok):
            i += 1
            nx, ny, nz = map(int, lines[i].split()); i += 1
            i += 1  # origin
            va = np.array(list(map(float, lines[i].split()))); i += 1
            vb = np.array(list(map(float, lines[i].split()))); i += 1
            vc = np.array(list(map(float, lines[i].split()))); i += 1
            vals = []
            while i < len(lines) and len(vals) < nx*ny*nz:
                if 'END_DATAGRID' in lines[i]: break
                vals.extend(float(x) for x in lines[i].split()); i += 1
            arr = np.asarray(vals[:nx*ny*nz], float)
            if order in ('F', 'C'):
                data = arr.reshape((nx,ny,nz), order=order); used = order
            else:
                dF = arr.reshape((nx,ny,nz), order='F')
                dC = arr.reshape((nx,ny,nz), order='C')
                if _roughness(dF) <= _roughness(dC): data, used = dF, 'F'
                else:                                 data, used = dC, 'C'
            print(f"    Axis order: order='{used}'")
            cell = (np.linalg.norm(va), np.linalg.norm(vb), np.linalg.norm(vc))
            shape = (nx, ny, nz)
            continue
        i += 1
    if data is None:
        sys.exit("Error: No DATAGRID_3D found.")
    return data, cell, shape, atoms, vecs


def primitive_vectors(vecs):
    """Primitive lattice vectors (rows) from PRIMVEC, or None if unusable.
    The PRIMVEC matrix can come out transposed (seen in 3ddens exports of the
    CXY graphene lattice), so the orientation is taken in which every CONVVEC
    vector is an integer combination of the primitive vectors."""
    P, C = vecs.get('PRIMVEC'), vecs.get('CONVVEC')
    if P is None or C is None:
        return None
    for Q in (P, P.T):
        if abs(np.linalg.det(Q)) < 1e-8:
            continue
        N = C @ np.linalg.inv(Q)
        if np.allclose(N, np.round(N), atol=1e-3):
            return Q
    return None


# ── radial line scan along all three axes through the atom ──────────────
def radial_linescan(data, cell, atom_cart, prim=None):
    (ax, by, cz), (nx, ny, nz) = cell, data.shape
    n_int = (nx-1, ny-1, nz-1)                  # XSF general grid
    L = np.array((ax, by, cz))
    # The grid is periodic: wrap the atom (PRIMCOORD may lie outside the cell) and
    # index modulo n_int (the last plane duplicates the first).  With known
    # primitive vectors, all lattice images of the atom are candidates and the one
    # with the deepest potential is taken: 3ddens leaves out the sphere part at
    # centring-translated sites of CXY/CYZ lattices, where PRIMCOORD puts the atoms.
    shifts = ([np.zeros(3)] if prim is None else
              [np.array(n) @ prim for n in itertools.product(range(-2, 3), repeat=3)])
    ac = idx0 = None
    for t in shifts:
        a_img = np.mod(np.asarray(atom_cart, float) + t, L)
        i_img = tuple(int(round(a_img[k]/L[k]*n_int[k])) % n_int[k] for k in range(3))
        if idx0 is None or data[i_img] < data[idx0]:
            ac, idx0 = a_img, i_img
    r_all, v_all = [], []
    for axis in range(3):
        grid = np.arange((nx,ny,nz)[axis]) * L[axis] / n_int[axis]
        d = grid - ac[axis]
        d = np.where(np.abs(d) > L[axis]/2, d - np.sign(d)*L[axis], d)
        if   axis == 0: v = data[:, idx0[1], idx0[2]]
        elif axis == 1: v = data[idx0[0], :, idx0[2]]
        else:           v = data[idx0[0], idx0[1], :]
        r_all.append(np.abs(d)); v_all.append(v)
    r = np.concatenate(r_all); v = np.concatenate(v_all)
    o = np.argsort(r)
    return r[o], v[o], ac, idx0


def normalize_like_paper(v_ry):
    """V_s = -V_C [V], shifted so that the smallest value is zero (paper p. 397)."""
    v_v = -np.asarray(v_ry, float) * RY_TO_V
    return v_v - v_v.min()


# ── lapw5 line scan (paper mode) ─────────────────────────────────────────
def lapw5_linescan(struct_path, vcoul_path, ppa=250.0, axis=0):
    """V_C [Ry] along the lattice axis `axis` from the first atom over half the
    period, ppa points per Å including both ends, evaluated by lapw5c.
    Returns r [Å], V_C [Ry], atom position (fractional), lattice constants [Å]."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lapw5grid import BOHR, Lapw5Runner, read_struct
    st = read_struct(struct_path)
    if any(abs(x - 90.0) > 1e-6 for x in st['angles']):
        sys.exit("Error: line scan mode needs an orthogonal cell")
    L = np.array(st['abc_bohr']) * BOHR
    pos = np.array(st['atoms'][0]['positions'][0], float)
    length = L[axis] / 2
    n = int(round(length * ppa)) + 1
    end = pos.copy(); end[axis] += 0.5
    other = pos.copy(); other[(axis + 1) % 3] += 0.25   # second, unused row of the lapw5 plane
    run = Lapw5Runner(struct_path, vcoul_path, norm='val', nproc=1)
    try:
        plane = run.plane(tuple(pos), tuple(end), tuple(other), n, 2)
    finally:
        run.close()
    return np.linspace(0.0, length, n), plane[:, 0], pos, L


def main():
    ap = argparse.ArgumentParser(description='Reproduce Kurasch et al. 2011, Fig. 2')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--vcoul', help='case.vcoul of the starting potential (lapw5 line scan, paper mode)')
    src.add_argument('--xsf', help='WIEN2k vcoul XSF (1 atom, large cell)')
    ap.add_argument('--struct', help='case.struct (with --vcoul)')
    ap.add_argument('--ppa', type=float, default=250.0, help='line scan points per Å (paper: 250)')
    ap.add_argument('--axis', type=int, choices=[0, 1, 2], default=0, help='lattice axis of the line scan')
    ap.add_argument('--Z', type=int, default=6, help='Nuclear charge (default 6 = C)')
    ap.add_argument('--symbol', default='C', help='Element for the Kirkland and Doyle-Turner potentials')
    ap.add_argument('--order', choices=['auto','F','C'], default='auto')
    ap.add_argument('-o', default='kurasch_fig2.png')
    args = ap.parse_args()

    if args.vcoul:
        if not args.struct:
            sys.exit("Error: --vcoul needs --struct")
        print(f"[1] lapw5c line scan of {args.vcoul} at {args.ppa:g} ppÅ along axis {args.axis}")
        r, v_ry, pos, L = lapw5_linescan(args.struct, args.vcoul, args.ppa, args.axis)
        print(f"    cell {L[0]:.3f} x {L[1]:.3f} x {L[2]:.3f} Å, atom at {pos}, "
              f"{r.size} points over 0..{r[-1]:.3f} Å, V_C(0) = {v_ry[0]:.1f} Ry")
        src_label = f'lapw5 line scan, {args.ppa:g} ppÅ'
    else:
        print(f"[1] XSF: {args.xsf}")
        data, cell, shape, atoms, vecs = read_xsf(args.xsf, args.order)
        print(f"    Grid {shape[0]}x{shape[1]}x{shape[2]}, "
              f"cell {cell[0]:.2f}x{cell[1]:.2f}x{cell[2]:.2f} A, {len(atoms)} atom(s)")
        if not atoms:
            sys.exit("No atom in PRIMCOORD.")
        atom = atoms[0]
        print(f"    Atom Z={atom['Z']} at {np.round(atom['cart'],3)} A")
        r, v_ry, ac, idx0 = radial_linescan(data, cell, atom['cart'], primitive_vectors(vecs))
        print(f"    Line scans through {np.round(ac,3)} A (grid index {list(idx0)}, "
              f"V = {data[idx0]:.1f} Ry)")
        src_label = 'XSF grid line scans'
    v = normalize_like_paper(v_ry)
    print(f"    normalisation: V_s = -V_C shifted by {-(-np.asarray(v_ry) * RY_TO_V).min():+.4f} V "
          f"(smallest value -> 0)")

    r_ref = np.logspace(np.log10(2e-3), 1, 800)
    v_kirk = kirkland_potential(r_ref, args.symbol)
    v_dt = doyle_turner_potential(r_ref, args.symbol)
    v_coul = args.Z * E2 / r_ref

    # layout of the paper's Fig. 2: log-log main panel, linear inset 0-0.4 Å
    fig, ax = plt.subplots(figsize=(7.5, 5))
    m = r > 0
    ax.loglog(r_ref, v_kirk, 'g-', lw=1.6, label='Kirkland')
    ax.loglog(r[m], np.where(v[m] > 0, v[m], np.nan), 'r--', lw=1.6, label='WIEN2k')
    ax.loglog(r_ref, v_dt, 'k-', lw=1.0, label='Doyle–Turner')
    ax.loglog(r_ref, v_coul, 'b-', lw=0.7, label='Coulomb of nucleus')
    ax.set_xlim(2e-3, 10); ax.set_ylim(1e-6, 1e5)
    ax.set_xlabel('distance in Å'); ax.set_ylabel('potential in V')
    ax.set_title(f'Single C atom in a 10 Å cell, WIEN2k starting potential ({src_label})\n'
                 'cf. Kurasch et al. 2011, Fig. 2', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    ins = ax.inset_axes([0.09, 0.08, 0.36, 0.5])
    rl = np.linspace(1e-3, 0.4, 400)
    mi = (r > 0) & (r <= 0.4)
    ins.plot(rl, kirkland_potential(rl, args.symbol), 'g-', lw=1.4)
    ins.plot(r[mi], v[mi], 'r--', lw=1.4)
    ins.plot(rl, doyle_turner_potential(rl, args.symbol), 'k-', lw=0.9)
    ins.plot(rl, args.Z * E2 / rl, 'b-', lw=0.7)
    ins.set_xlim(0, 0.4); ins.set_ylim(0, 3000); ins.tick_params(labelsize=7)
    plt.tight_layout(); plt.savefig(args.o, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.o}")

    # quantitative comparison at sample points
    print("\n  r [Å]    WIEN2k [V]   Kirkland [V]  dev.     Doyle-Turner [V]  dev.")
    shown = set()
    for rr in [0.004, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0]:
        j = int(np.argmin(np.abs(r - rr)))
        if r[j] <= 0 or j in shown:            # coarse XSF grids: no sample near rr
            continue
        shown.add(j)
        vk = float(kirkland_potential(r[j], args.symbol))
        vd = float(doyle_turner_potential(r[j], args.symbol))
        print(f"  {r[j]:6.3f}  {v[j]:11.4f}  {vk:11.4f}  {100*(v[j]-vk)/vk:+7.2f}%  {vd:11.4f}  {100*(v[j]-vd)/vd:+8.1f}%")


if __name__ == '__main__':
    main()
