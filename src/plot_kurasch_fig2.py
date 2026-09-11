#!/usr/bin/env python3
"""
plot_kurasch_fig2.py — Reproduction of Kurasch et al. 2011, Fig. 2

Radial line scan of the WIEN2k Coulomb potential through a single atom,
compared with the Kirkland IAM potential (and optionally the bare Z/r).

Expected input: a WIEN2k vcoul XSF with EXACTLY ONE atom in a
large cell (paper: 10x10x10 Å). If there are several atoms, the first one is taken.

The reader's axis order is detected automatically (as in vcoul2tiff_ulm4).

Usage:
    python3 plot_kurasch_fig2.py --xsf C_single_vcoul.xsf [--Z 6] \\
        [--order auto|F|C] [--coulomb] [-o fig2.png]
"""
import argparse, itertools, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

RY_TO_V = 13.605693122994
E2      = 14.399                 # e^2/(4 pi eps0) [eV*A] for the bare Coulomb potential

# ── Kirkland IAM potential (abTEM parameters + scaling) ─────────────────
# kappa from abtem/core/constants.py; the potential comes out in volts (eV/e).
KAPPA = 0.0208865737082965
# Raw parameters (abtem kirkland.json), layout [a(3), b(3), c(3), d(3)]:
_KIRK = {
    'C': [[0.212080767, 0.199811865, 0.168254385],
          [0.208605417, 0.208610186, 5.57870773],
          [0.14204836,  0.363830672, 0.000835012],
          [1.33311887,  3.80800263,  0.040398262]],
    # further elements: add the rows from abtem kirkland.json here
}

def kirkland_potential(r, symbol='C'):
    """Isolated Kirkland atomic potential V(r) [V]. r in Angstrom."""
    p = np.array(_KIRK[symbol])
    a = np.pi * p[0] / KAPPA
    b = 2.0 * np.pi * np.sqrt(p[1])
    c = np.pi**1.5 * p[2] / p[3]**1.5 / KAPPA
    d = np.pi**2 / p[3]
    r = np.asarray(r, float)
    return sum(a[i]*np.exp(-b[i]*r)/r + c[i]*np.exp(-d[i]*r**2) for i in range(3))


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


def normalize_like_paper(r, v_ry):
    """V_s = -V_C (volts), far field (smallest value) -> 0."""
    v_v = -v_ry * RY_TO_V
    far = v_v[r > r.max()*0.6]                  # outer region = plateau
    v_shift = np.median(far) if far.size else v_v.min()
    return v_v - v_shift


def main():
    ap = argparse.ArgumentParser(description='Reproduce Kurasch Fig. 2')
    ap.add_argument('--xsf', required=True, help='WIEN2k vcoul XSF (1 atom, large cell)')
    ap.add_argument('--Z', type=int, default=6, help='Nuclear charge (default 6 = C)')
    ap.add_argument('--symbol', default='C', help='Element for the Kirkland IAM')
    ap.add_argument('--order', choices=['auto','F','C'], default='auto')
    ap.add_argument('--coulomb', action='store_true', help='additionally show the bare Z/r')
    ap.add_argument('-o', default='kurasch_fig2.png')
    args = ap.parse_args()

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
    v = normalize_like_paper(r, v_ry)

    r_ref = np.logspace(-2, np.log10(min(cell)/2*0.95), 400)
    v_kirk = kirkland_potential(r_ref, args.symbol)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Radial potential — WIEN2k DFT vs Kirkland IAM\n'
                 '(cf. Kurasch et al. 2011, Fig. 2)', fontsize=11)
    for axp, (xlim, xscale, title) in zip(
            [axL, axR],
            [((0.01, 5.0), 'log',    'Log scale (as in Fig. 2)'),
             ((0.0,  0.5), 'linear', 'Near-core region 0–0.5 Å')]):
        m = (r > 5e-4) & (v > 0.05)
        axp.semilogy(r[m], v[m], 'r.', ms=3, label='WIEN2k vcoul (DFT)')
        axp.semilogy(r_ref, v_kirk, 'k-', lw=1.5, label='Kirkland IAM')
        if args.coulomb:
            axp.semilogy(r_ref, args.Z*E2/r_ref, 'b:', lw=1,
                         alpha=.6, label=f'Coulomb Z={args.Z} (unscreened)')
        axp.set_xscale(xscale); axp.set_xlim(xlim); axp.set_ylim(0.05, 1e4)
        axp.set_xlabel('Distance from core [Å]'); axp.set_ylabel('Potential [V]')
        axp.set_title(title, fontsize=9); axp.legend(fontsize=8)
        axp.grid(True, which='both', alpha=.25)
        axp.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    plt.tight_layout(); plt.savefig(args.o, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {args.o}")

    # quantitative comparison at sample points
    print("\n  r [Å]   WIEN2k [V]   Kirkland [V]   Dev.")
    for rr in [0.1, 0.2, 0.5, 1.0, 1.5]:
        j = np.argmin(np.abs(r - rr))
        vk = kirkland_potential(r[j], args.symbol)
        if v[j] > 0:
            print(f"  {r[j]:5.2f}  {v[j]:10.1f}   {vk:10.1f}   "
                  f"{100*(v[j]-vk)/vk:+6.1f}%")


if __name__ == '__main__':
    main()
