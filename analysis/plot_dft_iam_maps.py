#!/usr/bin/env python3
"""
plot_dft_iam_maps.py — graphene DFT vs IAM projected potential, Kurasch route

The analysis of Kurasch et al. (2011) Fig. 6 (panels a-c, e) applied to pristine
graphene: projected potential of the IAM (dstart) and the DFT (SCF) Coulomb
potential after identical processing (30 ppÅ grid, nn cutoff 50 kV, min V_s = 0,
projection along [001]; vcoul2tiff_ulm.py), their difference, the relative
difference, a line scan along the armchair direction through the atoms, and the
integrated changes inside the paper's "red ring" (diameter = half the bond length)
around every atom and over the whole cell.  The paper gives only the cell
number for graphene in the text (p. 400: 15.5 %).

Display: one-cell TIFFs (126 x 74 px: rows = armchair a*sqrt(3), columns = zigzag
a = 2.461 Å; cell size read from <casedir>/Graphene_ixchel_001.cfg) drawn with
armchair horizontal, so that bonds run horizontally as in the paper's Fig. 4, and
tiled 2 x 4, i.e. about the field of the paper's B/O supercell.

Usage:
    python3 plot_dft_iam_maps.py <casedir> <out.png>
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import tifffile                                                 # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

H2 = 47.88                       # V*Å², ixchel 1/Å -> V*Å
A_ZZ = B_AC = None               # Å, cell of Graphene_ixchel_001.cfg (set in main)
ATOMS = None                     # (armchair, zigzag) fractions of the atoms, from the cfg
NT_AC, NT_ZZ = 2, 4              # tiling for display

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
BLUE, ORANGE, MID, RED = '#2a78d6', '#eb6834', '#f0efec', '#e34948'
SEQ = ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b']


def read_cell(casedir):
    """Cell size (zigzag x, armchair y) [Å] and the atoms as (armchair, zigzag)
    fractions from the ixchel cfg (cfg positions are [x, y])."""
    text = (Path(casedir) / 'Graphene_ixchel_001.cfg').read_text()
    m = re.search(r'size_uc\s*=\s*\{\s*x\s*=\s*([0-9.]+);\s*y\s*=\s*([0-9.]+);', text)
    block = re.search(r'positions\s*=\s*\((.*?)\);', text, re.S).group(1)
    atoms = [(float(y) % 1.0, float(x) % 1.0)
             for x, y in re.findall(r'\[\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\]', block)]
    return float(m.group(1)), float(m.group(2)), atoms


def load(casedir, kind):
    """V_z [V*Å] as [armchair, zigzag] (TIFF page 1 rows = cfg y = armchair)."""
    d = tifffile.imread(Path(casedir) / f'ulm_{kind}_001' / 'atompota.tif', key=1)
    return d.astype(np.float64) * H2


def ring_mask(shape, radius):
    """Pixels within `radius` Å of any atom of one periodic cell."""
    n_ac, n_zz = shape
    u = (np.arange(n_ac) / n_ac)[:, None]
    v = (np.arange(n_zz) / n_zz)[None, :]
    m = np.zeros(shape, bool)
    for fu, fv in ATOMS:
        du = (u - fu + 0.5) % 1.0 - 0.5
        dv = (v - fv + 0.5) % 1.0 - 0.5
        m |= np.hypot(du * B_AC, dv * A_ZZ) <= radius
    return m


def main():
    global A_ZZ, B_AC, ATOMS
    casedir, out = sys.argv[1], sys.argv[2]
    A_ZZ, B_AC, ATOMS = read_cell(casedir)
    iam, dft = load(casedir, 'IAM'), load(casedir, 'DFT')
    n_ac, n_zz = iam.shape
    diff = dft - iam
    rel = diff / iam * 100.0

    bond = B_AC / 3
    ring = ring_mask(iam.shape, bond / 4)          # red ring: diameter = bond / 2
    stats = {
        'cell': (dft.sum() / iam.sum() - 1) * 100,
        'ring': (dft[ring].sum() / iam[ring].sum() - 1) * 100,
        'rest': (dft[~ring].sum() / iam[~ring].sum() - 1) * 100,
    }

    tile = lambda a: np.tile(a, (NT_AC, NT_ZZ))    # noqa: E731
    ext = [0, NT_AC * B_AC, 0, NT_ZZ * A_ZZ]
    show = dict(origin='lower', extent=ext, interpolation='nearest', aspect='equal')

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig = plt.figure(figsize=(12.0, 7.6), dpi=150)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.82], hspace=0.42, wspace=0.62,
                          left=0.06, right=0.955, top=0.93, bottom=0.08)

    lognorm = LogNorm(vmin=min(iam.min(), dft.min()), vmax=max(iam.max(), dft.max()))
    for col, (arr, title) in enumerate(((iam, 'a  IAM (dstart), log scale'),
                                         (dft, 'b  DFT (SCF), log scale'))):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(tile(arr).T, cmap='gray', norm=lognorm, **show)
        ax.set_title(title, loc='left', fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label('V$_z$ [V·Å]')

    lim = np.abs(diff).max()
    div = LinearSegmentedColormap.from_list('div', [BLUE, MID, RED])
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(tile(diff).T, cmap=div, vmin=-lim, vmax=lim, **show)
    ax.set_title('c  DFT − IAM (blue = lower in DFT)', loc='left', fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('ΔV$_z$ [V·Å]')

    seq = LinearSegmentedColormap.from_list('seq', SEQ[::-1])     # dark = larger decrease
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(tile(rel).T, cmap=seq, vmin=rel.min(), vmax=0, **show)
    ax.set_title('d  (DFT − IAM) / IAM', loc='left', fontsize=9.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('relative change [%]')
    # red ring around every atom of the display (paper Fig. 6 c)
    for i in range(NT_AC):
        for j in range(NT_ZZ):
            for fu, fv in ATOMS:
                ax.add_patch(plt.Circle(((fu + i) * B_AC, (fv + j) * A_ZZ), bond / 4,
                                        fill=False, lw=0.8, ec=RED))
    for a in fig.axes[0:8:2]:
        a.set_xlabel('armchair x [Å]')
        a.set_ylabel('zigzag y [Å]')

    # line scan along armchair through the atom row at zigzag y = 0 (TIFF column 0)
    x = np.arange(NT_AC * n_ac) * B_AC / n_ac
    li, ld, lr = np.tile(iam[:, 0], NT_AC), np.tile(dft[:, 0], NT_AC), np.tile(rel[:, 0], NT_AC)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.semilogy(x, li, color=BLUE, lw=1.5, label='IAM')
    ax.semilogy(x, ld, color=ORANGE, lw=1.5, label='DFT')
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel('armchair x [Å] (line through the atoms, zigzag y = 0)')
    ax.set_ylabel('V$_z$ [V·Å]')
    ax.set_title('e  line scan', loc='left', fontsize=9.5)
    ax.set_ylim(4.0, None)
    # legend and direct labels in the hexagon centre (line minimum) nearest to the middle
    mins = np.where((li <= np.roll(li, 1)) & (li <= np.roll(li, -1)) & (li < 1.5 * li.min()))[0]
    k = int(mins[np.argmin(np.abs(x[mins] - x[-1] / 2))])
    ax.legend(loc='upper center', bbox_to_anchor=(x[k] / x[-1], 0.99), frameon=False)
    ax.annotate('IAM', (x[k], li[k]), xytext=(4, 6), textcoords='offset points', color=INK2)
    ax.annotate('DFT', (x[k], ld[k]), xytext=(4, -12), textcoords='offset points', color=INK2)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, lr, color=INK2, lw=1.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_xlim(x[0], x[-1])
    ax.set_xlabel('armchair x [Å]')
    ax.set_ylabel('(DFT − IAM) / IAM [%]')
    ax.set_title('f  relative change on the line', loc='left', fontsize=9.5)

    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    ax.set_title('g  integrated change (Fig. 6 e)', loc='left', fontsize=9.5)
    rows = [('red ring (d = bond/2)', stats['ring'], RED),
            ('outside the rings', stats['rest'], INK),
            ('total cell (MIP)', stats['cell'], INK)]
    for i, (label, val, c) in enumerate(rows):
        ax.text(0.34, 0.80 - 0.17 * i, f'{val:+.1f} %', transform=ax.transAxes, ha='right',
                color=c, fontsize=12, fontweight='bold')
        ax.text(0.39, 0.80 - 0.17 * i, label, transform=ax.transAxes, color=INK2, fontsize=9)
    ax.text(0.0, 0.18, 'paper, pristine graphene (p. 400):\nMIP change 15.5 %', transform=ax.transAxes,
            color=INK2, fontsize=8.5)

    fig.suptitle('Pristine graphene, projected potential [001]: DFT vs IAM after identical Kurasch processing '
                 '(30 ppÅ, nn cutoff 50 kV, min V$_s$ = 0)', fontsize=10.5, x=0.05, ha='left')
    fig.savefig(out, facecolor='white')
    print(f"cell {A_ZZ:.5f} x {B_AC:.5f} Å (b/a = {B_AC / A_ZZ:.6f}); "
          f"V_z range IAM [{iam.min():.2f}, {iam.max():.2f}], DFT [{dft.min():.2f}, {dft.max():.2f}] V*Å")
    print(f"cell {stats['cell']:+.2f} %, ring (r = {bond / 4:.3f} Å, {ring.sum()} px/cell) {stats['ring']:+.2f} %, "
          f"rest {stats['rest']:+.2f} %, rel. range {rel.min():+.1f} … {rel.max():+.1f} %")


if __name__ == '__main__':
    main()
