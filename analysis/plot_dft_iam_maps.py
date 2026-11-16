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

Layout of this file: the CORE comes first, the two functions that define the
numbers of the comparison (ring_mask: which pixels belong to the red rings;
integrated_changes: the change DFT vs IAM inside, outside and in total).
After the core: the input (cell from the cfg, V_z from the TIFFs) and main(),
which draws the figure.

Usage:
    python3 plot_dft_iam_maps.py <casedir> <out.png>

Arguments and where they come from:
    <casedir>    the run directory of the Kurasch route.  Read from it:
                   Graphene_ixchel_001.cfg         the crystruct cfg (cell size, atoms)
                   ulm_DFT_001/atompota.tif        written by vcoul2tiff_ulm.py
                   ulm_IAM_001/atompota.tif        (the same from case.vcoul_IAM)
    <out.png>    the figure to write
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
NT_AC, NT_ZZ = 2, 4              # tiling for display

INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d3'
BLUE, ORANGE, MID, RED = '#2a78d6', '#eb6834', '#f0efec', '#e34948'
SEQ = ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#184f95', '#0d366b']


# ═══════════════════════════════════════════════════════════════
# CORE: the numbers of the comparison
#
# The maps themselves are simple: diff = DFT - IAM and rel = diff / IAM, pixel
# by pixel (first lines of main).  These two functions define the areas and
# the integrated changes of panel g (paper Fig. 6 e).
# ═══════════════════════════════════════════════════════════════

def ring_mask(shape, radius, atoms, a_zz, b_ac):
    """Pixels within `radius` Å of any atom of one periodic cell."""
    n_ac, n_zz = shape
    u = (np.arange(n_ac) / n_ac)[:, None]           # position of every pixel row (armchair), as a fraction of the cell
    v = (np.arange(n_zz) / n_zz)[None, :]           # the same for every pixel column (zigzag)
    mask = np.zeros(shape, bool)
    for fu, fv in atoms:
        du = (u - fu + 0.5) % 1.0 - 0.5             # distance to the atom, to its nearest periodic image,
        dv = (v - fv + 0.5) % 1.0 - 0.5             # in fractions of the cell: always within -1/2 ... 1/2
        mask |= np.hypot(du * b_ac, dv * a_zz) <= radius            # the distance in Å
    return mask


def integrated_changes(iam, dft, ring):
    """Change of the summed V_z, DFT against IAM, in per cent: over the whole cell ('cell', the change
    of the mean inner potential), inside the rings ('ring') and outside them ('rest')."""
    return {
        'cell': (dft.sum() / iam.sum() - 1) * 100,
        'ring': (dft[ring].sum() / iam[ring].sum() - 1) * 100,
        'rest': (dft[~ring].sum() / iam[~ring].sum() - 1) * 100,
    }


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: the input (cfg, TIFFs) and the figure
# ═══════════════════════════════════════════════════════════════

# ── input ──────────────────────────────────────────────────────

def read_cell(casedir):
    """Cell size (zigzag x, armchair y) [Å] and the atoms as (armchair, zigzag)
    fractions from the ixchel cfg (cfg positions are [x, y])."""
    text = (Path(casedir) / 'Graphene_ixchel_001.cfg').read_text()
    # the two cfg entries that are read (x = zigzag, y = armchair):
    #     size_uc = { x = 2.4609990418; y = 4.2625753778; z = 14.9999940395; };
    #     positions = (
    #         [ 0.5000000050, 0.1666666650 ],
    #         [ -0.0000000050, 0.3333333350 ], ...
    #     );
    size = re.search(r'size_uc\s*=\s*\{\s*x\s*=\s*([0-9.]+);\s*y\s*=\s*([0-9.]+);', text)
    block = re.search(r'positions\s*=\s*\((.*?)\);', text, re.S).group(1)
    atoms = [(float(y) % 1.0, float(x) % 1.0)                   # (armchair, zigzag), inside the cell
             for x, y in re.findall(r'\[\s*([-0-9.Ee+]+)\s*,\s*([-0-9.Ee+]+)\s*\]', block)]
    return float(size.group(1)), float(size.group(2)), atoms


def load(casedir, kind):
    """V_z [V*Å] as [armchair, zigzag] (TIFF page 1 rows = cfg y = armchair)."""
    d = tifffile.imread(Path(casedir) / f'ulm_{kind}_001' / 'atompota.tif', key=1)
    return d.astype(np.float64) * H2


# ── the figure and the command line ────────────────────────────

def main():
    casedir, out = sys.argv[1], sys.argv[2]

    # ── the data: V_z of one cell for IAM and DFT, their difference, the changes integrated over areas ──
    a_zz, b_ac, atoms = read_cell(casedir)
    iam, dft = load(casedir, 'IAM'), load(casedir, 'DFT')
    n_ac = iam.shape[0]
    diff = dft - iam
    rel = diff / iam * 100.0

    bond = b_ac / 3                                 # C-C bond length: the armchair period holds three of them
    ring = ring_mask(iam.shape, bond / 4, atoms, a_zz, b_ac)      # red ring: diameter = bond / 2
    stats = integrated_changes(iam, dft, ring)

    # ── the figure: maps a-d in the upper row, line scan e, f and the numbers g in the lower row ──
    # tile(): the cell repeated NT_AC x NT_ZZ times for display
    tile = lambda a: np.tile(a, (NT_AC, NT_ZZ))    # noqa: E731
    ext = [0, NT_AC * b_ac, 0, NT_ZZ * a_zz]        # size of the displayed area [Å]: armchair, zigzag

    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                         'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig = plt.figure(figsize=(12.0, 7.6), dpi=150)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.82], hspace=0.42, wspace=0.62,
                          left=0.06, right=0.955, top=0.93, bottom=0.08)

    def image(col, arr, title, label, **style):
        """One map of the top row with its colour bar."""
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(tile(arr).T, origin='lower', extent=ext, interpolation='nearest', aspect='equal', **style)
        ax.set_title(title, loc='left', fontsize=9.5)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label(label)
        ax.set(xlabel='armchair x [Å]', ylabel='zigzag y [Å]')
        return ax

    # panels a and b share one logarithmic grey scale; c and d have their own colour maps
    lognorm = LogNorm(vmin=min(iam.min(), dft.min()), vmax=max(iam.max(), dft.max()))
    image(0, iam, 'a  IAM (dstart), log scale', 'V$_z$ [V·Å]', cmap='gray', norm=lognorm)
    image(1, dft, 'b  DFT (SCF), log scale', 'V$_z$ [V·Å]', cmap='gray', norm=lognorm)
    lim = np.abs(diff).max()
    image(2, diff, 'c  DFT − IAM (blue = lower in DFT)', 'ΔV$_z$ [V·Å]',
          cmap=LinearSegmentedColormap.from_list('div', [BLUE, MID, RED]), vmin=-lim, vmax=lim)
    ax = image(3, rel, 'd  (DFT − IAM) / IAM', 'relative change [%]', vmin=rel.min(), vmax=0,
               cmap=LinearSegmentedColormap.from_list('seq', SEQ[::-1]))       # dark = larger decrease
    # red ring around every atom of the display (paper Fig. 6 c)
    for i in range(NT_AC):
        for j in range(NT_ZZ):
            for fu, fv in atoms:
                ax.add_patch(plt.Circle(((fu + i) * b_ac, (fv + j) * a_zz), bond / 4,
                                        fill=False, lw=0.8, ec=RED))

    # panel e: line scan along armchair through the atom row at zigzag y = 0 (TIFF column 0)
    x = np.arange(NT_AC * n_ac) * b_ac / n_ac
    line_iam = np.tile(iam[:, 0], NT_AC)
    line_dft = np.tile(dft[:, 0], NT_AC)
    line_rel = np.tile(rel[:, 0], NT_AC)
    ax = fig.add_subplot(gs[1, 0:2])
    ax.semilogy(x, line_iam, color=BLUE, lw=1.5, label='IAM')
    ax.semilogy(x, line_dft, color=ORANGE, lw=1.5, label='DFT')
    ax.grid(True, which='major', color=GRID, lw=0.6)
    ax.set(xlim=(x[0], x[-1]), ylim=(4.0, None), ylabel='V$_z$ [V·Å]',
           xlabel='armchair x [Å] (line through the atoms, zigzag y = 0)')
    ax.set_title('e  line scan', loc='left', fontsize=9.5)
    # legend and direct labels in the hexagon centre (line minimum) nearest to the middle
    minima = np.where((line_iam <= np.roll(line_iam, 1)) & (line_iam <= np.roll(line_iam, -1))
                      & (line_iam < 1.5 * line_iam.min()))[0]
    centre = int(minima[np.argmin(np.abs(x[minima] - x[-1] / 2))])
    ax.legend(loc='upper center', bbox_to_anchor=(x[centre] / x[-1], 0.99), frameon=False)
    ax.annotate('IAM', (x[centre], line_iam[centre]), xytext=(4, 6), textcoords='offset points', color=INK2)
    ax.annotate('DFT', (x[centre], line_dft[centre]), xytext=(4, -12), textcoords='offset points', color=INK2)

    # panel f: the relative change on the same line
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(x, line_rel, color=INK2, lw=1.5)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set(xlim=(x[0], x[-1]), xlabel='armchair x [Å]', ylabel='(DFT − IAM) / IAM [%]')
    ax.set_title('f  relative change on the line', loc='left', fontsize=9.5)

    # panel g: no plot, the three integrated changes as text
    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')
    ax.set_title('g  integrated change (Fig. 6 e)', loc='left', fontsize=9.5)
    for i, (label, val, c) in enumerate((('red ring (d = bond/2)', stats['ring'], RED),
                                         ('outside the rings', stats['rest'], INK),
                                         ('total cell (MIP)', stats['cell'], INK))):
        ax.text(0.34, 0.80 - 0.17 * i, f'{val:+.1f} %', transform=ax.transAxes, ha='right',
                color=c, fontsize=12, fontweight='bold')
        ax.text(0.39, 0.80 - 0.17 * i, label, transform=ax.transAxes, color=INK2, fontsize=9)
    ax.text(0.0, 0.18, 'paper, pristine graphene (p. 400):\nMIP change 15.5 %', transform=ax.transAxes,
            color=INK2, fontsize=8.5)

    fig.suptitle('Pristine graphene, projected potential [001]: DFT vs IAM after identical Kurasch processing '
                 '(30 ppÅ, nn cutoff 50 kV, min V$_s$ = 0)', fontsize=10.5, x=0.05, ha='left')
    fig.savefig(out, facecolor='white')
    print(f"cell {a_zz:.5f} x {b_ac:.5f} Å (b/a = {b_ac / a_zz:.6f}); "
          f"V_z range IAM [{iam.min():.2f}, {iam.max():.2f}], DFT [{dft.min():.2f}, {dft.max():.2f}] V*Å")
    print(f"cell {stats['cell']:+.2f} %, ring (r = {bond / 4:.3f} Å, {ring.sum()} px/cell) {stats['ring']:+.2f} %, "
          f"rest {stats['rest']:+.2f} %, rel. range {rel.min():+.1f} … {rel.max():+.1f} %")


if __name__ == '__main__':
    main()
