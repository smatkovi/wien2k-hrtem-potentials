#!/usr/bin/env python3
"""
Comparison WIEN2k (DFT+Poisson) vs ixchel (Kirkland IAM)
Generic — reads NCELL/dim/size_uc automatically from the ixchel .cfg.

Displayed is the projected electrostatic potential V_z(x,y) in
Kirkland units [1/Å] (= V·Å / 47.88 V·Å²), as stored in the ixchel TIFFs.

Both sides are reduced to one unit cell with identical geometry:
  WIEN2k:  wien2k_{direction}/atompot*.tif  (output of vcoul2tiffFourier.py)
  ixchel:  ixchel_{direction}/atompot*.tif  (output of ixchel image)

A WIEN2k TIFF with the ixchel dim shape covers the NCELL supercell; any other
shape is ONE unit cell (periodic grid, no repeated edge) and is first
put on the ixchel supercell grid with tiff2ixchel.tile_fourier.  From both
supercell images the unit cell is then extracted by Fourier harmonic
extraction (the unit-cell harmonics sit at multiples of NCELL) and sampled on
round(dim/NCELL) pixels at x_i = i*a/n, so WIEN2k is band-limited to the ixchel
resolution (not the other way round) and peak heights are comparable, also when
dim/NCELL is not an integer.

Usage:
    cd ~/wien2k_tests/Rutile
    python3 compare_all.py -p Rutile_v4 -d 001 100 110
"""

import numpy as np
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import string, sys, re, argparse

try:
    import tifffile
except ImportError:
    print("ERROR: pip install tifffile --break-system-packages")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tiff2ixchel import tile_fourier


# =============================================================================
# Crystallographic directions
# =============================================================================

def get_inplane_directions(hkl_str):
    """In-plane directions (u→x, v→y) for projection [hkl]."""
    mapping = {
        '001': ('[100]', '[010]'),
        '010': ('[100]', '[001]'),
        '100': ('[010]', '[001]'),
        '110': (r'[$\bar{1}$10]', '[001]'),
        '101': ('[010]', r'[$\bar{1}$01]'),
        '011': ('[100]', r'[0$\bar{1}$1]'),
        '111': (r'[$\bar{1}$10]', r'[$\bar{1}\bar{1}$2]'),
    }
    return mapping.get(hkl_str,
                       (r'$u \perp$ [' + hkl_str + ']',
                        r'$v \perp$ [' + hkl_str + ']'))


# =============================================================================
# Read TIFF (shared by WIEN2k and ixchel)
# =============================================================================

def read_ixchel_tiff(path):
    """Reads float32 data from an ixchel-format TIFF (Dir 1 = float32 data)."""
    with tifffile.TiffFile(str(path)) as tif:
        # First look for an explicit float32 page
        for page in tif.pages:
            if page.dtype == np.float32 and page.ndim == 2:
                return page.asarray()
        # Fallback: page index 1 (ixchel convention: 0=RGB, 1=float32, 2=checksums)
        if len(tif.pages) > 1:
            data = tif.pages[1].asarray().astype(np.float32)
            if data.ndim == 2:
                return data
    raise ValueError(f"No float32 data in {path}")


def extract_unit_cell(img, ncell, uc_px):
    """
    One unit cell from a supercell image by Fourier harmonic extraction.
    img   : [Ny, Nx] covering NCx x NCy unit cells, pixel (0,0) at the cell origin
    ncell : (NCx, NCy)
    uc_px : (nx, ny) output pixels per unit cell
    returns [ny, nx] sampled at x_i = i*a/nx, y_j = j*b/ny
    Only the supercell harmonics at multiples of NCELL belong to the lattice
    period; they are kept strictly below the Nyquist frequency of both grids.
    """
    Ny, Nx = img.shape
    ncx, ncy = ncell
    nx, ny = uc_px
    coef = np.fft.fft2(np.asarray(img, dtype=np.float64)) / img.size
    KX = np.fft.fftfreq(Nx, d=1.0 / Nx).astype(int)
    KY = np.fft.fftfreq(Ny, d=1.0 / Ny).astype(int)
    sx = (KX % ncx == 0) & (2 * np.abs(KX) < Nx) & (2 * np.abs(KX // ncx) < nx)
    sy = (KY % ncy == 0) & (2 * np.abs(KY) < Ny) & (2 * np.abs(KY // ncy) < ny)
    target = np.zeros((ny, nx), dtype=complex)
    target[np.ix_((KY[sy] // ncy) % ny, (KX[sx] // ncx) % nx)] = coef[np.ix_(sy, sx)]
    return (np.fft.ifft2(target) * (nx * ny)).real


# =============================================================================
# Parse config
# =============================================================================

def parse_cfg(cfg_path):
    text = re.sub(r'#[^\n]*', '', Path(cfg_path).read_text())      # drop comments

    ncell_m = re.search(r'\bNCELL\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', text)
    dim_m   = re.search(r'\bdim\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', text)

    # ixchel defaults: NCELL = 1×1; dim is required by ixchel main
    ncell = (int(ncell_m.group(1)), int(ncell_m.group(2))) if ncell_m else (1, 1)
    if not dim_m:
        sys.exit(f"ERROR: no 'dim' in {cfg_path}")
    dim   = (int(dim_m.group(1)),   int(dim_m.group(2)))

    size_uc_m = re.search(
        r'size_uc\s*=\s*\{[^}]*?x\s*=\s*([\d.]+)[^}]*?y\s*=\s*([\d.]+)',
        text, re.DOTALL)
    size_uc = (float(size_uc_m.group(1)), float(size_uc_m.group(2))) if size_uc_m else None

    layers = []
    layer_pattern = re.compile(
        r'\b([a-z])\s*=\s*\{[^}]*?'
        r'size_uc\s*=\s*\{[^}]*?z\s*=\s*([\d.]+)[^}]*?\}[^}]*?'
        r'filename\s*=\s*"([^"]+)"',
        re.DOTALL)

    for m in layer_pattern.finditer(text):
        letter   = m.group(1)
        cz       = float(m.group(2))
        filename = m.group(3)

        block_start = m.start()
        depth = 0
        block_end = block_start
        for i, ch in enumerate(text[block_start:], block_start):
            if ch == '{': depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    block_end = i
                    break

        block = text[block_start:block_end]
        # crystruct names may carry a site label ("C 1", "Si1"): keep the element symbol
        atom_names = []
        for name in re.findall(r'\bname\s*=\s*"([^"]+)"', block):   # not 'filename'
            sym = re.match(r'\s*([A-Z][a-z]?)', name)
            atom_names.append(sym.group(1) if sym else name.strip())

        unique = []
        seen = set()
        for name in atom_names:
            if name not in seen:
                unique.append(name)
                seen.add(name)

        layers.append({
            'letter': letter, 'atoms': '+'.join(unique),
            'cz': cz, 'filename': filename,
        })

    return {'ncell': ncell, 'dim': dim, 'size_uc': size_uc, 'layers': layers}


def find_cfg(search_dir, direction=None):
    """Looks for a .cfg file in search_dir, then in the parent, then in the default crystruct path."""
    search_dir = Path(search_dir)
    cfgs = list(search_dir.glob("*.cfg"))
    if cfgs:
        return cfgs[0]

    if direction:
        for pattern in [f"*ixchel*{direction}*.cfg", f"*{direction}*.cfg", "*.cfg"]:
            cfgs = list(search_dir.parent.glob(pattern))
            if cfgs:
                return cfgs[0]

    for base in [Path.home() / "Downloads" / "crystruct-master" / "unittest" / "src"]:
        patterns = ([f"*ixchel*{direction}*.cfg", f"*{direction}*.cfg"] if direction else []) + ["*.cfg"]
        for pattern in patterns:
            cfgs = list(base.glob(pattern))
            if cfgs:
                return cfgs[0]

    return None


# =============================================================================
# Comparison
# =============================================================================

def compare_direction(direction):
    """Reads WIEN2k and ixchel TIFFs; both are cropped to one UC."""
    wien_dir   = Path(f"wien2k_{direction}")
    ixchel_dir = Path(f"ixchel_{direction}")

    # Config from ixchel folder (or parent)
    cfg_path = find_cfg(ixchel_dir, direction)
    if cfg_path:
        print(f"  Config: {cfg_path}")
        cfg = parse_cfg(cfg_path)
    else:
        print(f"  ERROR: no .cfg found for [{direction}] (dim and NCELL are needed)")
        return None

    ncx, ncy   = cfg['ncell']
    dimx, dimy = cfg['dim']
    # dim/NCELL need not be an integer (graphene 512/13 = 39.38): both sides are
    # sampled on round(dim/NCELL) pixels spanning exactly one unit cell
    uc_px_x    = int(round(dimx / ncx))
    uc_px_y    = int(round(dimy / ncy))
    uc_shape   = (uc_px_y, uc_px_x)   # (rows, cols) = (y, x)

    size_uc = cfg.get('size_uc')
    if size_uc:
        print(f"  NCELL=[{ncx},{ncy}], dim=[{dimx},{dimy}], "
              f"UC: {size_uc[0]:.3f} x {size_uc[1]:.3f} Å, "
              f"ixchel UC px: {uc_px_x}×{uc_px_y}")
    else:
        print(f"  NCELL=[{ncx},{ncy}], dim=[{dimx},{dimy}], ixchel UC px: {uc_px_x}×{uc_px_y}")

    # ── Read WIEN2k TIFFs ─────────────────────────────────────────────────────
    wien_files = sorted(wien_dir.glob("atompot*.tif"))
    if not wien_files:
        print(f"  MISSING: {wien_dir}/atompot*.tif")
        return None

    # Read the WIEN2k UC size from the TIFFs themselves (may differ from ixchel)
    sample = read_ixchel_tiff(wien_files[0])
    w_ny, w_nx = sample.shape

    # A TIFF with the ixchel dim shape covers the NCELL supercell (tiff2ixchel.py
    # output); any other shape is ONE unit cell (vcoul2tiff*.py, xsf2tiff*.py).
    supercell = (w_ny, w_nx) == (dimy, dimx)
    if supercell:
        w_desc = f"supercell {ncx}×{ncy} UC"
    else:
        w_desc = f"WIEN2k UC px: {w_nx}×{w_ny}"

    print(f"  WIEN2k: {len(wien_files)} layers, TIFF shape={sample.shape}, {w_desc}")

    wien_layers = []
    for f in wien_files:
        raw = read_ixchel_tiff(f).astype(np.float64)
        if not supercell:
            # one UC (periodic grid) -> ixchel supercell grid (exact Fourier interpolation)
            raw, _ = tile_fourier(raw, (ncx, ncy), (dimx, dimy))
        wien_layers.append(extract_unit_cell(raw, (ncx, ncy), (uc_px_x, uc_px_y)))

    # ── Read ixchel TIFFs ────────────────────────────────────────────────────
    ixchel_files = sorted(ixchel_dir.glob("atompot*.tif"))
    if not ixchel_files:
        print(f"  MISSING: {ixchel_dir}/atompot*.tif")
        return None

    print(f"  ixchel: {len(ixchel_files)} layers, UC extraction: [{uc_px_y},{uc_px_x}]")

    ixchel_layers = []
    for f in ixchel_files:
        raw = read_ixchel_tiff(f)
        if raw.shape != (dimy, dimx):
            print(f"  WARNING: {f} has shape {raw.shape}, cfg dim is [{dimy},{dimx}]")
        ixchel_layers.append(extract_unit_cell(raw, (ncx, ncy), (uc_px_x, uc_px_y)))

    n_wien  = len(wien_layers)
    n_ix    = len(ixchel_layers)
    n_compare = min(n_wien, n_ix)
    if n_wien != n_ix:
        print(f"  WARNING: {n_wien} WIEN2k vs {n_ix} ixchel layers, comparing {n_compare}")

    layer_info = cfg.get('layers', [])

    results = []
    for i in range(n_compare):
        w  = wien_layers[i]
        ix = ixchel_layers[i]
        diff = w - ix

        r, _    = pearsonr(w.ravel(), ix.ravel())
        ratio   = w.max() / ix.max() if ix.max() > 0 else float('inf')
        mean_offset = diff.mean()

        if i < len(layer_info):
            atoms  = layer_info[i]['atoms']
            letter = layer_info[i]['letter']
            cz     = layer_info[i]['cz']
        else:
            atoms  = '?'
            letter = string.ascii_lowercase[i] if i < 26 else '?'
            cz     = 0

        results.append({
            'idx': i, 'letter': letter,
            'r': r, 'ratio': ratio, 'atoms': atoms, 'cz': cz,
            'w': w, 'ix': ix,
            'w_range':  (w.min(),  w.max()),
            'ix_range': (ix.min(), ix.max()),
            'mean_offset': mean_offset,
            'size_uc':   size_uc,
            'uc_shape':  uc_shape,
            'direction': direction,
        })

        print(f"  L{i}({letter}) {atoms:>8}: r={r:.4f}, ratio={ratio:.2f}, "
              f"<DFT-IAM>={mean_offset:+.3f} 1/Å, "
              f"WIEN2k=[{w.min():.2f},{w.max():.2f}], "
              f"ixchel=[{ix.min():.2f},{ix.max():.2f}]")

    return results


# =============================================================================
# Plotting
# =============================================================================

POT_UNIT = r"$V_z$ [1/Å]"


def make_extent(shape, size_uc):
    if size_uc is None:
        return None
    # pixel centres at x_i = i*a/n (as sampled by extract_unit_cell)
    hx, hy = 0.5 * size_uc[0] / shape[1], 0.5 * size_uc[1] / shape[0]
    return [-hx, size_uc[0] - hx, -hy, size_uc[1] - hy]


def plot_results(all_results, prefix):
    n_rows = sum(min(len(r), 2) for r in all_results.values())
    fig = plt.figure(figsize=(22, 5.5 * n_rows))
    row = 0

    for direction, results in all_results.items():
        u_label, v_label = get_inplane_directions(direction)
        x_ax = u_label + r"  [Å]"
        y_ax = v_label + r"  [Å]"

        # Max 2 layers per direction: one with a heavy atom, one with a light one
        if len(results) <= 2:
            plot_idx = list(range(len(results)))
        else:
            heavy = [i for i, r in enumerate(results)
                     if any(x in r['atoms'] for x in ('Ti', 'Fe', 'Sr', 'Si', 'C'))]
            light = [i for i, r in enumerate(results)
                     if r['atoms'] in ('O', 'F')]
            if not heavy:
                heavy = list(range(len(results)))
            if not light:
                light = [i for i in range(len(results)) if i not in heavy]
            best_h = max(heavy, key=lambda i: results[i]['r']) if heavy else 0
            best_l = max(light,  key=lambda i: results[i]['r']) if light else (
                1 if len(results) > 1 else 0)
            plot_idx = sorted(set([best_h, best_l]))

        for idx in plot_idx:
            res = results[idx]
            w, ix = res['w'], res['ix']
            diff  = w - ix
            size_uc = res.get('size_uc')
            extent  = make_extent(w.shape, size_uc)
            ny, nx  = w.shape
            has_phys = (size_uc is not None)

            peak_row = np.argmax(np.max(w, axis=1))

            if has_phys:
                ax_A, ay_A = size_uc
                x_line = np.arange(nx) * ax_A / nx
                y_cut  = peak_row * ay_A / ny
                cut_str = f"Cut at {v_label} = {y_cut:.2f} Å"
            else:
                x_line  = np.arange(nx)
                y_cut   = peak_row
                cut_str = f"Cut at row = {peak_row}"

            # ==========  Column 1: line profile  ==========
            ax1 = fig.add_subplot(n_rows, 4, row * 4 + 1)
            ax1.plot(x_line, w[peak_row,  :], 'b-',  label='WIEN2k (DFT)', lw=1.5)
            ax1.plot(x_line, ix[peak_row, :], 'r--', label='ixchel (IAM)', lw=1.5)
            ax1.set_title(
                f"[{direction}] Layer {res['idx']} ({res['letter']}): "
                f"{res['atoms']}\n"
                f"$r$ = {res['r']:.3f},  ratio = {res['ratio']:.2f},  "
                f"$c_z$ = {res['cz']:.3f} Å\n"
                f"{cut_str},  "
                r"$\langle\Delta V_z\rangle$"
                f" = {res['mean_offset']:+.3f} 1/Å",
                fontsize=8)
            ax1.set_xlabel(x_ax if has_phys else "pixel", fontsize=9)
            ax1.set_ylabel(POT_UNIT, fontsize=9)
            ax1.legend(fontsize=7, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=7)
            if w.min() < 0:
                ax1.axhline(0, color='gray', lw=0.5, ls=':')

            # ==========  Column 2: WIEN2k 2D  ==========
            vmax = max(w.max(), ix.max())
            vmin = min(0, w.min(), ix.min())

            ax2 = fig.add_subplot(n_rows, 4, row * 4 + 2)
            im2 = ax2.imshow(w, origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                             extent=extent,
                             aspect='equal' if has_phys else 'auto')
            ax2.set_title(f"WIEN2k (DFT+Poisson)\n"
                          f"[{w.min():.2f}, {w.max():.2f}] 1/Å", fontsize=8)
            if has_phys:
                ax2.set_xlabel(x_ax, fontsize=8)
                ax2.set_ylabel(y_ax, fontsize=8)
                ax2.axhline(y_cut, color='cyan', lw=0.8, ls='--', alpha=0.7)
            ax2.tick_params(labelsize=7)
            cb = plt.colorbar(im2, ax=ax2, shrink=0.75, pad=0.02)
            cb.set_label(POT_UNIT, fontsize=7)
            cb.ax.tick_params(labelsize=6)

            # ==========  Column 3: ixchel 2D  ==========
            ax3 = fig.add_subplot(n_rows, 4, row * 4 + 3)
            im3 = ax3.imshow(ix, origin='lower', cmap='hot', vmin=vmin, vmax=vmax,
                             extent=extent,
                             aspect='equal' if has_phys else 'auto')
            ax3.set_title(f"ixchel (Kirkland IAM)\n"
                          f"[{ix.min():.2f}, {ix.max():.2f}] 1/Å", fontsize=8)
            if has_phys:
                ax3.set_xlabel(x_ax, fontsize=8)
                ax3.set_ylabel(y_ax, fontsize=8)
                ax3.axhline(y_cut, color='cyan', lw=0.8, ls='--', alpha=0.7)
            ax3.tick_params(labelsize=7)
            cb = plt.colorbar(im3, ax=ax3, shrink=0.75, pad=0.02)
            cb.set_label(POT_UNIT, fontsize=7)
            cb.ax.tick_params(labelsize=6)

            # ==========  Column 4: difference  ==========
            dmax = max(abs(diff.min()), abs(diff.max()))

            ax4 = fig.add_subplot(n_rows, 4, row * 4 + 4)
            im4 = ax4.imshow(diff, origin='lower', cmap='RdBu_r',
                             vmin=-dmax, vmax=dmax,
                             extent=extent,
                             aspect='equal' if has_phys else 'auto')
            ax4.set_title(
                r"$V_z^{\mathrm{DFT}} - V_z^{\mathrm{IAM}}$ [1/Å]" "\n"
                f"mean = {diff.mean():+.3f}, "
                f"[{diff.min():.2f}, {diff.max():.2f}]",
                fontsize=8)
            if has_phys:
                ax4.set_xlabel(x_ax, fontsize=8)
                ax4.set_ylabel(y_ax, fontsize=8)
            ax4.tick_params(labelsize=7)
            cb = plt.colorbar(im4, ax=ax4, shrink=0.75, pad=0.02)
            cb.set_label(r"$\Delta V_z$ [1/Å]", fontsize=7)
            cb.ax.tick_params(labelsize=6)

            row += 1

    fig.suptitle(
        f"{prefix} — Proj. electrostatic potential "
        r"$V_z(x,y)$ [1/Å]" "\n"
        "WIEN2k (DFT + Kirkland IAM with subslicing) vs ixchel (IAM) "
        "— compared at ixchel UC resolution",
        fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = f"{prefix}_subslicing_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nPlot: {out}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='WIEN2k vs ixchel — proj. potential V_z [1/Å]')
    parser.add_argument('--prefix', '-p', default='Rutile',
                        help='Only used for the output file name (e.g. Rutile, Si).')
    parser.add_argument('--directions', '-d', nargs='+', default=None,
                        help='Projection directions (e.g. 001 100 110).')
    args = parser.parse_args()

    prefix = args.prefix

    if args.directions:
        directions = args.directions
    else:
        # Auto-detection: all wien2k_* folders with matching ixchel_* folders
        directions = []
        for p in sorted(Path('.').glob('wien2k_*')):
            d = p.name.replace('wien2k_', '')
            if Path(f'ixchel_{d}').exists():
                directions.append(d)

    if not directions:
        print("No wien2k_*/ixchel_* folder pairs found. Use --directions.")
        sys.exit(1)

    print(f"Comparison: {prefix}, directions: {directions}")
    print(f"WIEN2k is downsampled to ixchel UC resolution (if necessary)")
    print("=" * 70)

    all_results = {}
    for d in directions:
        print(f"\n[{d}]:")
        u_lbl, v_lbl = get_inplane_directions(d)
        print(f"  Projection || [{d}],  plane: x || {u_lbl}, y || {v_lbl}")
        r = compare_direction(d)
        if r:
            all_results[d] = r

    if not all_results:
        print("\nNo data found.")
        sys.exit(1)

    plot_results(all_results, prefix)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'[hkl]':<8} {'L':<4} {'Atoms':<10} {'cz [Å]':<8} "
          f"{'r':<8} {'ratio':<8} {'DFT max':<12} {'IAM max':<12} {'<DFT-IAM>':<10}")
    print("-" * 90)
    for d, results in all_results.items():
        for res in results:
            print(f"[{d}]   {res['idx']}    {res['atoms']:<10} "
                  f"{res['cz']:<8.3f} {res['r']:<8.4f} {res['ratio']:<8.2f} "
                  f"{res['w_range'][1]:<12.3f} {res['ix_range'][1]:<12.3f} "
                  f"{res['mean_offset']:<+10.3f}")
