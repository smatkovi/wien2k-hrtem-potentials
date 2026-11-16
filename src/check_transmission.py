#!/usr/bin/env python3
"""
check_transmission.py — verify that ixchel uses a layer potential as intended

Run `ixchel main` once with `writeTrans = 1;` in simulation.cfg; ixchel then
writes trans_<label>.tif, the transmission function t = exp(i σ V_bl) of each
layer, where V_bl is the layer potential after ixchel's bandwidth limit
(|k| < 2/3 of the Nyquist frequency of the supercell grid) and
σ = λ (1 + E/511 keV) (src/ixchelfun.cpp).

This script recomputes σ V_bl from the potential TIFF and compares its phase
with the one ixchel wrote.  Agreement to ~1e-6 rad means ixchel read the file
with the intended size, orientation, origin and unit.  Optionally a second
pair (e.g. the IAM run) gives the phase difference DFT − IAM, i.e. the part
of the object phase that is due to bonding.

Usage:
    python3 check_transmission.py --crystal crystal_struct.cfg --sim simulation.cfg \\
        --pot atompota.tif --trans trans_a.tif \\
        [--ref-pot ../iam/atompota.tif --ref-trans ../iam/trans_a.tif]

Layout of this file: the CORE comes first, the two formulas behind
t = exp(i σ V_bl): wavelength_and_sigma gives σ, band_limit gives V_bl.
After the core: the file formats (complex TIFF, cfg values) and the command
line, which reads the files and prints the comparison.

Dependencies: numpy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ixchel_tiff import read_ixchel_tiff        # noqa: E402


# ═══════════════════════════════════════════════════════════════
# CORE: what ixchel makes of a layer potential V
#
# The transmission function of a layer is t = exp(i σ V_bl).  σ follows from
# the high tension (wavelength_and_sigma); V_bl is V with the high spatial
# frequencies removed (band_limit).  Both follow the ixchel source.
# ═══════════════════════════════════════════════════════════════

def wavelength_and_sigma(v0):
    """Electron wavelength [Å] and interaction scale σ [Å] for the high tension v0 [kV], as in ixchel
    (utils.h / ixchelfun.cpp).  The phase of the transmission function is σ * V with V in 1/Å."""
    m0c2, hc = 510.99895, 12.398419843320026         # keV, keV·Å
    lam = hc / np.sqrt(v0 * (2.0 * m0c2 + v0))
    sigma = lam * (1.0 + v0 / 511.0)
    return lam, sigma


def band_limit(v, nx, ny, ax, by):
    """The potential v [ny, nx] after ixchel's bandwidth limit: only |k| below 2/3 of the Nyquist
    frequency of the supercell grid (nx x ny pixels on ax x by Å) is kept."""
    # ixchel's mask bit for bit: float32 arithmetic, BW = 2.0F/3.0F, keep k² < k2max
    # (ixchelfun.cpp incwf; slicelib.cpp freqn: index i > n/2 is negative)
    f32 = np.float32

    def k2(n, a):
        """k² [1/Å²] for the n pixels along an edge of length a."""
        i = np.arange(n)
        k = np.where(i > n // 2, i - n, i).astype(f32) / f32(a)
        return k * k

    kmax = f32(2.0) / f32(3.0) * min(f32(nx) / (f32(2.0) * ax), f32(ny) / (f32(2.0) * by))
    keep = (k2(ny, by)[:, None] + k2(nx, ax)[None, :]) < f32(kmax) * f32(kmax)
    F = np.fft.fft2(v)                               # [ny, nx]
    F[~keep] = 0
    return np.fft.ifft2(F).real


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: file formats (complex TIFF, cfg values), command line
# ═══════════════════════════════════════════════════════════════

# ── file formats ───────────────────────────────────────────────

def read_complex(path):
    """ixchel complex TIFF: each row holds nx real parts followed by nx
    imaginary parts (src/TiffFile.cpp)."""
    a = read_ixchel_tiff(path)
    nx = a.shape[1] // 2
    return a[:, :nx] + 1j * a[:, nx:]


def cfg_value(text, pattern, what, path):
    """The match of `pattern` in a cfg text (comments removed); stops with a message if there is none."""
    m = re.search(pattern, re.sub(r'#[^\n]*', '', text))
    if not m:
        sys.exit(f"Error: {what} not found in {path}")
    return m


# ── command line: reads the files, prints the comparison ───────

def main():
    ap = argparse.ArgumentParser(description='Compare ixchel trans_*.tif with exp(i sigma V)')
    ap.add_argument('--crystal', required=True, help='crystal-structure cfg (size_uc; NCELL, default 1 1)')
    ap.add_argument('--sim', required=True, help='simulation cfg (dim, v0)')
    ap.add_argument('--pot', required=True, help='layer potential TIFF given to ixchel [1/Å]')
    ap.add_argument('--trans', required=True, help='trans_<label>.tif written by ixchel main')
    ap.add_argument('--ref-pot', help='reference layer potential (e.g. IAM run)')
    ap.add_argument('--ref-trans', help='reference trans_<label>.tif')
    args = ap.parse_args()

    # 1. the numbers from the two cfg files, e.g.
    #        crystal cfg:     NCELL = [ 14, 8, 1];     size_uc = { x = 2.4609990418; y = 4.2625753778; ... };
    #        simulation cfg:  dim = [ 1008, 1008];     v0 = 80.;
    cry = Path(args.crystal).read_text()
    sim = Path(args.sim).read_text()
    ncell = re.search(r'\bNCELL\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', re.sub(r'#[^\n]*', '', cry))
    ncx, ncy = (int(ncell.group(1)), int(ncell.group(2))) if ncell else (1, 1)   # ixchel default
    size = cfg_value(cry, r'size_uc\s*=\s*\{[^}]*?x\s*=\s*([0-9.eE+\-]+)[^}]*?y\s*=\s*([0-9.eE+\-]+)',
                     'size_uc', args.crystal)
    dim = cfg_value(sim, r'\bdim\s*=\s*\[\s*(\d+)\s*,\s*(\d+)', 'dim', args.sim)
    v0 = float(cfg_value(sim, r'\bv0\s*=\s*([0-9.eE+\-]+)', 'v0', args.sim).group(1))     # high tension [kV]

    # supercell size [Å] in float32, as ixchel computes it (ixchelfun.cpp, incwf)
    ax = np.float32(ncx) * np.float32(float(size.group(1)))
    by = np.float32(ncy) * np.float32(float(size.group(2)))
    nx, ny = int(dim.group(1)), int(dim.group(2))

    # 2. electron wavelength and interaction scale
    lam, sigma = wavelength_and_sigma(v0)

    # 3. the comparison of one potential file with one transmission file
    def check(pot_path, trans_path, label):
        """Print how well the phase of ixchel's transmission function agrees with sigma * V."""
        V = read_ixchel_tiff(pot_path)
        t = read_complex(trans_path)
        if V.shape != (ny, nx) or t.shape != (ny, nx):
            sys.exit(f"Error: {label}: shapes {V.shape} / {t.shape} differ from dim {nx} x {ny}")
        pred = sigma * band_limit(V, nx, ny, ax, by)            # the phase ixchel should have written
        err = np.abs(np.angle(t * np.exp(-1j * pred)))          # what is left after removing it, within -pi ... pi
        print(f"{label}: |t| in [{np.abs(t).min():.6f}, {np.abs(t).max():.6f}], "
              f"phase in [{np.angle(t).min():.4f}, {np.angle(t).max():.4f}] rad, "
              f"max |phase - sigma*V| = {err.max():.2e} rad")
        return t, V

    # 4. the check itself, and with a reference pair the phase difference between the two runs
    print(f"v0 = {v0:g} kV, lambda = {lam:.5f} Å, sigma = {sigma:.5f} Å (phase = sigma * V[1/Å]), "
          f"supercell {ax:.3f} x {by:.3f} Å on {nx} x {ny} px")
    t, V = check(args.pot, args.trans, 'layer')
    if args.ref_pot and args.ref_trans:
        t0, V0 = check(args.ref_pot, args.ref_trans, 'reference')
        measured = np.angle(t * np.conj(t0))
        predicted = sigma * band_limit(V - V0, nx, ny, ax, by)
        print(f"phase difference layer - reference: [{measured.min():.4f}, {measured.max():.4f}] rad, "
              f"predicted [{predicted.min():.4f}, {predicted.max():.4f}] rad, "
              f"max deviation {np.abs(measured - predicted).max():.2e} rad")


if __name__ == '__main__':
    main()
