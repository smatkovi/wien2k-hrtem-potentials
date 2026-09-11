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

Dependencies: numpy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import tifffile


def read_page1(path):
    with tifffile.TiffFile(str(path)) as tif:
        return tif.pages[1].asarray().astype(np.float64)


def read_complex(path):
    """ixchel complex TIFF: each row holds nx real parts followed by nx
    imaginary parts (src/TiffFile.cpp)."""
    a = read_page1(path)
    nx = a.shape[1] // 2
    return a[:, :nx] + 1j * a[:, nx:]


def cfg_value(text, pattern, what, path):
    m = re.search(pattern, re.sub(r'#[^\n]*', '', text))
    if not m:
        sys.exit(f"Error: {what} not found in {path}")
    return m


def main():
    ap = argparse.ArgumentParser(description='Compare ixchel trans_*.tif with exp(i sigma V)')
    ap.add_argument('--crystal', required=True, help='crystal-structure cfg (size_uc; NCELL, default 1 1)')
    ap.add_argument('--sim', required=True, help='simulation cfg (dim, v0)')
    ap.add_argument('--pot', required=True, help='layer potential TIFF given to ixchel [1/Å]')
    ap.add_argument('--trans', required=True, help='trans_<label>.tif written by ixchel main')
    ap.add_argument('--ref-pot', help='reference layer potential (e.g. IAM run)')
    ap.add_argument('--ref-trans', help='reference trans_<label>.tif')
    args = ap.parse_args()

    cry = Path(args.crystal).read_text()
    sim = Path(args.sim).read_text()
    nc = re.search(r'\bNCELL\s*[=:]\s*[\[(]\s*(\d+)\s*,\s*(\d+)', re.sub(r'#[^\n]*', '', cry))
    ncx, ncy = (int(nc.group(1)), int(nc.group(2))) if nc else (1, 1)   # ixchel default
    su = cfg_value(cry, r'size_uc\s*=\s*\{[^}]*?x\s*=\s*([0-9.eE+\-]+)[^}]*?y\s*=\s*([0-9.eE+\-]+)',
                   'size_uc', args.crystal)
    dm = cfg_value(sim, r'\bdim\s*=\s*\[\s*(\d+)\s*,\s*(\d+)', 'dim', args.sim)
    v0 = float(cfg_value(sim, r'\bv0\s*=\s*([0-9.eE+\-]+)', 'v0', args.sim).group(1))

    # supercell size [Å] in float32, as ixchel computes it (ixchelfun.cpp, incwf)
    ax = np.float32(ncx) * np.float32(float(su.group(1)))
    by = np.float32(ncy) * np.float32(float(su.group(2)))
    nx, ny = int(dm.group(1)), int(dm.group(2))

    # electron wavelength [Å] and interaction scale, as in ixchel utils.h / ixchelfun.cpp
    m0c2, hc = 510.99895, 12.398419843320026         # keV, keV·Å
    lam = hc / np.sqrt(v0 * (2.0 * m0c2 + v0))
    sigma = lam * (1.0 + v0 / 511.0)

    def band_limit(v):
        # ixchel's mask bit for bit: float32 arithmetic, BW = 2.0F/3.0F, keep k² < k2max
        # (ixchelfun.cpp incwf; slicelib.cpp freqn: index i > n/2 is negative)
        f32 = np.float32

        def k2(n, a):
            i = np.arange(n)
            k = np.where(i > n // 2, i - n, i).astype(f32) / f32(a)
            return k * k

        kmax = f32(2.0) / f32(3.0) * min(f32(nx) / (f32(2.0) * ax), f32(ny) / (f32(2.0) * by))
        keep = (k2(ny, by)[:, None] + k2(nx, ax)[None, :]) < f32(kmax) * f32(kmax)
        F = np.fft.fft2(v)                               # [ny, nx]
        F[~keep] = 0
        return np.fft.ifft2(F).real

    def check(pot_path, trans_path, label):
        V = read_page1(pot_path)
        t = read_complex(trans_path)
        if V.shape != (ny, nx) or t.shape != (ny, nx):
            sys.exit(f"Error: {label}: shapes {V.shape} / {t.shape} differ from dim {nx} x {ny}")
        pred = sigma * band_limit(V)
        err = np.abs(np.angle(t * np.exp(-1j * pred)))
        print(f"{label}: |t| in [{np.abs(t).min():.6f}, {np.abs(t).max():.6f}], "
              f"phase in [{np.angle(t).min():.4f}, {np.angle(t).max():.4f}] rad, "
              f"max |phase - sigma*V| = {err.max():.2e} rad")
        return t, V

    print(f"v0 = {v0:g} kV, lambda = {lam:.5f} Å, sigma = {sigma:.5f} Å (phase = sigma * V[1/Å]), "
          f"supercell {ax:.3f} x {by:.3f} Å on {nx} x {ny} px")
    t, V = check(args.pot, args.trans, 'layer')
    if args.ref_pot and args.ref_trans:
        t0, V0 = check(args.ref_pot, args.ref_trans, 'reference')
        d = np.angle(t * np.conj(t0))
        p = sigma * band_limit(V - V0)
        print(f"phase difference layer - reference: [{d.min():.4f}, {d.max():.4f}] rad, "
              f"predicted [{p.min():.4f}, {p.max():.4f}] rad, max deviation {np.abs(d - p).max():.2e} rad")


if __name__ == '__main__':
    main()
