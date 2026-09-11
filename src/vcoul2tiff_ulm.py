#!/usr/bin/env python3
"""
vcoul2tiff_ulm4.py — Ulm-Ansatz (Kurasch et al. 2011), gepatcht

Aenderungen ggü. _ulm3:
  (1) read_xsf erkennt die XSF-Achsenreihenfolge automatisch (Fortran vs C),
      statt blind reshape((nx,ny,nz)) [C-Order] zu nehmen. Standard-XSF
      (WIEN2k/XCrySDen) ist x-schnellster = Fortran-Order. Falscher reshape
      verwürfelt die Voxel -> Kern-Cusp landet im Interstitial.
  (2) XSF general grid: n Punkte spannen 0..L INKLUSIVE (letzte Ebene =
      periodisches Abbild der ersten). Voxelgröße ist span/(n-1), und die
      Projektion lässt die Duplikatebene weg (sonst Doppelzählung + ~1/n
      Skalenfehler).

Methode (Kurasch et al. Beilstein J. Nanotechnol. 2011, 2, 394):
  1. 3D-Potential aus XSF lesen [Ry, Span-Vektoren in Å]
  2. Nearest-Neighbor-Cutoff: V < -cutoff_Ry -> max(Nachbarn im 3x3x3).
  3. Projektion pro Layer entlang der Strahlrichtung (Miller-Index).
  4. Vorzeichenumkehr: V_s = -V_WIEN2k  (Streupotential > 0 bei Kernen)
  5. Einheiten: V_Ry x dz_A x (V/Ry) / H2_2PI_ME -> 1/A

Verwendung:
    python3 vcoul2tiff_ulm4.py \\
        --xsf Graphene_vcoul_DFT.xsf \\
        --cfg Graphene_ixchel_001.cfg \\
        --miller 0 0 1 \\
        [--order auto|F|C] \\
        [--cutoff 50000] \\
        [-o wien2k_001]

Abhaengigkeiten: numpy, scipy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
import tifffile


# ═══════════════════════════════════════════════════════════════
# Konstanten
# ═══════════════════════════════════════════════════════════════

BOHR      = 0.529177210903   # Bohr -> Angstrom
RY_TO_V   = 13.605693122994  # Ry -> V
H2_2PI_ME = 47.88            # hbar^2/(2pi*m_e*e) [V*A^2]  (Kirkland/ixchel)


# ═══════════════════════════════════════════════════════════════
# XSF-Leser  (mit Achsenreihenfolge-Auto-Erkennung)
# ═══════════════════════════════════════════════════════════════

def _roughness(d):
    """Mittlere absolute Nachbardifferenz über alle 3 Achsen.
    Korrekt geordnete Potentialdaten sind glatt (außer am Cusp) -> klein.
    Verwürfelte Daten sind quasi-Rauschen -> groß."""
    return sum(np.mean(np.abs(np.diff(d, axis=ax))) for ax in range(3))


def _reshape_xsf(vals, nx, ny, nz, order):
    """order: 'F' (Standard-XSF, x schnellster), 'C' (z schnellster),
       'auto' (per Glattheit entscheiden)."""
    arr = np.asarray(vals[:nx * ny * nz], dtype=np.float64)
    if order in ('F', 'C'):
        return arr.reshape((nx, ny, nz), order=order), order
    # auto
    dF = arr.reshape((nx, ny, nz), order='F')
    dC = arr.reshape((nx, ny, nz), order='C')
    rF, rC = _roughness(dF), _roughness(dC)
    if rF <= rC:
        print(f"    Achsen-Auto: order='F' (Standard-XSF)  "
              f"[roughness F={rF:.3f} < C={rC:.3f}]")
        return dF, 'F'
    else:
        print(f"    Achsen-Auto: order='C' (nicht-standard)  "
              f"[roughness C={rC:.3f} < F={rF:.3f}]")
        return dC, 'C'


def read_xsf(path, order='auto'):
    """
    Liest 3D-Daten aus XSF (von WIEN2k). Span-Vektoren in Angstrom,
    Werte in Ry.

    Rueckgabe: data [nx,ny,nz] (Ry), (ax,by,cz) [A], (nx,ny,nz)
    """
    with open(path) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        ln = lines[i]
        if 'DATAGRID_3D' in ln and 'BEGIN_BLOCK' not in ln and 'END' not in ln:
            break
        i += 1
    else:
        sys.exit("Fehler: Kein DATAGRID_3D-Block in XSF gefunden.")
    i += 1

    nx, ny, nz = map(int, lines[i].split());  i += 1
    i += 1  # origin (ignoriert)
    va = np.array(list(map(float, lines[i].split())));  i += 1
    vb = np.array(list(map(float, lines[i].split())));  i += 1
    vc = np.array(list(map(float, lines[i].split())));  i += 1

    vals = []
    while i < len(lines) and len(vals) < nx * ny * nz:
        if 'END_DATAGRID' in lines[i]:
            break
        vals.extend(float(x) for x in lines[i].split())
        i += 1

    if len(vals) < nx * ny * nz:
        sys.exit(f"Fehler: Zu wenig Datenpunkte ({len(vals)} < {nx*ny*nz}).")

    data, used = _reshape_xsf(vals, nx, ny, nz, order)

    ax_ang = np.linalg.norm(va)
    by_ang = np.linalg.norm(vb)
    cz_ang = np.linalg.norm(vc)

    return data, (ax_ang, by_ang, cz_ang), (nx, ny, nz)


# ═══════════════════════════════════════════════════════════════
# cfg-Parser (ixchel crystruct.cfg)  — unverändert
# ═══════════════════════════════════════════════════════════════

def parse_cfg(cfg_path):
    text = Path(cfg_path).read_text()

    sm = re.search(r'stack_seq\s*=\s*\"\d+\(([^)]+)\)\"', text)
    if not sm:
        sys.exit("Fehler: stack_seq nicht in cfg gefunden.")
    seq = list(sm.group(1))

    lm = re.search(r'layers\s*=\s*\{', text)
    if not lm:
        sys.exit("Fehler: layers-Block nicht in cfg gefunden.")
    depth = 0; j = lm.end() - 1
    while j < len(text):
        if text[j] == '{':   depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0: break
        j += 1
    layers_text = text[lm.end():j]

    layer_defs = {}
    pos = 0
    while pos < len(layers_text):
        lbm = re.search(r'\b([a-z])\s*=\s*\{', layers_text[pos:])
        if not lbm: break
        label = lbm.group(1)
        bstart = pos + lbm.end() - 1
        d = 0; p2 = bstart
        while p2 < len(layers_text):
            if layers_text[p2] == '{':   d += 1
            elif layers_text[p2] == '}':
                d -= 1
                if d == 0: break
            p2 += 1
        block = layers_text[bstart+1:p2]
        su = re.search(
            r'size_uc\s*=\s*\{[^}]*x\s*=\s*([0-9.E+\-]+)[^}]*'
            r'y\s*=\s*([0-9.E+\-]+)[^}]*z\s*=\s*([0-9.E+\-]+)', block)
        fn_m = re.search(r'filename\s*=\s*\"([^\"]+)\"', block)
        if not su or not fn_m:
            pos = p2 + 1; continue
        layer_defs[label] = {
            'ax': float(su.group(1)),
            'by': float(su.group(2)),
            'cz': float(su.group(3)),
            'filename': fn_m.group(1),
        }
        pos = p2 + 1

    layers = []
    z0 = 0.0
    for label in seq:
        if label not in layer_defs:
            sys.exit(f"Fehler: Layer '{label}' nicht in layer_defs.")
        ld = layer_defs[label]
        layers.append({
            'label': label,
            'ax': ld['ax'], 'by': ld['by'], 'cz': ld['cz'],
            'z0': z0,
            'filename': ld['filename'],
        })
        z0 += ld['cz']

    return {'layers': layers, 'total_z': z0}


# ═══════════════════════════════════════════════════════════════
# Nearest-Neighbor-Cutoff (Kurasch et al. 2011)  — unverändert
# ═══════════════════════════════════════════════════════════════

def apply_nn_cutoff(data_3d, cutoff_ry):
    data = data_3d.copy()
    n_replaced = int((data < -cutoff_ry).sum())
    if n_replaced == 0:
        return data, 0

    size = 3
    for _ in range(20):
        bad = data < -cutoff_ry
        if not bad.any():
            break
        tmp = np.where(bad, -np.inf, data)
        nb_max = maximum_filter(tmp, size=size, mode='wrap')
        no_valid = bad & (nb_max == -np.inf)
        if no_valid.any():
            size += 2
            continue
        data = np.where(bad, nb_max, data)
        size = 3

    remaining = int((data < -cutoff_ry).sum())
    if remaining > 0:
        print(f"    WARNUNG: {remaining} Voxel nach Cutoff immer noch < -Schwellwert")
    return data, n_replaced


# ═══════════════════════════════════════════════════════════════
# Projektion entlang Miller-Richtung  (Endpunkt-/dz-Konvention korrigiert)
# ═══════════════════════════════════════════════════════════════

def project_layer(data_cut, cell_ang, shape, miller, z0_ang, cz_lay):
    h, k, l = miller
    ax_ang, by_ang, cz_ang = cell_ang
    nx, ny, nz = shape

    nonzero = [x for x in [h, k, l] if x != 0]
    if len(nonzero) != 1:
        sys.exit(
            f"Fehler: Miller [{h}{k}{l}] nicht unterstuetzt.\n"
            f"  Nur [001], [100], [010] moeglich (achsenparallele Projektionen)."
        )

    if l != 0:       # [001]: beam entlang c, sum axis=2
        n_pts = nz;   span = cz_ang;   axis = 2
    elif h != 0:     # [100]: beam entlang a, sum axis=0
        n_pts = nx;   span = ax_ang;   axis = 0
    else:            # [010]: beam entlang b, sum axis=1
        n_pts = ny;   span = by_ang;   axis = 1

    # XSF general grid: n_pts Punkte spannen 0..span INKLUSIVE.
    # Es gibt n_pts-1 eindeutige Intervalle; letzte Ebene == erste.
    n_int  = max(n_pts - 1, 1)
    d_proj = span / n_int                       # korrekte Voxelgröße [A]

    i_lo = int(round(z0_ang            / span * n_int))
    i_hi = int(round((z0_ang + cz_lay) / span * n_int))
    i_lo = max(0, min(i_lo, n_int))
    i_hi = max(0, min(i_hi, n_int))             # Slice [..:i_hi) lässt Index n_pts-1 (Duplikat) weg
    n_slices = i_hi - i_lo

    if n_slices == 0:
        return None, d_proj, 0

    if axis == 2:
        proj = data_cut[:, :, i_lo:i_hi].sum(axis=2)   # [nx, ny]
    elif axis == 0:
        proj = data_cut[i_lo:i_hi, :, :].sum(axis=0)   # [ny, nz]
    else:
        proj = data_cut[:, i_lo:i_hi, :].sum(axis=1)   # [nx, nz]

    proj_yx = proj.T   # ixchel: [nrow=y, ncol=x]
    return proj_yx, d_proj, n_slices


# ═══════════════════════════════════════════════════════════════
# TIFF-Ausgabe (ixchel-Format)  — unverändert
# ═══════════════════════════════════════════════════════════════

def write_ixchel_tiff(filename, data_yx):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    data = data_yx.astype(np.float32)
    vmin, vmax = float(data.min()), float(data.max())
    if vmax == vmin:
        vmax = vmin + 1e-6
    preview = np.clip((data - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
    preview_rgb = np.stack([preview, preview, preview], axis=-1)
    meta = np.zeros((1, 2), dtype=np.uint32)
    with tifffile.TiffWriter(filename) as tfw:
        tfw.write(preview_rgb, photometric='rgb',        contiguous=False)
        tfw.write(data,        photometric='minisblack', contiguous=False)
        tfw.write(meta,        photometric='minisblack', contiguous=False)
    print(f"    -> {filename}  [{vmin:.4g} ... {vmax:.4g}] 1/A")


# ═══════════════════════════════════════════════════════════════
# Hauptprogramm
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Ulm-Ansatz (Kurasch 2011): vcoul XSF -> ixchel TIF')
    ap.add_argument('--xsf',    required=True)
    ap.add_argument('--cfg',    required=True)
    ap.add_argument('--miller', type=int, nargs=3, default=[0, 0, 1],
                    metavar=('H', 'K', 'L'))
    ap.add_argument('--order',  choices=['auto', 'F', 'C'], default='auto',
                    help="XSF-Achsenreihenfolge (Standard: auto)")
    ap.add_argument('--cutoff', type=float, default=50000,
                    help='Cutoff-Schwellwert in Volt (Standard: 50000)')
    ap.add_argument('-o', '--output', default='wien2k_ulm')
    args = ap.parse_args()

    h, k, l   = args.miller
    cutoff_Ry = args.cutoff / RY_TO_V
    out_dir   = args.output

    print(f"\n[1] XSF lesen: {args.xsf}")
    data_3d, (ax_ang, by_ang, cz_ang), (nx, ny, nz) = read_xsf(args.xsf, args.order)
    print(f"    Gitter:        {nx} x {ny} x {nz} Voxel")
    print(f"    Einheitszelle: {ax_ang:.4f} x {by_ang:.4f} x {cz_ang:.4f} A")
    print(f"    V (roh):       [{data_3d.min():.1f}, {data_3d.max():.1f}] Ry"
          f" = [{data_3d.min()*RY_TO_V:.0f}, {data_3d.max()*RY_TO_V:.0f}] V")

    print(f"\n[2] Nearest-Neighbor-Cutoff bei {args.cutoff:.0f} V = {cutoff_Ry:.1f} Ry")
    n_over = int((data_3d < -cutoff_Ry).sum())
    print(f"    Voxel unter -Schwellwert: {n_over} ({100*n_over/data_3d.size:.4f}%)")
    data_cut, n_replaced = apply_nn_cutoff(data_3d, cutoff_Ry)
    print(f"    Ersetzt:  {n_replaced} Voxel")
    print(f"    V (nach Cutoff): [{data_cut.min()*RY_TO_V:.0f}, {data_cut.max()*RY_TO_V:.0f}] V")

    print(f"\n[3] cfg lesen: {args.cfg}")
    cfg    = parse_cfg(args.cfg)
    layers = cfg['layers']
    print(f"    {len(layers)} Layer, Gesamtdicke {cfg['total_z']:.3f} A,  Miller [{h}{k}{l}]")

    print(f"\n[4] Projektion pro Layer ...")
    outs = []
    for ly in layers:
        proj_raw, d_proj, n_slices = project_layer(
            data_cut, (ax_ang, by_ang, cz_ang), (nx, ny, nz),
            (h, k, l), ly['z0'], ly['cz'])
        if proj_raw is None:
            print(f"\n  Layer '{ly['label']}': keine Schichten, uebersprungen")
            continue
        to_kirk  = d_proj * RY_TO_V / H2_2PI_ME
        proj_yx  = -proj_raw * to_kirk   # V_s = -V_C
        print(f"\n  Layer '{ly['label']}': {n_slices} Schichten, d={d_proj*1000:.1f} mA")
        print(f"    V_proj: [{proj_yx.min():.4f}, {proj_yx.max():.4f}] 1/A")
        fname = f"{out_dir}/{ly['filename']}"
        write_ixchel_tiff(fname, proj_yx)
        outs.append(fname)

    print(f"\nFertig -- {len(outs)} Datei(en) in '{out_dir}/'")


if __name__ == '__main__':
    main()
