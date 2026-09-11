#!/usr/bin/env python3
"""
vcoul2tiff_ulm.py — re-implementation of the Kurasch et al. (2011) route

Projects the total WIEN2k Coulomb potential (case.vcoul exported to XSF with
3ddens) onto ixchel-compatible TIFFs following the recipe of

    S. Kurasch, J. C. Meyer, D. Künzel, A. Groß, U. Kaiser,
    "Simulation of bonding effects in HRTEM images of light element materials",
    Beilstein J. Nanotechnol. 2, 394-404 (2011), doi:10.3762/bjnano.2.45

as literally as the paper allows.  The script exists as a comparison for
vcoul2tiff.py; deviations from the paper are bugs, not improvements.

Method (paper p. 397-399):
  1. Read the 3D potential from XSF                 [Ry, span vectors in Å]
  2. Nearest-neighbour cutoff (p. 397): "Whenever a value in the 3D potential
     is higher than some cutoff value, we change this value to the highest
     value in the neighboring pixels."  Cutoff used in the paper: -50 kV (p. 399).
  3. Normalisation (p. 397): "the potential was shifted (i.e., smallest value
     was set to zero)"; p. 399: the 3D potentials were "sliced, normalized and
     projected".
  4. Projection along the beam direction, layer by layer (ixchel cfg).
  5. Units: Ry -> V·Å -> Kirkland units 1/Å (divide by h²/(2π m_e e) = 47.88 V·Å²).

Sign convention:
  WIEN2k stores the Coulomb potential energy of an electron in Ry: V_C < 0 near
  the nuclei, V_C -> small positive constant in vacuum.  The paper plots the
  scattering potential V_s = -V_C (Fig. 2/3, "potential in V" > 0) but quotes
  the cutoff as "-50 kV", i.e. in WIEN2k sign.  "Higher than the cutoff" thus
  means V_C < -50 kV (larger magnitude) and "highest value in the neighboring
  pixels" means the neighbour with the largest V_s, i.e. the most negative V_C.
  See apply_nn_cutoff() for the two readings and their consequences.

XSF conventions (3ddens):
  General grid: n points span 0..L inclusive, the last plane is the periodic
  image of the first.  The duplicate end planes are dropped along all three
  axes right after reading, so cutoff (periodic wrap), normalisation and
  projection work on the periodic grid of n-1 points with voxel size L/(n-1),
  and every TIFF holds exactly one period (pixel j at j*L/(n-1), as in ixchel).
  Standard XSF data order is x fastest (Fortran order); it is detected
  automatically because older exports differed.

Supported beam directions: [001], [100], [010] (axis-parallel projections).
General [hkl] would need a rotated/interpolated grid and is rejected.
Image frame (x = TIFF columns, y = rows, z = layer stacking from 0) and the
crystruct --ixchel "vx vy vz" triplet whose cfg matches it:
  [001]:  x = +a, y = +b, z = +c      crystruct --ixchel "1 0 0  0 1 0  0 0 1"
  [100]:  x = +b, y = +c, z = +a      crystruct --ixchel "0 1 0  0 0 1  1 0 0"
  [010]:  x = +a, y = +c, z = +b      crystruct --ixchel "1 0 0  0 0 1  0 1 0"
The [010] frame is left-handed (a x c = -b).  crystruct accepts either
handedness, so a cfg made with a right-handed triplet such as
"-1 0 0  0 0 1  0 1 0" (the xsf2tiff_dwFactor.inplane_vectors frame) is
mirrored in x relative to these TIFFs.

Usage:
    python3 vcoul2tiff_ulm.py \\
        --xsf Graphene_vcoul_DFT.xsf \\
        --cfg Graphene_ixchel_001.cfg \\
        --miller 0 0 1 \\
        [--cutoff 50000] [--stencil 6|26] [--nn-value highest|lowest] \\
        [--no-normalize] [--order auto|F|C] [-o wien2k_ulm_001]

Dependencies: numpy, scipy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
import tifffile


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

BOHR      = 0.529177210903   # Bohr -> Å (not needed for XSF input, kept for reference)
RY_TO_V   = 13.605693122994  # 1 Ry = 13.606 eV -> potential of an electron: 13.606 V
H2_2PI_ME = 47.88            # h²/(2π m_e e) [V·Å²]  (Kirkland/ixchel unit of V_z)


# ═══════════════════════════════════════════════════════════════
# XSF reader (with automatic detection of the data order)
# ═══════════════════════════════════════════════════════════════

def _roughness(d):
    """Mean absolute neighbour difference over all three axes.  Correctly
    ordered potential data is smooth (except at the cusps) -> small; scrambled
    data is quasi-noise -> large."""
    return sum(np.mean(np.abs(np.diff(d, axis=ax))) for ax in range(3))


def _reshape_xsf(vals, nx, ny, nz, order):
    """order: 'F' (standard XSF, x fastest), 'C' (z fastest),
       'auto' (decide by smoothness)."""
    arr = np.asarray(vals[:nx * ny * nz], dtype=np.float64)
    if order in ('F', 'C'):
        return arr.reshape((nx, ny, nz), order=order), order
    dF = arr.reshape((nx, ny, nz), order='F')
    if nx == nz:
        # the C reshape is then exactly the F reshape with x and z swapped, and
        # the roughness (a sum over all axes) is identical for both
        print(f"    data order: 'F' (standard XSF)  [auto-detection cannot discriminate "
              f"for nx == nz; pass --order C for a non-standard file]")
        return dF, 'F'
    dC = arr.reshape((nx, ny, nz), order='C')
    rF, rC = _roughness(dF), _roughness(dC)
    if np.isclose(rF, rC, rtol=1e-6, atol=0.0):
        print(f"    data order: 'F' (standard XSF)  [auto-detection cannot discriminate: "
              f"roughness F={rF:.6g} ~ C={rC:.6g}; pass --order C for a non-standard file]")
        return dF, 'F'
    if rF < rC:
        print(f"    data order: 'F' (standard XSF)  [roughness F={rF:.4g} < C={rC:.4g}]")
        return dF, 'F'
    print(f"    data order: 'C' (non-standard)  [roughness C={rC:.4g} < F={rF:.4g}]")
    return dC, 'C'


def read_xsf(path, order='auto'):
    """
    Read a 3D data grid from an XSF file written by WIEN2k 3ddens.
    Span vectors are in Å, values in Ry.

    Returns: data [nx,ny,nz] (Ry), (ax,by,cz) [Å], (nx,ny,nz)
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
        sys.exit("Error: no DATAGRID_3D block found in XSF.")
    i += 1

    nx, ny, nz = map(int, lines[i].split());  i += 1
    i += 1  # origin (ignored)
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
        sys.exit(f"Error: too few data points ({len(vals)} < {nx*ny*nz}).")

    data, used = _reshape_xsf(vals, nx, ny, nz, order)

    ax_ang = np.linalg.norm(va)
    by_ang = np.linalg.norm(vb)
    cz_ang = np.linalg.norm(vc)

    return data, (ax_ang, by_ang, cz_ang), (nx, ny, nz)


def drop_periodic_duplicates(data):
    """
    Remove the duplicated end plane of the XSF general grid along each axis.

    3ddens writes n points spanning 0..L inclusive, so plane n-1 is the
    periodic image of plane 0.  The plane is dropped only if it really equals
    plane 0; otherwise a warning is printed and the plane is kept, i.e. that
    axis is taken as an already periodic grid of n points (voxel size L/n).

    Returns data with the periodic shape (n_x, n_y, n_z); the voxel size along
    each axis is then L/n for that n.
    """
    for ax, name in enumerate('xyz'):
        n = data.shape[ax]
        if n < 2:
            continue
        first = np.take(data, 0, axis=ax)
        last  = np.take(data, n - 1, axis=ax)
        if np.allclose(first, last, rtol=1e-5, atol=1e-8):
            data = np.take(data, np.arange(n - 1), axis=ax)
        else:
            dev = float(np.max(np.abs(first - last)))
            print(f"    WARNING: last {name} plane differs from the first (max |diff| = "
                  f"{dev:.3g} Ry); kept, {name} treated as a periodic grid of {n} points")
    return data


# ═══════════════════════════════════════════════════════════════
# cfg parser (ixchel crystruct.cfg)
# ═══════════════════════════════════════════════════════════════

def parse_cfg(cfg_path):
    """Read the layer geometry (size_uc, filename, stacking) from an ixchel cfg."""
    text = Path(cfg_path).read_text()

    sm = re.search(r'stack_seq\s*=\s*\"\d+\(([^)]+)\)\"', text)
    if not sm:
        sys.exit("Error: stack_seq not found in cfg.")
    seq = list(sm.group(1))

    lm = re.search(r'layers\s*=\s*\{', text)
    if not lm:
        sys.exit("Error: layers block not found in cfg.")
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
            sys.exit(f"Error: layer '{label}' not defined in layers block.")
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
# Nearest-neighbour cutoff (Kurasch et al. 2011, p. 397 and Fig. 3)
# ═══════════════════════════════════════════════════════════════

def _footprint(stencil, size):
    """Neighbourhood used to pick the replacement value.
    size == 3: 6 face neighbours (stencil=6) or the full 3x3x3 cube (stencil=26).
    size  > 3: growing cube, only used for flagged voxels that have no valid
               neighbour in the requested stencil (cluster of flagged voxels)."""
    if stencil == 26 or size > 3:
        return np.ones((size, size, size), dtype=bool)
    fp = np.zeros((3, 3, 3), dtype=bool)
    fp[1, 1, :] = fp[1, :, 1] = fp[:, 1, 1] = True
    return fp


def apply_nn_cutoff(data_3d, cutoff_ry, stencil=6, nn_value='highest'):
    """
    Replace voxels beyond the cutoff by a value taken from their neighbours.

    Paper (p. 397): "Whenever a value in the 3D potential is higher than some
    cutoff value, we change this value to the highest value in the neighboring
    pixels."  This is a replacement, not a clamp: the flagged voxel takes a
    neighbour's value, the threshold itself never appears in the data.

    Flagging.  data_3d is in WIEN2k sign (V_C < 0 at the nuclei) and the paper
    quotes its cutoff as "-50 kV", so "higher than the cutoff" is read as
    larger in magnitude:  flagged  <=>  V_C < -cutoff_ry.

    Replacement value ("highest value in the neighboring pixels"):
      nn_value='highest' (default, the paper's reading): the neighbour with the
        highest scattering potential V_s = -V_C, i.e. the most negative V_C —
        the neighbour closest to the nucleus.  The flagged voxel becomes a
        plateau at the level of its nearest neighbour.
      nn_value='lowest': the least negative neighbour (what a maximum_filter on
        WIEN2k-sign data gives; behaviour of earlier versions of this script).
        This carves a dip into the potential at the nucleus and is kept only
        for comparison.
    For a nucleus sitting exactly on a grid point of an isotropic grid all face
    neighbours are equal by symmetry and the two readings coincide; they differ
    for off-grid nuclei, anisotropic voxels and clusters of flagged voxels.

    Neighbourhood.  The paper says "neighboring pixels" without a stencil.
    Default is the 6 face neighbours (the literal nearest neighbours; with
    nn_value='highest' the 20 edge/corner voxels of a 3x3x3 cube can hardly
    win anyway because they are farther from the nucleus); stencil=26 uses the
    full cube.  Periodic boundaries (mode='wrap') because the cell is periodic;
    data_3d must therefore be the periodic grid without the duplicated XSF end
    planes (drop_periodic_duplicates), otherwise a boundary voxel would see its
    own flagged copy instead of its true neighbour.

    Clusters.  Only voxels below the threshold are ever changed, and replacement
    values are always taken from unflagged voxels.  A flagged voxel with a valid
    neighbour in the requested stencil takes its value from that stencil; only
    flagged voxels without any valid stencil neighbour (all flagged) search a
    cube that grows shell by shell until it contains an unflagged voxel.

    Returns (data, n_replaced) with n_replaced the number of voxels actually
    replaced.
    """
    data = data_3d.copy()
    flagged = data < -cutoff_ry
    if not flagged.any():
        return data, 0

    # mask flagged voxels with +/-inf so the rank filter ignores them
    if nn_value == 'highest':
        rank_filter, masked = minimum_filter, np.where(flagged, np.inf, data)
    else:
        rank_filter, masked = maximum_filter, np.where(flagged, -np.inf, data)

    todo = flagged.copy()
    size = 3
    for _ in range(20):
        nb  = rank_filter(masked, footprint=_footprint(stencil, size), mode='wrap')
        got = todo & np.isfinite(nb)
        data[got] = nb[got]
        todo &= ~got
        if not todo.any():
            break
        size += 2       # cluster of flagged voxels -> widen the search for the rest

    remaining = int(todo.sum())
    if remaining > 0:
        print(f"    WARNING: {remaining} voxels still below -cutoff "
              f"(no unflagged voxel within a {size - 2}^3 cube)")
    return data, int(flagged.sum()) - remaining


# ═══════════════════════════════════════════════════════════════
# Projection along an axis-parallel beam direction
# ═══════════════════════════════════════════════════════════════

def project_layer(data_cut, cell_ang, shape, miller, z0_ang, cz_lay):
    """
    Sum the 3D potential over the voxels of one ixchel layer along the beam.

    data_cut and shape describe the periodic grid (duplicate XSF end planes
    already removed): n points per axis, voxel size L/n.  Image frame: see
    the module docstring.

    Returns:
        proj_yx  : 2D array [nrow=y, ncol=x] in Ry (plain sum, no unit factor)
        d_proj   : voxel size along the projection axis [Å]
        n_slices : number of voxel planes summed
    """
    h, k, l = miller
    ax_ang, by_ang, cz_ang = cell_ang
    nx, ny, nz = shape

    nonzero = [x for x in [h, k, l] if x != 0]
    if len(nonzero) != 1:
        sys.exit(
            f"Error: Miller index [{h}{k}{l}] not supported.\n"
            f"  Only [001], [100], [010] (axis-parallel projections) are possible."
        )

    if l != 0:       # [001]: beam along c, sum over axis 2
        n_pts = nz;   span = cz_ang;   axis = 2
    elif h != 0:     # [100]: beam along a, sum over axis 0
        n_pts = nx;   span = ax_ang;   axis = 0
    else:            # [010]: beam along b, sum over axis 1
        n_pts = ny;   span = by_ang;   axis = 1

    # periodic grid: n_pts unique planes at i*span/n_pts (duplicate end plane
    # of the XSF general grid already removed)
    d_proj = span / n_pts                       # voxel size [Å]

    # map the layer boundaries (from the cfg) onto voxel planes
    i_lo = int(round(z0_ang            / span * n_pts))
    i_hi = int(round((z0_ang + cz_lay) / span * n_pts))
    i_lo = max(0, min(i_lo, n_pts))
    i_hi = max(0, min(i_hi, n_pts))
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
# TIFF output (ixchel format)
# ═══════════════════════════════════════════════════════════════

def write_ixchel_tiff(filename, data_yx):
    """Write an ixchel-compatible TIFF: page 0 RGB preview, page 1 float32 data,
    page 2 metadata placeholder."""
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
    print(f"    -> {filename}  [{vmin:.4g} ... {vmax:.4g}] 1/Å")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description='Kurasch et al. (2011) route: vcoul XSF -> ixchel TIFF')
    ap.add_argument('--xsf',    required=True,
                    help='total Coulomb potential as XSF (3ddens on case.vcoul; for CXY/CYZ '
                         'lattices use lapw5grid.py, 3ddens misses the centring atoms)')
    ap.add_argument('--cfg',    required=True,
                    help='ixchel crystruct.cfg (layer geometry)')
    ap.add_argument('--miller', type=int, nargs=3, default=[0, 0, 1],
                    metavar=('H', 'K', 'L'),
                    help='beam direction (default: 0 0 1)')
    ap.add_argument('--order',  choices=['auto', 'F', 'C'], default='auto',
                    help='XSF data order (default: auto)')
    ap.add_argument('--cutoff', type=float, default=50000,
                    help='cutoff magnitude in volt, the sign is ignored: voxels with '
                         'V_C < -|cutoff| are replaced (default: 50000, paper: -50 kV)')
    ap.add_argument('--stencil', type=int, choices=[6, 26], default=6,
                    help='neighbourhood for the replacement value (default: 6 face neighbours)')
    ap.add_argument('--nn-value', choices=['highest', 'lowest'], default='highest',
                    help="'highest' scattering potential among the neighbours (paper) "
                         "or 'lowest' (earlier behaviour); see apply_nn_cutoff()")
    ap.add_argument('--no-normalize', action='store_true',
                    help='skip the paper normalisation (smallest V_s set to zero)')
    ap.add_argument('-o', '--output', default='wien2k_ulm',
                    help='output directory')
    args = ap.parse_args()

    h, k, l   = args.miller
    cutoff_V  = abs(args.cutoff)        # paper "-50 kV" in WIEN2k sign == magnitude 50000
    cutoff_Ry = cutoff_V / RY_TO_V      # 50000 V / 13.606 V/Ry ~ 3675 Ry
    out_dir   = args.output

    # -- 1. read XSF -------------------------------------------------
    print(f"\n[1] reading XSF: {args.xsf}")
    data_3d, (ax_ang, by_ang, cz_ang), (nx, ny, nz) = read_xsf(args.xsf, args.order)
    print(f"    grid:       {nx} x {ny} x {nz} points (XSF general grid)")
    data_3d = drop_periodic_duplicates(data_3d)
    nx, ny, nz = data_3d.shape
    print(f"    periodic:   {nx} x {ny} x {nz} points (duplicate end planes removed)")
    print(f"    cell:       {ax_ang:.4f} x {by_ang:.4f} x {cz_ang:.4f} Å  "
          f"({nx/ax_ang:.1f} x {ny/by_ang:.1f} x {nz/cz_ang:.1f} points/Å)")
    print(f"    V_C (raw):  [{data_3d.min():.1f}, {data_3d.max():.3f}] Ry"
          f" = [{data_3d.min()*RY_TO_V:.0f}, {data_3d.max()*RY_TO_V:.2f}] V")

    # -- 2. nearest-neighbour cutoff ---------------------------------
    print(f"\n[2] nearest-neighbour cutoff at -{cutoff_V:.0f} V = -{cutoff_Ry:.1f} Ry "
          f"(stencil {args.stencil}, nn-value {args.nn_value})")
    if args.cutoff < 0:
        print(f"    (--cutoff {args.cutoff:.0f} taken as magnitude {cutoff_V:.0f} V)")
    n_over = int((data_3d < -cutoff_Ry).sum())
    print(f"    voxels below -cutoff: {n_over} of {data_3d.size} ({100*n_over/data_3d.size:.4f}%)")
    data_cut, n_replaced = apply_nn_cutoff(data_3d, cutoff_Ry, args.stencil, args.nn_value)
    print(f"    replaced:  {n_replaced} voxels")
    print(f"    V_C (after cutoff): [{data_cut.min()*RY_TO_V:.0f}, {data_cut.max()*RY_TO_V:.2f}] V")

    # -- 3. normalisation --------------------------------------------
    # Paper p. 397: shift so that the smallest value of V_s = -V_C is zero,
    # i.e. subtract the largest V_C (vacuum / interstitial plateau).  A constant
    # shift only adds a global phase in the multislice calculation, but it
    # matters for any comparison with IAM potentials, which vanish in vacuum.
    if args.no_normalize:
        v_ref = 0.0
        print(f"\n[3] normalisation skipped (--no-normalize)")
    else:
        v_ref = float(data_cut.max())
        data_cut = data_cut - v_ref
        print(f"\n[3] normalisation: V_C shifted by -{v_ref:.5f} Ry = -{v_ref*RY_TO_V:.3f} V "
              f"so that min(V_s) = 0")

    # -- 4. cfg ------------------------------------------------------
    print(f"\n[4] reading cfg: {args.cfg}")
    cfg    = parse_cfg(args.cfg)
    layers = cfg['layers']
    print(f"    {len(layers)} layer(s), total thickness {cfg['total_z']:.3f} Å,  beam [{h}{k}{l}]")

    # -- 5. projection per layer -------------------------------------
    #
    # unit chain:
    #   proj_raw [Ry]  = sum of V_C over n_slices voxels
    #   V_z [V·Å]      = -proj_raw * d_proj[Å] * RY_TO_V[V/Ry]     (V_s = -V_C)
    #   V_z [1/Å]      = V_z [V·Å] / H2_2PI_ME [V·Å²]
    # => to_kirk = d_proj * RY_TO_V / H2_2PI_ME  (no Bohr factor: XSF spans are in Å)
    print(f"\n[5] projecting layer by layer ...")
    outs = []
    for ly in layers:
        proj_raw, d_proj, n_slices = project_layer(
            data_cut, (ax_ang, by_ang, cz_ang), (nx, ny, nz),
            (h, k, l), ly['z0'], ly['cz'])
        if proj_raw is None:
            print(f"\n  layer '{ly['label']}': no voxel planes in "
                  f"z = {ly['z0']:.3f} ... {ly['z0']+ly['cz']:.3f} Å, skipped")
            continue
        to_kirk  = d_proj * RY_TO_V / H2_2PI_ME
        proj_yx  = -proj_raw * to_kirk   # V_s = -V_C
        print(f"\n  layer '{ly['label']}': z = {ly['z0']:.3f} ... {ly['z0']+ly['cz']:.3f} Å  "
              f"({n_slices} planes, d = {d_proj*1000:.1f} mÅ)")
        print(f"    V_z: [{proj_yx.min():.4f}, {proj_yx.max():.4f}] 1/Å"
              f" = [{proj_yx.min()*H2_2PI_ME:.1f}, {proj_yx.max()*H2_2PI_ME:.1f}] V·Å")
        fname = f"{out_dir}/{ly['filename']}"
        write_ixchel_tiff(fname, proj_yx)
        outs.append(fname)

    print(f"\nDone -- {len(outs)} file(s) in '{out_dir}/'")


if __name__ == '__main__':
    main()
