#!/usr/bin/env python3
"""
vcoul2tiff_ulm.py — re-implementation of the Kurasch et al. (2011) route

Projects the total WIEN2k Coulomb potential (case.vcoul exported to XSF with
lapw5grid.py) onto ixchel-compatible TIFFs following the recipe of

    S. Kurasch, J. C. Meyer, D. Künzel, A. Groß, U. Kaiser,
    "Simulation of bonding effects in HRTEM images of light element materials",
    Beilstein J. Nanotechnol. 2, 394-404 (2011), doi:10.3762/bjnano.2.45

as literally as the paper allows; deviations from the paper are bugs, not
improvements.

Method (paper p. 397-399):
  1. Read the 3D potential from XSF  [Ry, the vectors that span the grid are in Å]
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
  apply_nn_cutoff() implements this reading.

XSF conventions (the general grid of XSF, as written by lapw5grid.py):
  n points span 0..L inclusive, the last plane is the periodic image of the
  first.  The duplicate end planes are dropped along all three axes right
  after reading, so cutoff (periodic wrap), normalisation and projection work
  on the periodic grid of n-1 points with voxel size L/(n-1), and every TIFF
  holds exactly one period (pixel j at j*L/(n-1), as in ixchel).  The values
  are listed with x running fastest.

Supported beam directions: [001], [100], [010] (axis-parallel projections).
General [hkl] would need a rotated/interpolated grid and is rejected.
Image frame (x = TIFF columns, y = rows, z = layer stacking from 0) and the
crystruct --ixchel "vx vy vz" triplet whose cfg matches it:
  [001]:  x = +a, y = +b, z = +c      crystruct --ixchel "1 0 0  0 1 0  0 0 1"
  [100]:  x = +b, y = +c, z = +a      crystruct --ixchel "0 1 0  0 0 1  1 0 0"
  [010]:  x = +a, y = +c, z = +b      crystruct --ixchel "1 0 0  0 0 1  0 1 0"
The [010] frame is left-handed (a x c = -b).  crystruct accepts either
handedness, so a cfg made with a right-handed triplet such as
"-1 0 0  0 0 1  0 1 0" is mirrored in x relative to these TIFFs.

Layout of this file: the CORE comes first, steps 2 to 5 of the method above
(apply_nn_cutoff, normalise, project_layer, to_kirkland).  After the core:
the file formats (the XSF reader of step 1, the ixchel cfg) and the command
line, main(), which calls the five steps in order.

Usage:
    python3 vcoul2tiff_ulm.py \\
        --xsf Graphene_vcoul_DFT.xsf \\
        --cfg Graphene_ixchel_001.cfg \\
        --miller 0 0 1 \\
        [--cutoff 50000] [--no-normalize] [-o wien2k_ulm_001]

Arguments and where they come from:
    --xsf        the Coulomb potential on a 3D grid: the -o file of lapw5grid.py
                 --clm case.vcoul_DFT (or case.vcoul_IAM) --grid NX NY NZ.  The grid
                 box must be the cell of the cfg (lapw5grid --edges = crystruct
                 triplet).
    --cfg        the ixchel crystal-structure cfg for this beam direction, written
                 by crystruct --ixchel "<triplet>" case.struct (triplets: see
                 "Supported beam directions" above).  Read from it: the layers
                 (thickness cz, file names).  The lateral cell size is taken from
                 the XSF and not compared with the cfg.
    --miller     beam direction h k l in units of the grid box axes; only 0 0 1,
                 1 0 0 and 0 1 0 (default 0 0 1)
    --cutoff     magnitude of the cutoff in volt (default 50000 = the paper's
                 -50 kV); see step 2
    --no-normalize  skip step 3 (smallest V_s set to zero)
    -o           output directory (default wien2k_ulm); one TIFF per layer, named
                 as in the cfg.  It is the -i input of tiff2ixchel.py.

Dependencies: numpy, tifffile
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ixchel_tiff import write_ixchel_tiff       # noqa: E402


# ═══════════════════════════════════════════════════════════════
# CORE: the method of Kurasch et al. (2011), steps 2 to 5 of the list above
#
#   step 2   apply_nn_cutoff   replace the voxels at the nuclei by a neighbour's value
#   step 3   normalise         shift the potential so that the vacuum level is zero
#   step 4   project_layer     sum the voxel planes of one layer along the beam
#   step 5   to_kirkland       convert the sum to the unit of ixchel, 1/Å
#
# Step 1, reading the XSF file, is a file format and comes after the core.
# main() at the end of the file calls the steps in order.
# ═══════════════════════════════════════════════════════════════

RY_TO_V   = 13.605693122994  # 1 Ry = 13.606 eV -> potential of an electron: 13.606 V
H2_2PI_ME = 47.88            # h²/(2π m_e e) [V·Å²]  (Kirkland/ixchel unit of V_z)


def apply_nn_cutoff(data_3d, cutoff_ry):
    """
    Step 2: replace the grid points beyond the cutoff by the value of a neighbour.

    Paper (p. 397): "Whenever a value in the 3D potential is higher than some
    cutoff value, we change this value to the highest value in the neighboring
    pixels."  The point takes a neighbour's value; the cutoff itself never
    appears in the data.

    Which points.  data_3d is in WIEN2k sign (V_C < 0 at the nuclei) and the
    paper quotes its cutoff as "-50 kV", so "higher than the cutoff" means
    larger in magnitude: a point is replaced if V_C < -cutoff_ry.

    Which value.  The paper's "highest value" refers to the scattering potential
    V_s = -V_C, so it is the most negative V_C among the neighbours: the value of
    the neighbour closest to the nucleus.

    Which neighbours.  The six points one grid step away along x, y and z.  The
    cell is periodic, so a point on the border has its neighbour on the opposite
    side (the % in the code); data_3d must therefore be the periodic grid without
    the duplicated XSF end planes (drop_periodic_duplicates).  A neighbour that
    is itself beyond the cutoff does not count.  A point with no neighbour left
    keeps its value and is reported.

    Returns (data, n_replaced) with n_replaced the number of points replaced.
    """
    data = data_3d.copy()
    nx, ny, nz = data_3d.shape
    n_replaced = n_left = 0
    for i, j, k in np.argwhere(data_3d < -cutoff_ry):                # the points beyond the cutoff
        neighbours = [data_3d[(i - 1) % nx, j, k], data_3d[(i + 1) % nx, j, k],
                      data_3d[i, (j - 1) % ny, k], data_3d[i, (j + 1) % ny, k],
                      data_3d[i, j, (k - 1) % nz], data_3d[i, j, (k + 1) % nz]]
        neighbours = [v for v in neighbours if v >= -cutoff_ry]      # a neighbour beyond the cutoff does not count
        if neighbours:
            data[i, j, k] = min(neighbours)                          # most negative V_C = highest V_s
            n_replaced += 1
        else:
            n_left += 1
    if n_left:
        print(f"    WARNING: {n_left} points beyond the cutoff have no neighbour within it and keep their value")
    return data, n_replaced


def normalise(data_cut):
    """
    Step 3: shift the potential so that its vacuum level is zero.

    Paper p. 397: "the potential was shifted (i.e., smallest value was set to
    zero)".  The smallest value of V_s = -V_C is the largest V_C (vacuum /
    interstitial plateau), which is subtracted.  A constant shift only adds a
    global phase in the multislice calculation, but it matters for any
    comparison with IAM potentials, which vanish in vacuum.

    Returns (shifted data, v_ref); v_ref is the value that was subtracted [Ry].
    """
    v_ref = float(data_cut.max())
    return data_cut - v_ref, v_ref


def project_layer(data_cut, cell_ang, miller, z0_ang, cz_lay):
    """
    Sum the 3D potential over the voxels of one ixchel layer along the beam.

    data_cut is the periodic grid (duplicate XSF end planes already removed):
    n points per axis, voxel size L/n, with L from cell_ang.  Image frame: see
    the module docstring.

    Returns:
        proj_yx  : 2D array [nrow=y, ncol=x] in Ry (plain sum, no unit factor)
        d_proj   : voxel size along the projection axis [Å]
        n_slices : number of voxel planes summed
    """
    h, k, l = miller
    if len([x for x in (h, k, l) if x != 0]) != 1:
        sys.exit(
            f"Error: Miller index [{h}{k}{l}] not supported.\n"
            f"  Only [001], [100], [010] (axis-parallel projections) are possible."
        )
    axis = 2 if l != 0 else 0 if h != 0 else 1      # beam along c for [001], a for [100], b for [010]

    # periodic grid: n_pts unique planes at i*span/n_pts (duplicate end plane
    # of the XSF general grid already removed)
    n_pts, span = data_cut.shape[axis], cell_ang[axis]
    d_proj = span / n_pts                           # voxel size [Å]

    # map the layer boundaries (from the cfg) onto voxel planes
    i_lo = int(round(z0_ang            / span * n_pts))
    i_hi = int(round((z0_ang + cz_lay) / span * n_pts))
    i_lo = max(0, min(i_lo, n_pts))
    i_hi = max(0, min(i_hi, n_pts))
    n_slices = i_hi - i_lo
    if n_slices == 0:
        return None, d_proj, 0

    window = [slice(None)] * 3
    window[axis] = slice(i_lo, i_hi)                # the planes of this layer, everything across
    proj = data_cut[tuple(window)].sum(axis=axis)
    return proj.T, d_proj, n_slices                 # ixchel: [nrow=y, ncol=x]


def to_kirkland(proj_raw, d_proj):
    """
    Step 5: the sum over the voxel planes of a layer in Kirkland units, as ixchel reads it.

    Unit chain:
      proj_raw [Ry]  = sum of V_C over the voxel planes (from project_layer)
      V_z [V·Å]      = -proj_raw * d_proj[Å] * RY_TO_V[V/Ry]     (V_s = -V_C)
      V_z [1/Å]      = V_z [V·Å] / H2_2PI_ME [V·Å²]
    There is no Bohr factor: the XSF spans are in Å.
    """
    to_kirk = d_proj * RY_TO_V / H2_2PI_ME
    return -proj_raw * to_kirk


# ═══════════════════════════════════════════════════════════════
# END OF CORE.  Below: file formats (XSF, ixchel cfg), command line
# ═══════════════════════════════════════════════════════════════

# ── file format: the XSF reader, step 1 ──

def read_xsf(path): #step 1 in order of execution
    """
    Read the 3D data grid of an XSF file written by lapw5grid.py.

    Returns: the values as an array [nx, ny, nz] (Ry) and the lengths (ax, by, cz)
    of the three vectors that span the grid (Å).
    """
    with open(path) as f:
        lines = f.readlines()

    # The data grid block of the file, here the 73 x 127 x 451 graphene grid written by lapw5grid.py:
    #     DATAGRID_3D_GRID                                         line i
    #     73 127 451                                               line i + 1: points along the three axes
    #       0.00000000E+00  0.00000000E+00  0.00000000E+00         line i + 2: origin of the grid (not needed)
    #       2.13128852E+00 -1.23050000E+00  0.00000000E+00         line i + 3: first vector that spans the grid [Å]
    #       2.13128852E+00  3.69150001E+00  0.00000000E+00         line i + 4: second vector
    #       0.00000000E+00  0.00000000E+00  1.50000001E+01         line i + 5: third vector
    #      -1.34201729E-01 -1.35038385E-01 -1.37563815E-01 ...     from line i + 6: the values [Ry]
    #     END_DATAGRID_3D
    for i, ln in enumerate(lines):                  # find line i, the one that opens the data grid
        if 'DATAGRID_3D' in ln and 'BEGIN_BLOCK' not in ln and 'END' not in ln:
            break

    nx, ny, nz = map(int, lines[i + 1].split())
    va = np.array(list(map(float, lines[i + 3].split())))
    vb = np.array(list(map(float, lines[i + 4].split())))
    vc = np.array(list(map(float, lines[i + 5].split())))

    vals = []                                       # all numbers up to the end of the block, in file order
    for ln in lines[i + 6:]:
        if len(vals) >= nx * ny * nz or 'END_DATAGRID' in ln:
            break
        vals.extend(float(x) for x in ln.split())

    # the list of numbers as an array [nx, ny, nz]; order='F': the first index (x) runs fastest in the file
    data = np.array(vals[:nx * ny * nz]).reshape((nx, ny, nz), order='F')
    return data, (np.linalg.norm(va), np.linalg.norm(vb), np.linalg.norm(vc))


def drop_periodic_duplicates(data):
    """
    Remove the last plane of the XSF grid along each axis.

    The XSF grid has n points from 0 to L, both ends included, so its last plane
    is the periodic image of the first.  Without it the grid is periodic: n - 1
    points per axis, voxel size L/(n-1).

    Stops if a last plane is not a copy of the first.  Then the grid box is not
    a periodic cell (lapw5grid.py --edges must be lattice vectors) or the file
    does not list its values with x running fastest.
    """
    for axis, name in enumerate('xyz'):
        first = np.take(data, 0, axis=axis)
        last = np.take(data, -1, axis=axis)
        if not np.allclose(first, last, rtol=1e-5, atol=1e-8):
            sys.exit(f"Error: the last {name} plane of the grid is not a copy of the first "
                     f"(largest difference {np.max(np.abs(first - last)):.3g} Ry); "
                     f"the grid box must be a periodic cell")
    return data[:-1, :-1, :-1]


# ── file format: the ixchel cfg written by crystruct (layer thickness, file name, stacking) ──

def parse_cfg(cfg_path):
    """Read the layers (thickness, file name) and their stacking from an ixchel cfg."""
    text = Path(cfg_path).read_text()

    # The parts of the cfg that are read, here rutile [001] with the two layers a and b (written by crystruct):
    #     stack_seq = "1(ab)";
    #     layers = {
    #         a = {
    #             size_uc = { x = 4.5939999819; y = 4.5939999819; z = 1.4790000021; };
    #             filename = "atompota.tif";
    #             atoms = ( ... );
    #         };
    #         b = { ... };
    #     };
    stack_match = re.search(r'stack_seq\s*=\s*\"\d+\(([^)]+)\)\"', text)

    # Every layer starts with a line "<letter> = {"; its text runs up to the start of the next layer.
    starts = list(re.finditer(r'^\s*([a-z])\s*=\s*\{', text, re.MULTILINE))
    layer_defs = {}                                 # label -> thickness [Å] and file name of that layer
    for n, start in enumerate(starts):
        end = starts[n + 1].start() if n + 1 < len(starts) else len(text)
        block = text[start.end():end]
        size_match = re.search(r'size_uc\s*=\s*\{[^}]*z\s*=\s*([0-9.E+\-]+)', block)
        file_match = re.search(r'filename\s*=\s*\"([^\"]+)\"', block)
        if size_match and file_match:
            layer_defs[start.group(1)] = {'cz': float(size_match.group(1)), 'filename': file_match.group(1)}

    # the layers in stacking order; z0 is the depth at which a layer starts
    layers = []
    z0 = 0.0
    for label in stack_match.group(1):              # the stacking sequence, one letter per layer
        ld = layer_defs[label]
        layers.append({'label': label, 'cz': ld['cz'], 'z0': z0, 'filename': ld['filename']})
        z0 += ld['cz']
    return {'layers': layers, 'total_z': z0}


# ── command line: the five steps in order ──────────────────────

def main():
    ap = argparse.ArgumentParser(description='Kurasch et al. (2011) route: vcoul XSF -> ixchel TIFF')
    ap.add_argument('--xsf', required=True, help='total Coulomb potential as XSF (lapw5grid.py on case.vcoul)')
    ap.add_argument('--cfg', required=True, help='ixchel crystruct.cfg (layer geometry)')
    ap.add_argument('--miller', type=int, nargs=3, default=[0, 0, 1], metavar=('H', 'K', 'L'),
                    help='beam direction (default: 0 0 1)')
    ap.add_argument('--cutoff', type=float, default=50000,
                    help='cutoff magnitude in volt, the sign is ignored: voxels with '
                         'V_C < -|cutoff| are replaced (default: 50000, paper: -50 kV)')
    ap.add_argument('--no-normalize', action='store_true',
                    help='skip the paper normalisation (smallest V_s set to zero)')
    ap.add_argument('-o', '--output', default='wien2k_ulm', help='output directory')
    args = ap.parse_args()

    h, k, l   = args.miller
    cutoff_V  = abs(args.cutoff)        # paper "-50 kV" in WIEN2k sign == magnitude 50000
    cutoff_Ry = cutoff_V / RY_TO_V      # 50000 V / 13.606 V/Ry ~ 3675 Ry
    out_dir   = args.output

    # The numbers in brackets that are printed, [1] ... [5], are the numbering of the log, not the steps
    # of the method: [4] is the cfg and [5] the projection together with the unit conversion.

    # -- 1. read XSF -------------------------------------------------
    print(f"\n[1] reading XSF: {args.xsf}")
    data_3d, (ax_ang, by_ang, cz_ang) = read_xsf(args.xsf)
    nx, ny, nz = data_3d.shape
    print(f"    grid:       {nx} x {ny} x {nz} points (XSF general grid)")
    data_3d = drop_periodic_duplicates(data_3d)
    nx, ny, nz = data_3d.shape
    print(f"    periodic:   {nx} x {ny} x {nz} points (duplicate end planes removed)")
    print(f"    cell:       {ax_ang:.4f} x {by_ang:.4f} x {cz_ang:.4f} Å  "
          f"({nx/ax_ang:.1f} x {ny/by_ang:.1f} x {nz/cz_ang:.1f} points/Å)")
    print(f"    V_C (raw):  [{data_3d.min():.1f}, {data_3d.max():.3f}] Ry"
          f" = [{data_3d.min()*RY_TO_V:.0f}, {data_3d.max()*RY_TO_V:.2f}] V")

    # -- 2. nearest-neighbour cutoff ---------------------------------
    print(f"\n[2] nearest-neighbour cutoff at -{cutoff_V:.0f} V = -{cutoff_Ry:.1f} Ry (6 neighbours)")
    if args.cutoff < 0:
        print(f"    (--cutoff {args.cutoff:.0f} taken as magnitude {cutoff_V:.0f} V)")
    n_over = int((data_3d < -cutoff_Ry).sum())
    print(f"    voxels below -cutoff: {n_over} of {data_3d.size} ({100*n_over/data_3d.size:.4f}%)")
    data_cut, n_replaced = apply_nn_cutoff(data_3d, cutoff_Ry)
    print(f"    replaced:  {n_replaced} voxels")
    print(f"    V_C (after cutoff): [{data_cut.min()*RY_TO_V:.0f}, {data_cut.max()*RY_TO_V:.2f}] V")

    # -- 3. normalisation --------------------------------------------
    if args.no_normalize:
        v_ref = 0.0
        print(f"\n[3] normalisation skipped (--no-normalize)")
    else:
        data_cut, v_ref = normalise(data_cut)
        print(f"\n[3] normalisation: V_C shifted by -{v_ref:.5f} Ry = -{v_ref*RY_TO_V:.3f} V "
              f"so that min(V_s) = 0")

    # -- the layers from the cfg -------------------------------------
    print(f"\n[4] reading cfg: {args.cfg}")
    cfg = parse_cfg(args.cfg)
    layers = cfg['layers']
    print(f"    {len(layers)} layer(s), total thickness {cfg['total_z']:.3f} Å,  beam [{h}{k}{l}]")

    # -- 4./5. projection per layer, conversion to 1/Å, TIFF ---------
    print(f"\n[5] projecting layer by layer ...")
    outs = []
    for ly in layers:
        proj_raw, d_proj, n_slices = project_layer(data_cut, (ax_ang, by_ang, cz_ang), (h, k, l),
                                                   ly['z0'], ly['cz'])
        if proj_raw is None:
            print(f"\n  layer '{ly['label']}': no voxel planes in "
                  f"z = {ly['z0']:.3f} ... {ly['z0']+ly['cz']:.3f} Å, skipped")
            continue
        proj_yx = to_kirkland(proj_raw, d_proj)
        print(f"\n  layer '{ly['label']}': z = {ly['z0']:.3f} ... {ly['z0']+ly['cz']:.3f} Å  "
              f"({n_slices} planes, d = {d_proj*1000:.1f} mÅ)")
        print(f"    V_z: [{proj_yx.min():.4f}, {proj_yx.max():.4f}] 1/Å"
              f" = [{proj_yx.min()*H2_2PI_ME:.1f}, {proj_yx.max()*H2_2PI_ME:.1f}] V·Å")
        fname = f"{out_dir}/{ly['filename']}"
        vmin, vmax = write_ixchel_tiff(fname, proj_yx)
        print(f"    -> {fname}  [{vmin:.4g} ... {vmax:.4g}] 1/Å")
        outs.append(fname)

    print(f"\nDone -- {len(outs)} file(s) in '{out_dir}/'")


if __name__ == '__main__':
    main()
