#!/usr/bin/env python3
"""
mip_compare.py — mean inner potential of graphene, DFT vs IAM, Kurasch route

Kurasch et al. (2011), p. 400: "For ideal graphene we find a difference in the
mean inner potential of 15.5%."  Both potentials go through identical
processing (30 ppÅ grid, nn cutoff, shift so that min V_s = 0, projection),
so the mean of the projected potential over the cell is the mean inner
potential times the layer thickness.

Prints
  * mean V_z of the vcoul2tiff_ulm.py TIFFs (DFT, IAM) and the relative change
  * the same from the 3D grids with and without the cutoff (effect of the
    four nucleus voxels on the mean)
  * vacuum flatness: plane-averaged V_C at z = c/2 and z = c/2 +- 1 Å

Usage:
    python3 mip_compare.py <casedir>
"""
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
import vcoul2tiff_ulm as ulm   # noqa: E402

RY_TO_V = ulm.RY_TO_V
H2 = ulm.H2_2PI_ME             # 47.88 V*Å²
CUTOFF_RY = 50000.0 / RY_TO_V

d = Path(sys.argv[1] if len(sys.argv) > 1 else '.')

res = {}
for k in ('DFT', 'IAM'):
    # ixchel TIFF: page 0 RGB preview, page 1 float32 data [1/Å], page 2 metadata
    tif = tifffile.imread(d / f'ulm_{k}_001' / 'atompota.tif', key=1).astype(np.float64)
    data, (_, _, cz), _ = ulm.read_xsf(d / f'Graphene_vcoul_{k}_ongrid.xsf')
    data = ulm.drop_periodic_duplicates(data)                    # periodic grid
    nz = data.shape[2]
    dz = cz / nz
    cut, nrep = ulm.apply_nn_cutoff(data, CUTOFF_RY, 6, 'highest')
    mip_raw = ((data.max() - data) * RY_TO_V).mean()             # [V], no cutoff
    mip_cut = ((cut.max() - cut) * RY_TO_V).mean()               # [V], with cutoff
    plane_mean = data.mean(axis=(0, 1)) * RY_TO_V                # [V] vs z
    izs = [nz // 2 - int(round(1.0 / dz)), nz // 2, nz // 2 + int(round(1.0 / dz))]
    res[k] = dict(tif=tif.mean() * H2 / cz, raw=mip_raw, cut=mip_cut)
    print(f"{k}: TIFF mean V_z = {tif.mean():.5f} 1/Å = {tif.mean() * H2:.3f} V*Å -> MIP {res[k]['tif']:.4f} V;"
          f"  3D grid: MIP {mip_cut:.4f} V with cutoff ({nrep} voxels), {mip_raw:.4f} V without")
    print(f"     vacuum: plane-averaged V_C at z = " +
          ", ".join(f"{iz * dz:.2f} Å: {plane_mean[iz]:.5f} V" for iz in izs) +
          f";  max V_C {data.max() * RY_TO_V:.5f} V")

dft, iam = res['DFT'], res['IAM']
print()
for key, what in (('tif', 'TIFFs (cutoff, normalised, projected)'),
                  ('cut', '3D grid with cutoff'), ('raw', '3D grid without cutoff')):
    print(f"MIP change DFT vs IAM, {what:38s}: {(dft[key] / iam[key] - 1) * 100:+.2f} %")
print("paper (p. 400): 15.5 %")
