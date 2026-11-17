#!/usr/bin/env python3
"""
mip_compare.py — mean inner potential of graphene, DFT vs IAM, Kurasch route

Kurasch et al. (2011), p. 400: "For ideal graphene we find a difference in the
mean inner potential of 15.5%."  Both potentials go through identical
processing (30 ppÅ grid, nn cutoff, shift so that min V_s = 0, projection),
so the mean of the projected potential over the cell is the mean inner
potential times the thickness of the cell.

Prints the mean projected potential of the two layer files, the mean inner
potential that follows from it, and the relative change DFT vs IAM.

Usage:
    python3 mip_compare.py <casedir>

Argument and where it comes from:
    <casedir>    the run directory of the Kurasch route (default: the current
                 directory).  Read from it:
                   Graphene_ixchel_001.cfg         the crystruct cfg (thickness of the cell)
                   ulm_DFT_001/atompota.tif        written by vcoul2tiff_ulm.py
                   ulm_IAM_001/atompota.tif        (the same from case.vcoul_IAM)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ixchel_tiff import read_ixchel_tiff        # noqa: E402
from vcoul2tiff_ulm import H2_2PI_ME, parse_cfg  # noqa: E402

casedir = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
thickness = parse_cfg(casedir / 'Graphene_ixchel_001.cfg')['total_z']      # of the cell along the beam [Å]

mip = {}
for kind in ('DFT', 'IAM'):
    v_z = read_ixchel_tiff(casedir / f'ulm_{kind}_001' / 'atompota.tif')   # projected potential of one cell [1/Å]
    mean_vz = v_z.mean() * H2_2PI_ME                                       # its mean [V·Å]
    mip[kind] = mean_vz / thickness                                        # mean potential of the cell [V]
    print(f"{kind}: mean V_z = {v_z.mean():.5f} 1/Å = {mean_vz:.3f} V*Å -> mean inner potential {mip[kind]:.4f} V")

print(f"change of the mean inner potential, DFT vs IAM: {(mip['DFT'] / mip['IAM'] - 1) * 100:+.2f} %   "
      f"(paper, p. 400: 15.5 %)")
