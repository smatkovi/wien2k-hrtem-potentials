# Review of `src/vcoul2tiff_ulm.py` against Kurasch et al. (2011)

Reference: S. Kurasch, J. C. Meyer, D. Künzel, A. Groß, U. Kaiser,
*Simulation of bonding effects in HRTEM images of light element materials*,
Beilstein J. Nanotechnol. **2**, 394–404 (2011), doi:[10.3762/bjnano.2.45](https://doi.org/10.3762/bjnano.2.45)
(open access, CC-BY 2.0; the PDF is not part of this repository).

First review 2026-09-11 (script version `vcoul2tiff_ulm4.py` of 8 Jun 2026),
revised the same day after an independent multi-agent code review and
re-validation. Test system: graphene, `~/wien2k_tests/Graphene` (CXY cell
2.461 × 4.264 × 15.0 Å, C listed at (0, 1/6, 0) and (0, 5/6, 0), centring
copies at (1/2, 2/3, 0) and (1/2, 1/3, 0)), WIEN2k 24.1.

Second revision 2026-10-13: re-run on clean graphene data
(`~/wien2k_tests/Graphene/lebF/Graphene`: Lebedev reduction off,
scalar-relativistic valence, neutral IAM; see the README *Known issues*).
Fig. 3 now follows the paper's definitions (§6); Fig. 2 (§7) and the
mean-inner-potential check through the full route (§8) are added.

## 0. What the paper actually provides

* The text on p. 397 and p. 399 (quoted below) is the **only** description of
  the cutoff. Supporting Information File 2 (the modified `wien2venus.py`,
  `…/2190-4286-2-45-S2.txt`) contains **no cutoff code**. It only drives
  `lapw5c` slice by slice (`RHO` / `ATU VAL NODEBUG` / `NONORTHO`) and adds
  the origin-shift and sub-volume options mentioned on p. 397. The cutoff and
  the projection were done elsewhere and are not published.
* Fig. 2 was made with a single C atom in a 10 Å × 10 Å × 10 Å cell and a
  250 ppÅ line scan (p. 397). The paper does not say which potential Fig. 3
  was computed from.
* `wien2venus.py` slices along x at x_i = i/n_x (n_x planes per period, no
  duplicate) and samples each y–z plane with n_y × n_z points including both
  ends (spacing L/(n−1)), NSHELL 3 3 3. Its optional shift (dx, dy, dz) moves
  the grid off the nuclei; without it the grid starts at the cell origin, so
  nuclei at rational positions can fall on voxels, the case the −50 kV cutoff
  is made for (§6).
* Supporting Information File 1 holds the WIEN2k convergence tests for ideal
  graphene: 100 k-points suffice; between RKmax 7 and 9 the projected potential
  changes by at most 0.06 %; GMAX 16 is recommended (default 12 within
  0.25 %); a vacuum of 20–25 Å decouples the layers. Its total energies of the
  two-atom cell (−152.40 … −152.416 Ry) are a direct check of an SCF: the
  clean rerun gives −152.41590 Ry, the old SCF −152.45159 Ry.

## 1. Cutoff semantics

Paper p. 397:

> "Interestingly, for 30 ppÅ, the sampling error is significant only for the
> center pixel and is caused by a single value in the 3D potential. We use a
> very simple method to handle this problem: Whenever a value in the 3D
> potential is higher than some cutoff value, we change this value to the
> highest value in the neighboring pixels."

p. 399: "… projected along the *z*-direction using a cutoff of −50 kV".
Fig. 3 legend: "nn cutoff 50000".

* **Replacement, not clamping.** The original code already did this
  (`np.where(bad, nb_max, data)`); the threshold value never enters the data.
* **Threshold sign.** WIEN2k `case.vcoul` holds the Coulomb potential *energy
  of an electron* in Ry: −1.6 MV at a nucleus that sits on a voxel (§6) and
  +2.31 V in the vacuum of the graphene cell. The paper quotes "−50 kV", i.e.
  in WIEN2k sign, while Figs. 2 and 3 plot the positive scattering potential
  V_s = −V_C. "Higher than the cutoff" therefore means *larger in magnitude*:
  a voxel is flagged if V_C < −50 kV. The original code did this.
* **Replacement value (deviation, fixed).** In the paper's positive
  convention, "the highest value in the neighboring pixels" is the neighbour
  with the largest V_s, i.e. the *most negative* V_C, which is the neighbour
  closest to the nucleus. The original code applied `maximum_filter` to
  WIEN2k-sign data and so picked the *least* negative neighbour. Fixed:
  `--nn-value highest` (default) implements the paper, `--nn-value lowest`
  keeps the old reading.
* For a nucleus on a voxel of the nearly isotropic 30 ppÅ grid the face
  neighbours are almost, but not exactly, equal (grid spacings 33.3/33.8/33.3
  mÅ), so the two readings give nearly the same central pixel (446 vs
  444 V·Å). They differ more for off-grid nuclei, anisotropic voxels and
  clusters of flagged voxels.

## 2. Neighbourhood

The paper does not specify the stencil ("neighboring pixels", legend "nn").
The default is the **6 face neighbours**, the literal nearest neighbours on
the grid. With the paper's reading of "highest", the 20 edge and corner
voxels of a 3×3×3 cube rarely win because they are farther from the nucleus.
With `--nn-value lowest`, the 26-stencil picks a corner voxel and lowers the
central pixel by a further 8 % (410 vs 444 V·Å). `--stencil 26` is available.
When a flagged voxel has no valid neighbour at all (a cluster), only that voxel
searches a wider cube. The cutoff runs on the periodic grid after the
duplicated XSF end planes have been removed (§3), so `mode='wrap'` reaches the
true periodic neighbour.

## 3. Sampling rate and layer boundaries

p. 397: "a constant sampling rate of 30 ppÅ in the *z*-direction (parallel to
the incident beam), which turned out to be a realistic compromise, but a much
higher rate in the perpendicular direction. … The blue boxes in Figure 3
indicate the mean value within one pixel at a resolution of 30 ppÅ".
p. 399: "The 3D potentials were sliced with a resolution of 30 ppÅ".

* The higher perpendicular rate belongs to the analysis behind Fig. 3 (the
  green curve); the production potentials were sliced at 30 ppÅ.
* 3ddens writes an XSF *general grid*: n points span each axis including both
  ends, and the last plane repeats the first. The script now removes the
  duplicated plane on **all three axes** right after reading. Voxel size is
  L/(n−1), and each TIFF holds exactly one period. The original script removed
  it only along the beam, so the TIFFs repeated their first row and column.
  The even older `vcoul2tiff_ulm.py` (18 May) also reshaped the data in C
  order, which scrambles the voxels (roughness F = 0.05 vs C = 0.29 on the
  graphene file).
* The existing graphene export `74 128 450` has 73 × 127 × 449 unique points,
  i.e. 29.7 × 29.8 × 29.9 ppÅ, and the 15 Å layer sums all 449 planes. For
  `75 127 451` all 450 planes are summed.
* When nx = nz, the F and C reshapes are exact transposes and cannot be told
  apart from the data. The script then assumes the standard order F and says
  so (`--order` overrides).
* Layer boundaries from the cfg are rounded to whole voxel planes; there is no
  sub-voxel splitting, which is inherent to the method.

## 4. Normalisation (deviation, fixed)

p. 397: "Far from the core, the WIEN2k potential approaches a non-zero constant
value. In order to obtain the usual normalization, the potential was shifted
(i.e., smallest value was set to zero)." p. 399: "sliced, normalized and
projected".

The original code did not shift. In the graphene cell the vacuum level of
`vcoul` is +2.314 V (WIEN2k sign), so V_s = −V_C is −2.31 V in vacuum and the
projected 15 Å layer carries a constant −34.7 V·Å = −0.72 1/Å (the old TIFF
minimum was −0.584 1/Å). For the multislice intensity a constant per-slice
offset is only a global phase, but any comparison with IAM potentials, which
vanish in vacuum, is biased by it. Fixed: after the cutoff the script
subtracts max(V_C), so min(V_s) = 0 as in the paper; `--no-normalize`
restores the old behaviour. For bulk crystals without vacuum the "smallest
value" is the interstitial minimum, not the vacuum zero of the IAM; this is a
limitation of the paper's rule, not of the script.

## 5. Units

* h²/(2π mₑ e) = (6.62607015e-34)² / (2π · 9.1093837e-31 · 1.602176634e-19)
  = 4.788e-19 V·m² = **47.88 V·Å²**. The Kirkland projected potential in V·Å
  divided by this is what ixchel stores (1/Å); ixchel builds the transmission
  function as exp(i λ(1 + E/511 keV) V_z), which `check_transmission.py`
  confirms to 3.5e-8 rad.
* XSF route: `to_kirk = d[Å] · 13.6057 [V/Ry] / 47.88` per summed voxel.
* All labels in `compare_all_subsliced.py` now say 1/Å.

## 6. Sanity target: reproduction of Fig. 3 on graphene

Inputs: `lapw5c` x–z planes through a C nucleus (x at 600 or 3000 ppÅ over
±0.117 Å, z over the full 15 Å at 30 ppÅ, nucleus on a sample;
`reference/lapw5_fig3/`) and 3D grids of 75 × 127 × 451 points, chosen so
that **all four** nuclei of the cell sit exactly on voxels (126/6 = 21,
74/2 = 37). Script: `src/plot_kurasch_fig3.py`; figure:
`reference/kurasch_fig3_graphene.png` (3000 ppÅ plane, clean data of
2026-10-13).

The curves follow the paper's definitions:

* green: V_z(x) = Σ_z V_s(x, z) dz with z at 30 ppÅ and x at the fine rate;
* red: the same after the nn cutoff, applied for every lateral position on the
  30 ppÅ grid laid through that column (neighbours at x ± 1/30 Å and
  z ± 1/30 Å; the neighbours normal to the plane are farther from the nucleus
  and never win). Only columns whose z = 0 voxel lies within 1.7 mÅ of the
  nucleus exceed 50 kV, hence the narrow notch of the paper's figure;
* blue: "the mean value of the green line at a resolution of 30 ppÅ"
  (caption), i.e. the 1-D mean of the green curve over each pixel. The first
  revision of this review read the boxes as 2-D pixel means (642 V·Å); the
  box heights of the paper confirm the 1-D reading (pixel 2: paper ≈ 396,
  1-D mean 392, 2-D mean 383 V·Å; pixel 1 below).

The 1-D mean over the centre pixel does not converge with the lateral
sampling of the green curve: its z = 0 sample grows like 1/|x| towards the
core (finite only because lapw5 holds V constant below R0 = 1e-4 bohr), so the
mean is 3570 / 1671 / 1519 / 1515 V·Å at 600 / 3000 / 6000 / 30 000 ppÅ
(continuum limit 1518). The paper's ≈ 1660 V·Å corresponds to about 3000 ppÅ.
The true mean of the projected potential over the pixel column (fine z
integration of the spherical part inside the sphere) is 563 V·Å; the figure
draws it as an outline.

| quantity (central 30 ppÅ pixel, 15 Å column) | paper (read off Fig. 3) | V·Å | 1/Å |
|---|---|---|---|
| WIEN2k voxel at the nucleus (r = r₀ = 1e-4 bohr) | | 1.632 MV | |
| no cutoff | ≈ 57 000 | **54 781** | 1144 |
| nn cutoff, paper reading (highest, 6 neighbours), normalised | ≈ 453 | **446** (notch 446–450) | 9.31 |
| nn cutoff, old reading (lowest, 6) | | 444 | 9.28 |
| nn cutoff, old reading (lowest, 26) | | 410 | 8.56 |
| blue box, pixel 1: 1-D mean of the green line | ≈ 1660 | **1671** (3000 ppÅ), 1518 (continuum) | |
| blue box, pixel 2 | ≈ 396 | **392** | |
| true pixel-column mean (not in the paper) | | **563** | 11.8 |
| 2-D pixel mean of the green curve (first revision's "paper-style") | | 642 | 13.4 |
| Doyle–Turner projected potential at the core | ≈ 386 | 390 | |
| Kirkland IAM (C), pixel mean | | 566 | 11.8 |
| ratio blue box / cutoff (paper: "within a factor of three") | ≈ 3.7 | **3.7** (3000 ppÅ box), 3.4 (continuum box); true mean: 1.26 | |
| ratio no cutoff / true mean (paper: "up to two orders of magnitude") | | **97** | |

The reproduction matches the paper. Against the paper's own box the cutoff
value is a factor 3.4–3.7 low, against the true pixel-column mean only 1.26. Near
the core the true pixel mean agrees with the Kirkland value to 0.5 %. The
numbers are the same on the old graphene SCF: the core region is not affected
by the Lebedev reduction or the relativistic mode.

**3ddens and centred lattices.** WIEN2k 24.1 `3ddens` adds the centring
translations only for F, B, CXZ and R lattices (`SRC_3ddens/3ddens.F:403`).
In every graphene (CXY) XSF written by 3ddens, the atoms at (1/2, 1/3, 0) and
(1/2, 2/3, 0) therefore have no sphere contribution, only the smooth
plane-wave value (on the 74 × 128 × 450 grid −206 V instead of −3.9 kV at
the voxel nearest to the nucleus; on a grid through the nucleus the −1.63 MV
cusp is missing).
`src/lapw5grid.py` writes the same XSF with lapw5, which handles the centring;
it agrees with 3ddens to 1e-8 V everywhere else and to 1e-9 Ry on rutile (P).

| grid (flagged voxels) | XSF from | listed C (0, 1/6) | centring C (1/2, 1/3) |
|---|---|---|---|
| 75 × 127 × 451 (nuclei on voxels) | 3ddens (2 flagged) | 446 V·Å | 142 V·Å |
| 75 × 127 × 451 (nuclei on voxels) | lapw5grid (4 flagged) | 446 V·Å | 446 V·Å |
| 74 × 128 × 450 (existing export; no voxel on a nucleus, 0 flagged) | 3ddens | 871 V·Å (nearest voxel 5.6 mÅ away) | 141 V·Å |
| 74 × 128 × 450 | lapw5grid | 871 V·Å | 471 V·Å (nearest voxel 20.2 mÅ away) |

On the existing grid the −50 kV cutoff never triggers, and the central pixel
depends strongly on how far the nearest voxel is from the nucleus
(1.55 × or 0.84 × the ideal 563 V·Å here). The paper's threshold is tuned to
a nucleus on a voxel; check this per material (the script prints the number
of flagged voxels).

## 7. Fig. 2: single carbon atom

p. 397: a single C atom in a 10 Å × 10 Å × 10 Å cell, a line scan of the
electrostatic potential at 250 ppÅ, shifted so that the smallest value is
zero; the caption calls it the WIEN2k *starting* potential. Reproduced with
`reference/fig2_C_single/C_single.struct` (P, a = 10 Å, C at the origin,
RMT 1.33 bohr as in graphene, R0 = 1e-4 bohr, RELA), `init_lapw -b`, a checked
`x dstart` and `x lapw0`; `src/plot_kurasch_fig2.py --vcoul` drives `lapw5c`
for the line scan along x over half the cell (1251 points, 0–5 Å). Figure:
`reference/kurasch_fig2_carbon.png`.

| r [Å] | WIEN2k [V] | Kirkland [V] | dev. | Doyle–Turner [V] |
|---|---|---|---|---|
| 0.004 | 21 200 | 21 345 | −0.7 % | 1587 |
| 0.020 | 3928 | 3928 | 0.0 % | 1530 |
| 0.100 | 560.0 | 560.3 | −0.06 % | 668.6 |
| 0.500 | 39.78 | 40.24 | −1.1 % | 39.51 |
| 1.000 | 4.457 | 4.508 | −1.1 % | 4.421 |
| 2.000 | 0.1056 | 0.1002 | +5.4 % | 0.0970 |

As in the paper, WIEN2k and Kirkland agree closely, while Doyle–Turner is
finite at the core (1589 V; paper inset ≈ 1550–1600 V) and falls off too fast
beyond 2 Å. Beyond about 1.5 Å the WIEN2k values (< 1 V) depend on the
normalisation constant and the periodic images; the small kink just below
5 Å (half the cell) appears in the paper's figure as well. Doyle–Turner
parameters: Doyle & Turner, Acta Cryst. A24, 390 (1968); Kirkland parameters:
ixchel `dat/fparams.dat`.

## 8. Mean inner potential of graphene through the full route

p. 400: "For ideal graphene we find a difference in the mean inner potential
of 15.5%."; p. 398: "The absolute differences between the DFT and IAM
potentials are in the range of 10–30%". The DFT (SCF) and IAM (`x dstart`,
checked) potentials of `Graphene/lebF/Graphene` were evaluated on the same
75 × 127 × 451 grid and projected with `vcoul2tiff_ulm.py`:

| quantity | DFT | IAM |
|---|---|---|
| vacuum level of V_C (shifted to V_s = 0) | +2.318 V | +2.815 V |
| voxels replaced by the cutoff | 4 | 4 |
| mean of V_z (TIFF) | 0.80334 1/Å = 38.46 V·Å | 0.95389 1/Å = 45.67 V·Å |
| mean inner potential (15 Å cell) | 2.564 V | 3.045 V |

Change DFT vs IAM: **−15.78 %** (paper 15.5 %). Without the cutoff the four
nucleus voxels (−1.63 MV each) add 1.55 V to both means and the change drops
to −10.45 %, so the cutoff matters for this number too. The relative
difference (V_z,DFT − V_z,IAM)/V_z,IAM ranges from −30.6 % (bond region;
−30.2 % and −30.0 % at the vertical and inclined bond centres, −22.0 % at the
hexagon centre) to −1.7 % (at the nuclei), in line with the paper's 10–30 %. The
vacuum is flat to 1e-5 V over ±1 Å around the cell centre, as the
"smallest value = 0" normalisation of a slab requires.

## 9. Summary of changes to `src/vcoul2tiff_ulm.py`

1. `apply_nn_cutoff()`: replacement value = neighbour with the highest V_s;
   6-face stencil by default (`--nn-value`, `--stencil`); only voxels without
   any valid stencil neighbour use the widened cube.
2. Paper normalisation (subtract max V_C so that min V_s = 0); `--no-normalize`.
3. Duplicated XSF end planes removed on all axes before cutoff, normalisation
   and projection.
4. Data order: standard F assumed when the roughness test cannot decide
   (nx = nz).
5. `--cutoff` is a magnitude (−50000 and 50000 are equivalent); the log counts
   only voxels that were actually replaced.
6. The [010] in-plane frame is x = +a, y = +c, which is left-handed; crystruct
   accepts both handednesses, so the matching triplet is documented in the
   docstring rather than changed.

## 10. How to regenerate the checks

```bash
cd ~/wien2k_tests/Graphene/lebF/Graphene  # README step 1 done: vcoul_DFT, vcoul_IAM; case.vcoul = vcoul_DFT
cp <repo>/reference/lapw5_fig3/* .        # Graphene.in5_fig3(_3000), lapw5_fig3(_3000).def,
                                          # Graphene.in3d_ongrid, 3ddens_vcoul_ongrid.def
# 1. high-resolution x-z plane through the nucleus (600 ppÅ: 2 s, 3000 ppÅ: 25 s)
lapw5c lapw5_fig3_3000.def                # writes Graphene.rho_fig3_3000 (Ry)
# 2. 3D grids with all nuclei on voxels (lapw5grid handles the CXY centring)
for k in DFT IAM; do
  $PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul_$k \
      --grid 75 127 451 -o Graphene_vcoul_${k}_ongrid.xsf
  $PY <repo>/src/vcoul2tiff_ulm.py --xsf Graphene_vcoul_${k}_ongrid.xsf \
      --cfg Graphene_ixchel_001.cfg -o ulm_${k}_001
done
#    (3ddens 3ddens_vcoul_ongrid.def writes the same grid from Graphene.vcoul
#     but misses the sphere contributions of the centring atoms)
# 3. Fig. 3 and the mean inner potential (§6, §8)
$PY <repo>/src/plot_kurasch_fig3.py Graphene.rho_fig3_3000 kurasch_fig3_graphene.png
$PY -c "import tifffile; d, i = (tifffile.imread(f'ulm_{k}_001/atompota.tif', key=1).mean() for k in ('DFT', 'IAM')); print(f'MIP change {100*(d/i-1):+.2f} %')"
# 4. Fig. 2 (§7): README step 5e, case ~/wien2k_tests/C_single
```
