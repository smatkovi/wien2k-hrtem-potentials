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

Inputs: a `lapw5` x–z plane through a C nucleus (x at 600 ppÅ over ±0.117 Å,
z over the full 15 Å at 30 ppÅ; `reference/lapw5_fig3/`) and 3D grids of
75 × 127 × 451 points, chosen so that **all four** nuclei of the cell sit
exactly on voxels (126/6 = 21, 74/2 = 37). Script: `src/plot_kurasch_fig3.py`;
figure: `reference/kurasch_fig3_graphene.png`.

| quantity (central 30 ppÅ pixel, 15 Å column) | V·Å | 1/Å |
|---|---|---|
| WIEN2k voxel at the nucleus (r = r₀ = 1e-4 bohr) | 1.632 MV | |
| no cutoff (paper Fig. 3 green peak ≈ 50 000) | **54 781** | 1144 |
| nn cutoff, paper reading (highest, 6 neighbours), normalised | **446** | 9.31 |
| nn cutoff, old reading (lowest, 6) | 444 | 9.28 |
| nn cutoff, old reading (lowest, 26) | 410 | 8.56 |
| "ideal": true mean of V_z over the pixel column (fine z integration of the spherical part inside the sphere) | **563** | 11.8 |
| paper-style ideal: 2-D pixel mean of the green curve (30 ppÅ z sampling) | 642 | 13.4 |
| Kirkland IAM (C), same pixel mean | 566 | 11.8 |
| ratio ideal / cutoff (paper: "within a factor of three") | **1.26** (paper-style 1.44) | |
| ratio no cutoff / ideal (paper: "up to two orders of magnitude") | **97** (paper-style 85) | |

The reproduction matches the paper: a peak of about 50 000 V·Å, within a
factor of three after the cutoff, about two orders of magnitude without it.
Near the core the true pixel mean agrees with the Kirkland value to 0.5 %.

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

## 7. Summary of changes to `src/vcoul2tiff_ulm.py`

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

## 8. How to regenerate the Fig. 3 check

```bash
cd ~/wien2k_tests/Graphene
cp <repo>/reference/lapw5_fig3/* .        # Graphene.in5_fig3, lapw5_fig3.def,
                                          # Graphene.in3d_ongrid, 3ddens_vcoul_ongrid.def
# 1. high-resolution x-z plane through the nucleus (about 2 s)
lapw5 lapw5_fig3.def                      # writes Graphene.rho_fig3 (Ry)
# 2. 3D grid with all nuclei on voxels (lapw5grid handles the CXY centring)
$PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul \
    --grid 75 127 451 -o Graphene_vcoul_ongrid.xsf
#    (3ddens 3ddens_vcoul_ongrid.def writes the same grid but misses the
#     sphere contributions of the centring atoms)
# 3. analysis and figure
$PY <repo>/src/plot_kurasch_fig3.py Graphene.rho_fig3 kurasch_fig3_graphene.png
$PY <repo>/src/vcoul2tiff_ulm.py --xsf Graphene_vcoul_ongrid.xsf \
    --cfg Graphene_ixchel_001.cfg -o ulm_ongrid
```
