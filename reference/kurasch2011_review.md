# Review of `src/vcoul2tiff_ulm.py` against Kurasch et al. (2011)

Reference: S. Kurasch, J. C. Meyer, D. Künzel, A. Groß, U. Kaiser,
*Simulation of bonding effects in HRTEM images of light element materials*,
Beilstein J. Nanotechnol. **2**, 394–404 (2011), doi:[10.3762/bjnano.2.45](https://doi.org/10.3762/bjnano.2.45)
(open access, CC-BY 2.0; the PDF is not part of this repository).

Reviewed: 2026-09-11, script version `vcoul2tiff_ulm4.py` (8 Jun 2026), now
`src/vcoul2tiff_ulm.py`. Test system: graphene, `~/wien2k_tests/Graphene`
(CXY cell 2.461 × 4.264 × 15.0 Å, C at (0, 1/6, 0)), WIEN2k 24.1.

## 0. What the paper actually provides

* The text on p. 397 and p. 399 (quoted below) is the **only** description of
  the cutoff. Supporting Information File 2 (the modified `wien2venus.py`,
  `…/2190-4286-2-45-S2.txt`) contains **no cutoff code** — it only drives
  `lapw5c` slice by slice (`RHO` / `ATU VAL NODEBUG` / `NONORTHO`) and adds
  the origin-shift and sub-volume options mentioned on p. 397. The cutoff and
  the projection were done elsewhere and are not published.
* Fig. 2 was made with a single C atom in a 10 Å³ cell, 250 ppÅ line scan.
  Fig. 3 was made from the same single-atom potential.

## 1. Cutoff semantics

Paper p. 397:

> "Interestingly, for 30 ppÅ, the sampling error is significant only for the
> center pixel and is caused by a single value in the 3D potential. We use a
> very simple method to handle this problem: Whenever a value in the 3D
> potential is higher than some cutoff value, we change this value to the
> highest value in the neighboring pixels."

p. 399: "… projected along the *z*-direction using a cutoff of −50 kV".
Fig. 3 legend: "nn cutoff 50000".

* **Replacement, not clamping** — the old code did this correctly
  (`np.where(bad, nb_max, data)`; the threshold value never enters the data).
* **Threshold sign.** WIEN2k `case.vcoul` is the Coulomb potential *energy of
  an electron* in Ry: negative at the nuclei (−1.6 MV at the on-grid nucleus,
  see §6), +2.31 V in the vacuum of the graphene cell. The paper quotes
  "−50 kV", i.e. WIEN2k sign, while plotting the positive scattering potential
  V_s = −V_C in Figs. 2 and 3. "Higher than the cutoff" therefore has to be
  read as *larger in magnitude*: flagged ⇔ V_C < −50 kV. The old code did this
  (`data < -cutoff_ry`). ✓
* **Replacement value — deviation found.** Read consistently in the paper's
  positive convention, "the highest value in the neighboring pixels" is the
  neighbour with the largest V_s, i.e. the *most negative* V_C: the neighbour
  closest to the nucleus. The old code applied `maximum_filter` to
  WIEN2k-sign data, which picks the *least* negative neighbour (smallest |V|),
  i.e. the opposite reading — it carves a dip into the potential at the nucleus
  instead of a plateau. Fixed: `--nn-value highest` (default, `minimum_filter`
  on the valid neighbours) implements the paper; `--nn-value lowest` keeps the
  old behaviour for comparison.
* Numerically the two readings coincide for a nucleus exactly on a grid point
  of an isotropic grid (all six face neighbours are equal by symmetry: 446 vs
  444 V·Å in §6). They differ for off-grid nuclei, anisotropic voxels and
  clusters of flagged voxels.

## 2. Neighbourhood

The paper never specifies the stencil ("neighboring pixels", legend "nn").
Decision: default to the **6 face neighbours** — the literal nearest
neighbours on the grid. With the paper's reading of "highest" the 20
edge/corner voxels of a 3×3×3 cube practically never win because they are
farther from the nucleus, so the choice hardly matters for the default. It
does matter for the old reading: with `--nn-value lowest` the 26-stencil
picks a corner voxel at √3 × 33 mÅ and lowers the central pixel by a further
8 % (410 vs 444 V·Å). `--stencil 26` is available. Boundaries are periodic
(`mode='wrap'`), which is correct for a periodic cell (the duplicate XSF end
plane is harmless here). Documented in the `apply_nn_cutoff()` docstring.

## 3. Sampling rate and layer boundaries

p. 397: "a constant sampling rate of 30 ppÅ in the *z*-direction (parallel to
the incident beam), which turned out to be a realistic compromise, but a much
higher rate in the perpendicular direction. … The blue boxes in Figure 3
indicate the mean value within one pixel at a resolution of 30 ppÅ".
p. 399: "The 3D potentials were sliced with a resolution of 30 ppÅ".

* The finer perpendicular rate is the *analysis* grid used for the green
  curve of Fig. 3 only; the production potentials were 30 ppÅ isotropic. The
  handover's statement that the paper "uses … a much finer rate perpendicular"
  applies to Fig. 3, not to the images.
* The script takes whatever grid `3ddens` wrote. The existing graphene export
  (`74 128 450` in `case.in3d`) is a 30 ppÅ grid **if** the XSF is read as a
  general grid with n−1 intervals (73/2.461 = 29.7 ppÅ etc.); the file indeed
  has identical first and last planes along all three axes. `ulm4` handles
  this (voxel size L/(n−1), duplicate plane dropped) — the earlier
  `vcoul2tiff_ulm.py` (18 May) did not, and also reshaped in C order,
  scrambling the voxels. Both were confirmed on the data
  (roughness F = 0.05 vs C = 0.29).
* Layer boundaries: `z0`, `cz` from the cfg are mapped to voxel planes with
  `round(z/L·(n−1))`; for the single 15 Å graphene layer this gives planes
  0…449, i.e. all 450 unique planes exactly once. ✓ Note that a cfg whose
  layer boundaries are not multiples of the voxel size is rounded to the
  nearest plane (no sub-voxel splitting) — inherent to the method.

## 4. Normalisation — deviation found

p. 397: "Far from the core, the WIEN2k potential approaches a non-zero constant
value. In order to obtain the usual normalization, the potential was shifted
(i.e., smallest value was set to zero)." p. 399: "sliced, normalized and
projected".

The old code did not shift. In the graphene cell the vacuum level of
`vcoul` is +2.314 V (WIEN2k sign), so V_s = −V_C is −2.31 V in vacuum and the
projected 15 Å layer carries a constant −34.7 V·Å = −0.72 Å⁻¹ everywhere
(the old TIFF's minimum is −0.584 Å⁻¹). For the multislice intensity a
constant per-slice offset is only a global phase, but every comparison with
IAM/ixchel potentials (which vanish in vacuum), every ⟨ΔV⟩ and every peak
ratio is biased by it. Fixed: the script now subtracts max(V_C) after the
cutoff (`min(V_s) = 0`, exactly the paper's rule); `--no-normalize` restores
the old behaviour. For bulk crystals without vacuum the "smallest value" is
the interstitial minimum, which is *not* the vacuum zero of the IAM — a
known limitation of the paper's rule, not of the script.

## 5. Units

* h²/(2π mₑ e) = (6.62607015e-34)² / (2π · 9.1093837e-31 · 1.602176634e-19)
  = 4.788e-19 V·m² = **47.88 V·Å²**. ✓ Kirkland's projected potential in V·Å
  divided by this is the dimensionless-per-length quantity ixchel stores
  (1/Å), consistent with `kirkland_layer_fp()` in `vcoul2tiff.py`.
* XSF route: `to_kirk = d_proj[Å] · 13.6057[V/Ry] / 47.88` per summed voxel. ✓
  Native route (`vcoul2tiff.py`): `RY_TO_EV · BOHR / H2_2PI_ME` = 0.1504 per
  Ry·Bohr. ✓ Both checked by re-derivation; the script now also prints V·Å.
* Label inconsistency in `compare_all_subsliced.py` fixed: the data are in
  1/Å (ixchel TIFFs), so every label and the printed ⟨DFT−IAM⟩ now say 1/Å.

## 6. Sanity target — reproduction of Fig. 3 on graphene

`lapw5` x–z plane through a C nucleus (x: ±0.117 Å at 600 ppÅ, z: full 15 Å
at 30 ppÅ; input `lapw5_fig3/Graphene.in5_fig3`), plus a `3ddens` export on a
75 × 127 × 451 grid chosen so that the nucleus at y = 1/6 sits **exactly on a
voxel** (126/6 = 21). Script: `src/plot_kurasch_fig3.py`, figure:
`reference/kurasch_fig3_graphene.png`.

| quantity (central 30 ppÅ pixel, 15 Å column) | V·Å | 1/Å |
|---|---|---|
| WIEN2k voxel at the nucleus (r = r₀ = 1e-4 Bohr) | 1.632 MV | |
| no cutoff (paper Fig. 3 green peak ≈ 50 000) | **54 781** | 1144 |
| nn cutoff, paper reading (highest / 6 neighbours), normalised | **446** | 9.31 |
| nn cutoff, old reading (lowest / 6) | 444 | 9.28 |
| nn cutoff, old reading (lowest / 26) | 410 | 8.56 |
| old code exactly (lowest / 26 / no shift) | 375 | 7.83 |
| "ideal": 2-D area mean of the 600 ppÅ curve over the pixel | **642** | 13.4 |
| Kirkland IAM (C), same area mean | 566 | 11.8 |
| ratio ideal / cutoff (paper: "within a factor of three") | **1.44** | |
| ratio no-cutoff / ideal (paper: "up to two orders of magnitude") | **85** | |

The reproduction matches the paper's numbers (50 000 V·Å peak, factor < 3
after the cutoff, ~2 orders without) and confirms the mechanism: the cutoff
only ever touches the one voxel that sits on the nucleus (plus its 7 periodic
images on the grid faces).

**Consequence for the existing data.** On the grid actually used so far
(`74 128 450`) the nucleus is 5.6 mÅ off the nearest voxel in y; that voxel is
−15 kV, far above the −50 kV threshold, so **the cutoff is a no-op** and the
central pixel comes out at 871 V·Å (≈ 1.4 × the ideal) — the overshoot is
smaller than in the on-grid case but is not treated at all. The paper's
threshold is tuned to the on-grid singularity; whether a nucleus lands on a
voxel depends on the atom position and `in3d` grid, which should be checked
per material (the script prints the most negative voxel).

## 7. Patch summary (`src/vcoul2tiff_ulm.py`)

1. `apply_nn_cutoff()`: replacement value = neighbour with the highest V_s
   (`minimum_filter` on WIEN2k-sign data); 6-face stencil by default;
   options `--nn-value`, `--stencil`; cluster handling unchanged.
2. Paper normalisation (subtract max V_C so that min V_s = 0); `--no-normalize`.
3. Output also reported in V·Å; comments/docstrings in English; otherwise the
   XSF reader, cfg parser, projection and TIFF writer of `ulm4` are unchanged.

## 8. How to regenerate the Fig. 3 check

```bash
cd ~/wien2k_tests/Graphene
# 1. high-resolution x-z plane through the nucleus (1.7 s)
cp <repo>/reference/lapw5_fig3/Graphene.in5_fig3 <repo>/reference/lapw5_fig3/lapw5_fig3.def .
lapw5 lapw5_fig3.def                       # writes Graphene.rho_fig3 (Ry)
# 2. optional: 3D grid with the nucleus on a voxel (75 127 451 in case.in3d_ongrid)
$WIENROOT/3ddens 3ddens_vcoul_ongrid.def   # writes Graphene_vcoul_ongrid.xsf
# 3. analysis and figure
python3 <repo>/src/plot_kurasch_fig3.py Graphene.rho_fig3 kurasch_fig3_graphene.png
python3 <repo>/src/vcoul2tiff_ulm.py --xsf Graphene_vcoul_ongrid.xsf \
        --cfg Graphene_ixchel_001.cfg -o ulm_ongrid
```
