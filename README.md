# wien2k-hrtem-potentials

Ab-initio scattering potentials from [WIEN2k](http://www.wien2k.at) for HRTEM
multislice simulations with **ixchel**, instead of the usual Independent Atom
Model (IAM).

Master's thesis project, TU Wien (Technical Physics), supervised by
Assoc. Prof. Stefan Löffler (USTEM).

## What this does

A multislice code needs, for every slice, the projected electrostatic potential
V_z(x, y) = ∫ V(x, y, z) dz. ixchel builds it from Kirkland's parametrised
isolated-atom potentials. Here we take the self-consistent Coulomb potential of
the full crystal from WIEN2k (`case.vcoul`) and convert it to the same
per-layer float32 TIFFs ixchel expects, so that bonding effects (charge
redistribution) enter the image simulation.

The hard part is the −Z/r divergence of the potential at the nuclei, which
cannot be sampled on a grid. Three routes exist:

| script | approach | status |
|---|---|---|
| `src/vcoul2tiff.py` | **V_z = V_Kirkland + ΔV_z** with ΔV = vcoul_DFT − vcoul_IAM projected from the *native* WIEN2k representation (plane waves in the interstitial, Abel transform of the (l,m) channels in the spheres). The nuclear terms cancel in the difference, no cutoff is needed. | main route |
| `src/vcoul2tiff_ulm.py` | Re-implementation of Kurasch et al. (2011), the "Ulm" route: sample `case.vcoul` on an equidistant grid via `3ddens`, replace voxels beyond a −50 kV cutoff by their nearest-neighbour value, normalise, sum over z. Exists for comparison. | reviewed against the paper, see [`reference/kurasch2011_review.md`](reference/kurasch2011_review.md) |
| `src/xsf2tiff_dwFactor.py` | Older grid route: Δρ = ρ_DFT − ρ_IAM from `3ddens`, FFT Poisson solver for ΔV, numerical line integral along an arbitrary beam direction, Kirkland IAM with Debye–Waller factors and sub-slicing. Independent cross-check. | cross-check |

Cross-validation between `vcoul2tiff.py` and `xsf2tiff_dwFactor.py` gave
r = 0.999 for Si.

Helpers:

* `src/compare_all_subsliced.py` — compares `wien2k_<hkl>/atompot*.tif` with
  `ixchel_<hkl>/atompot*.tif` (Pearson r, peak ratio, mean offset, plots).
* `src/run_subsliced.sh` — regenerates the `xsf2tiff_dwFactor.py` TIFFs for
  Rutile, Si and Graphite_ort and runs the comparison.
* `src/plot_kurasch_fig2.py` — radial WIEN2k vs Kirkland potential of a single
  atom (paper Fig. 2).
* `src/plot_kurasch_fig3.py` — projected potential near the core with and
  without the nearest-neighbour cutoff, from a `lapw5` plane (paper Fig. 3).

## Repository layout

```
src/            the scripts listed above
reference/      review notes on Kurasch et al. 2011, Fig. 3 reproduction,
                lapw5/3ddens inputs used for it (no paper PDF — see the DOI)
docs/           thesis exposé and presentation (to be added)
docs/figures/   figures used by the presentation (to be added)
```

## Requirements

Python ≥ 3.10 with `numpy`, `scipy`, `tifffile`, `matplotlib`;
`xsf2tiff_dwFactor.py` additionally needs `pylibconfig2`. A working WIEN2k
installation (`$WIENROOT`, `x`, `lapw0`, `lapw5`, `3ddens`) to produce the
inputs. Kirkland's `fparams.dat` for the IAM part.

```bash
python3 -m venv venv && venv/bin/pip install numpy scipy tifffile matplotlib pylibconfig2
```

## Usage

All scripts take the ixchel `crystruct.cfg` for the layer geometry
(`size_uc`, `stack_seq`, atom positions) and write `<outdir>/<filename>.tif`
per layer, in ixchel's unit (Kirkland units, 1/Å = V·Å / 47.88 V·Å²).

Main route (native `vcoul`, no grid):

```bash
cd ~/wien2k_tests/Graphene
python3 <repo>/src/vcoul2tiff.py \
    --vcoul-dft Graphene.vcoul_DFT --vcoul-iam Graphene.vcoul_IAM \
    --struct Graphene.struct --fparams fparams.dat \
    --cfg Graphene_ixchel_001.cfg --miller 0 0 1 -o wien2k_001
```

Ulm route (needs the total potential on a grid, see *Regenerating* below):

```bash
python3 <repo>/src/vcoul2tiff_ulm.py \
    --xsf Graphene_vcoul.xsf --cfg Graphene_ixchel_001.cfg --miller 0 0 1 \
    [--cutoff 50000] [--stencil 6|26] [--nn-value highest|lowest] [--no-normalize] \
    -o wien2k_ulm_001
```

Grid route with Poisson solver:

```bash
python3 <repo>/src/xsf2tiff_dwFactor.py \
    --dft Rutile_DFT_total.xsf --iam Rutile_IAM_total.xsf --struct Rutile.struct \
    --fparams fparams.dat --cfg ixchel_110/Rutile_ixchel_110.cfg --miller 1 1 0 -o wien2k_110
```

Comparison with ixchel (run inside the material directory, expects
`wien2k_<hkl>/` and `ixchel_<hkl>/`):

```bash
python3 <repo>/src/compare_all_subsliced.py -p Rutile -d 001 100 110
```

## Test data (not in this repository)

The WIEN2k cases live in `~/wien2k_tests/<Material>/` on the development
machine: `Si` (face-centred), `Rutile` (primitive; directions 001/100/110),
`Graphite_ort` (primitive), `Graphene` (base-centred), plus `FeF2`,
`Graphite_conv`, `Si/Si_conv`. Each holds multi-MB `vcoul_DFT`/`vcoul_IAM`
pairs, `3ddens` XSF exports and generated TIFFs — none of that is committed
(`.gitignore`). ixchel config files sit in sub-directories, e.g.
`~/wien2k_tests/Rutile/ixchel_110/Rutile_ixchel_110.cfg`.

### Regenerating the inputs

DFT/IAM pair of Coulomb potentials (`case.in0` must contain `R2V`, not
`NR2V` — a zero-byte `vcoul` means `NR2V` was still set):

```bash
cd ~/wien2k_tests/<Material>
cp <case>.clmsum <case>.clmsum_scf_backup
x dstart; x lapw0; cp <case>.vcoul <case>.vcoul_IAM     # superposed atoms
cp <case>.clmsum_scf_backup <case>.clmsum
x lapw0; cp <case>.vcoul <case>.vcoul_DFT               # self-consistent
```

Total potential on a grid for the Ulm route (`3ddens` with `case.vcoul` on
unit 9; `case.in3d` = grid `nx ny nz`, three `0.0 0.0` range lines, `add`,
`conv`; the `add` is a no-op when unit 11 is a dummy file):

```
 5,'<case>.in3d',        'old',     'formatted',0
21,'<case>_vcoul.xsf',   'unknown', 'formatted',0
 8,'<case>.struct',      'old',     'formatted',0
 9,'<case>.vcoul',       'old',     'formatted',0
11,'dummyfile',          'unknown', 'formatted',0
```

```bash
$WIENROOT/3ddens 3ddens_vcoul.def
```

`3ddens` writes an XSF *general grid*: n points span the cell inclusive of
both ends, so 30 points/Å means `n = 30·L + 1` and the last plane duplicates
the first. Whether a nucleus falls exactly on a voxel depends on the atom
position and the grid (see the review, §6).

## Status and open items

* `vcoul2tiff_ulm.py` has now been run end to end on graphene (two grids) and
  reproduces the paper's Fig. 3 behaviour; two deviations from the paper were
  found and fixed (replacement value of the cutoff, missing normalisation),
  see `reference/kurasch2011_review.md`. The paper does not specify the
  neighbourhood stencil; 6 face neighbours are used by default (`--stencil 26`
  available).
* The −50 kV cutoff only acts when a voxel sits (almost) exactly on a nucleus.
  On the existing graphene grid (`74 128 450`) it is a no-op; the central
  pixel is then ≈ 1.4 × the pixel-averaged ideal.
* Correlation against ixchel is lowest for Graphite_ort (r ≈ 0.94), which is
  also where the bonding correction is largest (≈ 26 % of the Kirkland peak).
  Physics or residual bug — open.
* Third route, discussed but not implemented: regularise the full DFT
  potential by convolving with the thermal-displacement Gaussian
  (σ = √(B/8π²), giving −Z·erf(r/σ√2)/r, finite at r = 0); would remove the
  need for both the Kirkland basis and the difference. Future work.
* `docs/` (exposé, presentation, figures) still to be added.

## Reference

S. Kurasch, J. C. Meyer, D. Künzel, A. Groß, U. Kaiser, *Simulation of bonding
effects in HRTEM images of light element materials*, Beilstein J. Nanotechnol.
**2**, 394–404 (2011), doi:[10.3762/bjnano.2.45](https://doi.org/10.3762/bjnano.2.45)
(open access, CC-BY). Their Supporting Information File 2 is a modified
`wien2venus.py` that only drives `lapw5c`; it contains no cutoff code.
