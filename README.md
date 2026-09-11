# wien2k-hrtem-potentials

Ab-initio scattering potentials from [WIEN2k](http://www.wien2k.at) for HRTEM
multislice simulations with **ixchel**, instead of the usual Independent Atom
Model (IAM).

Master's thesis project, TU Wien (Technical Physics), supervised by
Assoc. Prof. Stefan Löffler (USTEM).

## What this does

A multislice code needs, for every slice, the projected electrostatic
potential V_z(x, y) = ∫ V(x, y, z) dz. ixchel builds it from Kirkland's
parametrised isolated-atom potentials. Here the bonding correction from a
self-consistent WIEN2k calculation is added:

    V_z = V_z^Kirkland + ΔV_z,     ΔV = V_DFT − V_IAM

where V_DFT is the Coulomb potential of the SCF density and V_IAM that of
the superposed atomic densities (`dstart`). The −Z/r nuclear terms cancel in
the difference, so ΔV is finite everywhere and needs no cutoff.

> **Results produced before 2026-09-11 are invalid.** A review found, among
> others: the ΔV sign was inverted in both `vcoul2tiff.py` and
> `xsf2tiff_dwFactor.py`; the old analytic `vcoul2tiff.py` evaluated the
> WIEN2k plane-wave stars, local rotations and centring atoms incorrectly;
> `xsf2tiff_dwFactor.py` read XSF data in the wrong axis order; the TIFFs
> were never usable by `ixchel main` (one unit cell instead of the `dim`
> grid); `compare_all_subsliced.py` misregistered the unit cell. The earlier
> Si cross-check r = 0.999 was dominated by the Kirkland term and did not test
> ΔV. Details: `reference/kurasch2011_review.md` and the git history.

## Routes

| script | ΔV from | status |
|---|---|---|
| `src/vcoul2tiff.py` | `case.vcoul_DFT − case.vcoul_IAM`, evaluated by **lapw5** (`RHO SUB`) on a grid over each layer, integrated along the beam | **main route**, validated (see below) |
| `src/xsf2tiff_dwFactor.py` | Δρ = ρ_DFT − ρ_IAM from XSF densities, FFT Poisson solver, line integral | independent cross-check; agrees with the main route (Rutile [001]: r = 1.0000) |
| `src/vcoul2tiff_ulm.py` | no ΔV: total `vcoul` on a grid with the nearest-neighbour cutoff of Kurasch et al. (2011) | faithful re-implementation for comparison |

Both ΔV routes share the cfg parser, the layer depth windows and the Kirkland
term (`kirkland_layer`, a port of ixchel's `atompot` including subslicing and
the band limit).

Tools:

* `src/lapw5grid.py` — evaluates `case.vcoul`, `case.clmsum`, … with lapw5 on
  planes or 3D grids and writes XSF in the 3ddens convention. Use it instead of
  `3ddens` for **CXY/CYZ lattices** (graphene): WIEN2k 24.1 `3ddens` adds the
  centring translations only for F, B, CXZ and R (`SRC_3ddens/3ddens.F:403`) and
  misses the sphere contributions of the centring atoms.
* `src/tiff2ixchel.py` — tiles a one-unit-cell layer TIFF NCELL times and
  resamples it onto the ixchel `dim` grid (exact Fourier interpolation), with
  ixchel's cfg checksum. Required before `ixchel main`.
* `src/check_transmission.py` — checks ixchel's transmission function
  `trans_<label>.tif` against exp(iσV) of the layer potential.
* `src/compare_all_subsliced.py` — compares `wien2k_<hkl>/` with ixchel's
  `ixchel_<hkl>/` layer potentials on an exactly registered unit cell
  (Pearson r, peak ratio, mean offset, plots).
* `src/run_subsliced.sh` — runs `xsf2tiff_dwFactor.py` and the comparison for
  Rutile, Si and Graphite_ort.
* `src/plot_kurasch_fig2.py`, `src/plot_kurasch_fig3.py` — reproductions of
  Figs. 2 and 3 of the paper.

## Requirements

* WIEN2k (`$WIENROOT` on `PATH`: `x`, `lapw0`, `dstart`, `lapw5`, `3ddens`)
* ixchel (`ixchel atompot`, `ixchel main`, `ixchel image`) and crystruct
* Python ≥ 3.10 with numpy, scipy, tifffile, matplotlib, pylibconfig2

```bash
python3 -m venv ~/venvs/hrtem
~/venvs/hrtem/bin/pip install numpy scipy tifffile matplotlib pylibconfig2
```

Run all scripts with that interpreter (`~/venvs/hrtem/bin/python src/...`).

## Workflow

The example is graphene; `<case>` is the WIEN2k case name, `<repo>` this
repository, `PY` the Python interpreter above.

### 1. WIEN2k: SCF and the two Coulomb potentials

After a converged SCF (`run_lapw`), `case.in0` line 2 must read `R2V`, not
`NR2V`, and must **not** contain `NONORM` (lapw5 needs the standard r²
normalisation). Edit it with an editor if necessary.

```bash
cd ~/wien2k_tests/Graphene
x lapw0                                   # SCF density -> case.vcoul
cp Graphene.vcoul Graphene.vcoul_DFT

cp Graphene.clmsum Graphene.clmsum_scf_backup
x dstart                                  # overwrites case.clmsum with superposed atoms
grep -E ':NEC01|Normalization of interstitial' Graphene.outputd
x lapw0
cp Graphene.vcoul Graphene.vcoul_IAM
cp Graphene.clmsum Graphene.clmsum_IAM           # IAM density, for check 5d
cp Graphene.clmsum_scf_backup Graphene.clmsum   # restore the SCF density!
```

The `grep` must show the `:NEC01` line (nuclear and electronic charge equal)
and, if needed, the renormalisation. A zero-byte `vcoul` means `NR2V` was
still set.

### 2. crystruct: layer geometry

Create the ixchel crystal cfg with crystruct for the beam direction, e.g.
`crystruct --ixchel "1 0 0  0 1 0  0 0 1" Graphene.struct` for [001]. Note the
triplet: `vcoul2tiff.py --frame` takes the same nine numbers. crystruct accepts
both handednesses (the existing Rutile [110] cfg uses `1 -1 0  0 0 1  1 1 0`).

### 3. Conversion (main route)

```bash
$PY <repo>/src/vcoul2tiff.py \
    --vcoul-dft Graphene.vcoul_DFT --vcoul-iam Graphene.vcoul_IAM \
    --struct Graphene.struct --cfg Graphene_ixchel_001.cfg \
    --frame 1 0 0  0 1 0  0 0 1 \
    --vacuum-depth 7.5 \
    -o wien2k_001
```

* `--frame` (or `--miller h k l`) sets the TIFF axes. The script matches the
  cfg atoms to the struct atoms (including centring copies), prints the offset
  and the largest mismatch, and stops if they do not coincide.
* `--vacuum-depth D` (slabs only): cfg depth of a plane in the vacuum. ΔV is
  shifted to zero there, so both potentials refer to the vacuum level like the
  Kirkland term. The script also checks the planes at D ± 1 Å and warns if ΔV
  is not constant (charged IAM density or plane not in vacuum). Without it the
  ΔV offset follows lapw0's arbitrary potential zero: irrelevant for images,
  but mean values are then meaningless.
* `--dz` (default 0.05 Å) and `--dx` (0.025 Å) are converged for graphene to
  about 1e-5 1/Å. Runtime: about 20 s for graphene on 16 cores.
* Output: `wien2k_001/atompot*.tif`, one unit cell per layer, in 1/Å.

### 4. ixchel

```bash
mkdir -p sim_dft_001 && cd sim_dft_001
cp ../Graphene_ixchel_001.cfg crystal.cfg           # same bytes -> same checksum
cp /path/to/simulation.cfg sim.cfg                  # dim must equal the crystal cfg's dim
$PY <repo>/src/tiff2ixchel.py --cfg crystal.cfg -i ../wien2k_001 -o .
ixchel main  -c crystal.cfg -s sim.cfg
ixchel image -c crystal.cfg -s sim.cfg
```

Do **not** run `ixchel atompot` in this directory; it would overwrite the
WIEN2k layer files with the IAM. For the IAM reference, run `ixchel atompot`
and `ixchel main` in a second directory with the same cfgs.

### 5. Plots and checks

**a) Geometry/unit null test.** With `--vcoul-dft` = `--vcoul-iam` the result
is pure Kirkland and must reproduce `ixchel atompot`:

```bash
mkdir -p nulltest && cd nulltest
ln -s ../ixchel_001 ixchel_001                       # ixchel atompot TIFFs + cfg
$PY <repo>/src/vcoul2tiff.py --vcoul-dft ../Graphene.vcoul_IAM --vcoul-iam ../Graphene.vcoul_IAM \
    --struct ../Graphene.struct --cfg ../Graphene_ixchel_001.cfg -o wien2k_001
$PY <repo>/src/compare_all_subsliced.py -p Graphene -d 001
```

Expected: r = 1.0000, ratio = 1.00, mean offset 0.000.

**b) DFT vs IAM comparison and plots.** The same command with the real
`wien2k_001/` writes `Graphene_subslicing_comparison.png` (line profile, both
maps, difference).

**c) What ixchel actually uses.** Add `writeTrans = 1;` to `sim.cfg`, run
`ixchel main`, then

```bash
$PY <repo>/src/check_transmission.py --crystal crystal.cfg --sim sim.cfg \
    --pot atompota.tif --trans trans_a.tif \
    --ref-pot ../sim_iam_001/atompota.tif --ref-trans ../sim_iam_001/trans_a.tif
```

Expected: |t| = 1 and phase deviations ≲ 1e-7 rad.

**d) Independent ΔV cross-check** with the density route. For P, F, B, CXZ
lattices 3ddens densities are fine; for CXY/CYZ generate them with lapw5grid
(same grid for both; `case.clmsum_IAM` is the `case.clmsum` written by
`x dstart` in step 1):

```bash
$PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.clmsum     --norm tot --grid 100 100 150 -o Graphene_DFT_rho.xsf
$PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.clmsum_IAM --norm tot --grid 100 100 150 -o Graphene_IAM_rho.xsf
$PY <repo>/src/xsf2tiff_dwFactor.py --dft Graphene_DFT_rho.xsf --iam Graphene_IAM_rho.xsf \
    --struct Graphene.struct --cfg Graphene_ixchel_001.cfg --miller 0 0 1 -o wien2k_rho_001
```

Compare `wien2k_001` and `wien2k_rho_001` after removing the means (the
Poisson route has zero-mean ΔV by construction).

**e) Kurasch et al. route and Fig. 3.**

```bash
$PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul --grid 75 127 451 -o Graphene_vcoul.xsf
$PY <repo>/src/vcoul2tiff_ulm.py --xsf Graphene_vcoul.xsf --cfg Graphene_ixchel_001.cfg -o wien2k_ulm_001
lapw5 lapw5_fig3.def        # inputs in reference/lapw5_fig3/
$PY <repo>/src/plot_kurasch_fig3.py Graphene.rho_fig3 kurasch_fig3_graphene.png
```

## Validation (2026-09-11, graphene CXY and rutile P4₂/mnm)

| check | result |
|---|---|
| lapw5grid vs 3ddens, rutile vcoul 138×138×89 | max 1e-9 Ry (output rounding) |
| lapw5grid vs 3ddens, rutile clmsum 100³ | identical |
| lapw5grid vs 3ddens, graphene | identical except inside the spheres of the centring atoms (3ddens bug, up to 3.7 kV) |
| null test (Kirkland only) vs `ixchel atompot`, tiled | graphene 7.7e-6, rutile [001]/[100] 4.9e-5, [110] 9.3e-5 1/Å |
| ΔV main route vs Poisson route, rutile [001] (3ddens densities) | r = 1.0000, rms difference 0.4 % of the ΔV rms |
| ΔV main route vs Poisson route, graphene (lapw5grid densities) | r = 1.0000, rms difference 0.2 % |
| graphene ΔV symmetry (mirrors / centring shift) | 2e-7 / 1e-4 1/Å |
| ΔV convergence: dz 67 mÅ vs 2 mÅ; 25 mÅ lateral aliasing | 1.3e-5 / 6e-6 1/Å |
| `ixchel main` transmission vs exp(iσV) | 3.5e-8 rad |
| graphene mean inner potential, DFT vs Kirkland (neutral IAM, vacuum-referenced) | −15.8 % (paper: 15.5 %) |
| `x lapw0` recipe reproduces `Graphene.vcoul_DFT` | 1e-15 |

## Test data (not in this repository)

WIEN2k cases live in `~/wien2k_tests/<Material>/`: `Si` (F), `Rutile` (P;
cfgs for 001/100/110), `Graphite_ort` (P), `Graphene` (CXY), plus `FeF2`,
`Graphite_conv`, `Si/Si_conv`. They hold multi-MB `vcoul`, XSF and TIFF files;
none of that is committed (`.gitignore`). ixchel cfgs sit in sub-directories,
e.g. `~/wien2k_tests/Rutile/ixchel_110/Rutile_ixchel_110.cfg`.

## Known issues and open items

* **Regenerate `case.vcoul_IAM`.** The graphene `vcoul_IAM` of 2026-03-15 was
  not made by the standard path of the installed `dstart`: its `outputd` has
  no `:NEC01`/renormalisation, the IAM density holds 12.027 instead of 12
  electrons and its vacuum potential varies by 0.38 V. A fresh `x dstart`
  gives a neutral density with a flat vacuum. The lateral ΔV structure hardly
  changes (difference ≤ 0.003 1/Å), the offset does. Check the other
  materials the same way (step 1 `grep`).
* `Graphite_ort_IAM_total.xsf` has negative densities and integrates to 49.6
  instead of 48 electrons; regenerate it before judging the Graphite_ort
  cross-check (the former r ≈ 0.94 open item).
* Handedness of in-plane axes for [010]/[110]: crystruct accepts both. Pass
  the triplet with `--frame` to `vcoul2tiff.py`; `xsf2tiff_dwFactor.py` and
  `vcoul2tiff_ulm.py` use fixed conventions (see their docstrings). For
  mirror-symmetric crystals such as rutile both give identical results.
* `tiff2ixchel.py` keeps a square band below Nyquist, ixchel atompot a circle;
  the difference is ≤ 1e-4 1/Å at Ti columns and is removed anyway by the 2/3
  band limit of `ixchel main`.
* Third route, not implemented: regularise the full DFT potential with the
  thermal displacement Gaussian (−Z·erf(r/σ√2)/r). Future work.
* `docs/` (exposé, presentation, figures) still to be added.

## Reference

S. Kurasch, J. C. Meyer, D. Künzel, A. Groß, U. Kaiser, *Simulation of bonding
effects in HRTEM images of light element materials*, Beilstein J. Nanotechnol.
**2**, 394–404 (2011), doi:[10.3762/bjnano.2.45](https://doi.org/10.3762/bjnano.2.45)
(open access, CC-BY). Their Supporting Information File 2 is a modified
`wien2venus.py` that only drives `lapw5c`; it contains no cutoff code.
