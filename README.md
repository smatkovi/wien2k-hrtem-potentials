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
>
> **Graphene data before 2026-10-13** additionally came from a non-relativistic
> valence calculation (mode line of `Graphene.struct`) and from the
> Lebedev-reduced lapw0; the clean rerun is `~/wien2k_tests/Graphene/lebF/Graphene`
> (see *Known issues*).

## Routes

| script | ΔV from | status |
|---|---|---|
| `src/vcoul2tiff.py` | `case.vcoul_DFT − case.vcoul_IAM`, evaluated by **lapw5** (`RHO SUB`) on a grid over each layer, integrated along the beam | **main route**, validated (see below) |
| `src/xsf2tiff_dwFactor.py` | Δρ = ρ_DFT − ρ_IAM from XSF densities, FFT Poisson solver, line integral | independent cross-check; agrees with the main route (Rutile [001]: r = 1.0000) |
| `src/vcoul2tiff_ulm.py` | no ΔV: total `vcoul` on a grid with the nearest-neighbour cutoff of Kurasch et al. (2011) | faithful re-implementation for comparison; reproduces the paper's graphene numbers and Figs. 2–3 (2026-10-13) |

Both ΔV routes share the cfg parser, the layer depth windows and the Kirkland
term (`kirkland_layer`, a port of ixchel's `atompot` including subslicing and
the band limit).

Tools:

* `src/lapw5grid.py` — evaluates `case.vcoul`, `case.clmsum`, … with lapw5c
  (complex build: required for structures without inversion, identical results
  otherwise) on planes or 3D grids and writes XSF in the 3ddens convention. Use it instead of
  `3ddens` for **CXY/CYZ lattices** (graphene): WIEN2k 24.1 `3ddens` adds the
  centring translations only for F, B, CXZ and R (`SRC_3ddens/3ddens.F:403`) and
  misses the sphere contributions of the centring atoms.
* `src/tiff2ixchel.py` — tiles a one-unit-cell layer TIFF NCELL times and
  resamples it onto the ixchel `dim` grid (exact Fourier interpolation), with
  ixchel's cfg checksum. Required before `ixchel main`.
* `src/check_transmission.py` — checks ixchel's transmission function
  `trans_<label>.tif` against exp(iσV) of the layer potential.
* `src/check_dstart.py` — checks a `x dstart` run (IAM density): standard code
  path, charge renormalisation, `case.rsp` consistent with `case.struct`.
* `src/compare_all_subsliced.py` — compares `wien2k_<hkl>/` with ixchel's
  `ixchel_<hkl>/` layer potentials on an exactly registered unit cell
  (Pearson r, peak ratio, mean offset, plots).
* `src/run_subsliced.sh` — runs `xsf2tiff_dwFactor.py` and the comparison for
  Rutile, Si and Graphite_ort.
* `src/plot_kurasch_fig2.py`, `src/plot_kurasch_fig3.py` — reproductions of
  Figs. 2 and 3 of the paper with the paper's definitions (Fig. 2: lapw5c line
  scan of the starting potential of a single atom at 250 ppÅ; Fig. 3: fine
  lateral line with the cutoff notch, boxes = mean of the green line), plus
  Doyle–Turner and Kirkland references; inputs in `reference/`.

## Requirements

* WIEN2k (`$WIENROOT` on `PATH`: `x`, `lapw0`, `dstart`, `lapw5`, `lapw5c`, `3ddens`)
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

**Two pitfalls of the local WIEN2k build** (details under *Known issues*):
the Lebedev-point reduction of the patched lapw0 is on by default and distorts
the density of sites without local mirror planes (graphene: 6.9 % between
equivalent bonds), so put a file `inlebreduce` containing `F` into the case
directory *before* `run_lapw` (`grep -c LEBREDUCE case.output0` must give 0);
and WIEN2k 24.1 `dstart` picks its code path at random (uninitialised
`tau_mode`), so check every run with `check_dstart.py`.

**Mode line of `case.struct`.** WIEN2k reads the calculation mode from columns
14–17 of line 3. Use `MODE OF CALC=RELA unit=bohr`; a bare `RELA` at the start of
the line is read as blank, and lapw1/lapw2 then treat the valence states
non-relativistically (`case.output2` prints `MODE OF CALCULATION IS = ` without
`RELA`). Check `grep 'MODE OF CALCULATION' case.output2` after the first cycle.

```bash
cd ~/wien2k_tests/Graphene
x lapw0                                   # SCF density -> case.vcoul
cp Graphene.vcoul Graphene.vcoul_DFT

cp Graphene.clmsum Graphene.clmsum_scf_backup
x dstart                                  # overwrites case.clmsum with superposed atoms
$PY <repo>/src/check_dstart.py Graphene   # must print PASS, otherwise run x dstart again
x lapw0
cp Graphene.vcoul Graphene.vcoul_IAM
cp Graphene.clmsum Graphene.clmsum_IAM           # IAM density, for check 5d
cp Graphene.clmsum_scf_backup Graphene.clmsum   # restore the SCF density
cp Graphene.vcoul_DFT Graphene.vcoul            # and the SCF potential
```

If `check_dstart.py` reports `ZZ != Z`, `case.rsp` was made for another RMT:
run `x lstart` first. A zero-byte `vcoul` means `NR2V` was still set.

### 2. crystruct: layer geometry

Create the ixchel crystal cfg with crystruct for the beam direction, e.g.
`crystruct --ixchel "1 0 0  0 1 0  0 0 1" Graphene.struct > Graphene_ixchel_001.cfg`
for [001] (crystruct writes the cfg to stdout). Note the
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
* `--dz` (default 0.05 Å): largest ΔV_z error 2.7e-5 1/Å for graphene
  (1.2e-5 at 0.025 Å, 1.4e-4 at 0.1 Å, against 0.005 Å). `--dx` (0.025 Å)
  leaves < 1e-5 1/Å aliasing. Runtime: about one minute for graphene and half a
  minute for rutile [110] on 16 cores.
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
cd ~/wien2k_tests/Graphene
mkdir -p nulltest && cd nulltest
ln -s ../ixchel_001 ixchel_001                       # ixchel atompot TIFFs + cfg
$PY <repo>/src/vcoul2tiff.py --vcoul-dft ../Graphene.vcoul_IAM --vcoul-iam ../Graphene.vcoul_IAM \
    --struct ../Graphene.struct --cfg ../Graphene_ixchel_001.cfg -o wien2k_001
$PY <repo>/src/compare_all_subsliced.py -p Graphene -d 001
```

Expected: r = 1.0000, ratio = 1.00, mean offset 0.000.

**b) DFT vs IAM comparison and plots.** The same command with the real
`wien2k_001/` writes `Graphene_subslicing_comparison.png` (line profile, both
maps, difference):

```bash
cd ~/wien2k_tests/Graphene                 # holds wien2k_001/ (step 3) and ixchel_001/
$PY <repo>/src/compare_all_subsliced.py -p Graphene -d 001
```

**c) What ixchel actually uses.** Add `writeTrans = 1;` to `sim.cfg`, run
`ixchel main`, then

```bash
cd ~/wien2k_tests/Graphene/sim_dft_001
$PY <repo>/src/check_transmission.py --crystal crystal.cfg --sim sim.cfg \
    --pot atompota.tif --trans trans_a.tif \
    --ref-pot ../sim_iam_001/atompota.tif --ref-trans ../sim_iam_001/trans_a.tif
```

Expected: |t| = 1 and phase deviations ≲ 2e-7 rad (float32).

**d) Independent ΔV cross-check** with the density route. For P, F, B, CXZ
lattices 3ddens densities are fine; for CXY/CYZ generate them with lapw5grid
(same grid for both; `case.clmsum_IAM` is the `case.clmsum` written by
`x dstart` in step 1):

```bash
cd ~/wien2k_tests/Graphene
$PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.clmsum     --norm tot --grid 100 100 150 -o Graphene_DFT_rho.xsf
$PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.clmsum_IAM --norm tot --grid 100 100 150 -o Graphene_IAM_rho.xsf
$PY <repo>/src/xsf2tiff_dwFactor.py --dft Graphene_DFT_rho.xsf --iam Graphene_IAM_rho.xsf \
    --struct Graphene.struct --cfg Graphene_ixchel_001.cfg --miller 0 0 1 -o wien2k_rho_001
```

Compare `wien2k_001` and `wien2k_rho_001` after removing the means (the
Poisson route has zero-mean ΔV by construction).

**e) Kurasch et al. route: both potentials, mean inner potential, Figs. 2 and 3.**
The paper processes the DFT and the IAM potential identically (30 ppÅ grid,
cutoff, normalisation, projection); its graphene check is the change of the
mean inner potential, 15.5 % (p. 400).

```bash
cd ~/wien2k_tests/Graphene/lebF/Graphene   # step 1 done; case.vcoul holds the SCF potential
for k in DFT IAM; do
  $PY <repo>/src/lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul_$k --grid 75 127 451 -o Graphene_vcoul_${k}_ongrid.xsf
  $PY <repo>/src/vcoul2tiff_ulm.py --xsf Graphene_vcoul_${k}_ongrid.xsf --cfg Graphene_ixchel_001.cfg -o ulm_${k}_001
done
$PY -c "import tifffile; d, i = (tifffile.imread(f'ulm_{k}_001/atompota.tif', key=1).mean() for k in ('DFT', 'IAM')); print(f'MIP change {100*(d/i-1):+.2f} %')"
cp <repo>/reference/lapw5_fig3/* .
lapw5c lapw5_fig3_3000.def                 # x-z plane through a nucleus, 3000 ppÅ lateral, reads Graphene.vcoul
$PY <repo>/src/plot_kurasch_fig3.py Graphene.rho_fig3_3000 kurasch_fig3_graphene.png
```

The 75 × 127 × 451 grid puts all four nuclei on voxels; both runs must report
`replaced: 4 voxels`. The TIFFs hold a preview (page 0), the data in 1/Å
(page 1) and metadata (page 2), hence `key=1`. `lapw5_fig3.def` gives the same
plane at 600 ppÅ.

Fig. 2 needs its own case: one C atom in a 10 Å cube and its *starting*
potential (before the first SCF cycle):

```bash
mkdir -p ~/wien2k_tests/C_single && cd ~/wien2k_tests/C_single
cp <repo>/reference/fig2_C_single/C_single.struct . && echo F > inlebreduce
init_lapw -b -rkmax 7.0 -numk 1 -vxc 13    # then case.in0 line 2: NR2V -> R2V
x dstart && $PY <repo>/src/check_dstart.py C_single      # repeat until PASS
x lapw0 && cp C_single.vcoul C_single.vcoul_start
$PY <repo>/src/plot_kurasch_fig2.py --vcoul C_single.vcoul_start --struct C_single.struct -o kurasch_fig2_carbon.png
```

## Validation of the Kurasch route (2026-10-13)

Clean graphene (`~/wien2k_tests/Graphene/lebF/Graphene`: `inlebreduce` = `F`,
scalar-relativistic, RKmax 7, 11 × 11 × 1 k, GMAX 16, PBE, 15 Å cell, neutral
IAM) and the single C atom (`~/wien2k_tests/C_single`). Paper values for Fig. 3
are read off the figure.

| check | result | paper |
|---|---|---|
| SCF total energy | −152.41590 Ry | SI File 1: −152.4134 … −152.4152 Ry |
| density / V_z at vertical vs inclined bond centre | 0.12 % / 1.6e-3 1/Å (IAM: 0.07 % / 0.6e-3 1/Å) | equal (hexagonal) |
| nucleus voxels replaced by the cutoff, DFT / IAM | 4 / 4 | — |
| mean inner potential change, DFT vs IAM | −15.78 % (without the cutoff −10.45 %) | 15.5 % |
| (V_z,DFT − V_z,IAM) / V_z,IAM over the cell | −30.6 … −1.7 % | 10–30 % |
| Fig. 3, centre column without / with the cutoff | 54 781 / 446 V·Å | ≈ 57 000 / ≈ 453 V·Å |
| Fig. 3, mean of the green line, pixel 1 / pixel 2 | 1671 (green sampled at 3000 ppÅ) / 392 V·Å | ≈ 1660 / ≈ 396 V·Å |
| Fig. 3, Doyle–Turner projected at the core | 390 V·Å | ≈ 386 V·Å |
| Fig. 2, starting potential vs Kirkland | ≤ 0.3 % at 0.01–0.2 Å, ≤ 1.2 % up to 1 Å, +4…6 % at 1.5–3 Å (< 1 V) | "very good agreement" |
| Fig. 2, Doyle–Turner at the core | 1589 V | ≈ 1550–1600 V |

## Validation (2026-09-11, graphene CXY and rutile P4₂/mnm)

Graphene rows use the old SCF (Lebedev reduction on, non-relativistic valence).

| check | result |
|---|---|
| lapw5grid vs 3ddens, rutile vcoul 138×138×89 | max 1e-9 Ry (output rounding) |
| lapw5grid vs 3ddens, rutile clmsum 100³ | identical |
| lapw5grid vs 3ddens, graphene | identical except inside the spheres of the centring atoms (3ddens bug; 3.7 kV on the 74×128×450 grid, 1.63 MV where a centring nucleus lies on a voxel) |
| lapw5c vs lapw5 | identical for rutile (centrosymmetric); GaAs (no inversion) needs lapw5c (lapw5 off by up to 1 Ry) |
| null test (Kirkland only) vs `ixchel atompot`, tiled | graphene 7.7e-6, rutile [001]/[100] 4.9e-5, [110] 9.3e-5 1/Å |
| ΔV main route vs Poisson route, rutile [001] (3ddens densities) | r = 1.0000, rms difference 0.4 % of the ΔV rms |
| ΔV main route vs Poisson route, graphene (lapw5grid densities) | r = 1.0000, rms difference 0.2 % |
| graphene ΔV symmetry (mirrors / centring shift) | 2e-7 / 1e-4 1/Å |
| graphene 3-fold symmetry (vertical vs inclined bond centres) | broken by 5.4e-3 1/Å — in the SCF input, not in the scripts (see Known issues); 1.6e-3 1/Å after the rerun |
| ΔV convergence against dz = 5 mÅ: dz 50 / 25 / 100 mÅ; 25 mÅ lateral aliasing | 2.7e-5 / 1.2e-5 / 1.4e-4 1/Å; 6e-6 1/Å |
| `ixchel main` transmission vs exp(iσV) | 3.5e-8 rad |
| graphene mean inner potential, DFT vs Kirkland (neutral IAM, vacuum-referenced) | −15.8 % (paper: 15.5 %) |
| `x lapw0` recipe reproduces `Graphene.vcoul_DFT` | 1e-15 |

## Test data (not in this repository)

WIEN2k cases live in `~/wien2k_tests/<Material>/`: `Si` (F), `Rutile` (P;
cfgs for 001/100/110), `Graphite_ort` (P), `Graphene` (CXY), plus `FeF2`,
`Graphite_conv`, `Si/Si_conv`. They hold multi-MB `vcoul`, XSF and TIFF files;
none of that is committed (`.gitignore`). ixchel cfgs sit in sub-directories,
e.g. `~/wien2k_tests/Rutile/ixchel_110/Rutile_ixchel_110.cfg`.

The clean graphene rerun is `~/wien2k_tests/Graphene/lebF/Graphene` (the case
directory must carry the case name), with the commands and results in
`~/wien2k_tests/Graphene/lebF/RUN.md`; the earlier non-relativistic results
are kept there as `*_nrel`. The single C atom of Fig. 2 is
`~/wien2k_tests/C_single`.

## Known issues and open items

* **WIEN2k 24.1 `dstart` is non-deterministic.** `tau_mode` is declared in
  `SRC_dstart/dstart.F` but never initialised (`init.F` only ever sets it to
  `.true.`), so a run can take the tau/potential path by accident: no density
  smoothing, no renormalisation, a charged IAM density. The graphene
  `vcoul_IAM` of 2026-03-15 is such a run (12.027 instead of 12 electrons,
  sloped vacuum potential). Check every run with `check_dstart.py` and
  regenerate the existing `vcoul_IAM` files. The lateral ΔV structure hardly
  changes (≤ 0.003 1/Å); the offset and the mean inner potential do. The
  permanent fix is `tau_mode=.false.` in `init.F` (not applied: the WIEN2k tree
  is left untouched). The graphene IAM was regenerated on 2026-10-13 in
  `Graphene/lebF/Graphene` (2 of 4 `x dstart` runs failed the check).
* **Lebedev-point reduction in the patched lapw0** (local commits `0c8e4d9`,
  `e2356a4`, on by default): `SRC_lapw0/xcpot1.f` keeps only the Lebedev
  points with local y ≥ 0 and z ≥ 0 for every non-cubic site, i.e. it assumes
  local mirror planes. The graphene C site (Cmmm, odd-l terms in its LM list)
  has no mirror along local z. The SCF density at the vertical and inclined
  bond centres then differs by 6.9 % (ΔV_z by 5.4e-3 1/Å). An SCF rerun with
  `inlebreduce` = `F` in the review reduced this to 0.12 % and changed the
  total energy by 42 mRy; a hexagonal k-mesh alone changed nothing. The
  reduction has been on by default since 2026-01-27 (`e2356a4`), so every SCF
  run with this lapw0 after that date is suspect (Rutile, Si, FeF2 on
  2026-02-14; Graphite_ort and Graphene on 2026-02-22); the later lapw0 runs
  of Rutile and Graphite_ort also print it in `output0`. Whether a site is
  affected depends on its site symmetry. Rerun the SCFs with
  `inlebreduce` = `F`, or restrict the reduction to sites that have those
  mirrors. Done for graphene on 2026-10-13 (`Graphene/lebF/Graphene`, density
  difference 0.12 %); Rutile, Si, FeF2 and Graphite_ort are still to be rerun.
* **Mode line of `Graphene.struct`.** Line 3 read `RELA` in columns 1–4, but
  WIEN2k reads the mode from columns 14–17 (`(13X,A4)`), so lapw1 and lapw2
  treated the valence states of every graphene SCF before 2026-10-13
  non-relativistically, including the −15.8 % of the 2026-09-11 validation
  (lstart treats a blank mode as relativistic, lcore ignores the mode). Fixed in
  `Graphene/lebF/Graphene` (`MODE OF CALC=RELA unit=bohr`): total energy
  −152.40946 → −152.41590 Ry, mean inner potential change −15.76 % → −15.78 %,
  Fig. 3 numbers unchanged. `Si/Si.struct` has `RELA` after 13 spaces and is
  read correctly; the other test structs use the standard line.
* **Open: residual hexagonal asymmetry of the graphene SCF.** After the rerun
  V_z still differs by 1.6e-3 1/Å between the vertical and inclined bond
  centres, about 3× the IAM value (0.6e-3 1/Å). Not explained yet; a
  hexagonal k-mesh instead of the rectangular 11 × 11 × 1 mesh of the CXY cell
  was only tried with the Lebedev reduction on.
* **Graphite_ort is invalid at the struct level.** Symmetry operations 5–8 of
  `Graphite_ort.struct` map four atoms onto empty sites (the z+1/2 translation
  is missing), and its RMT 1.34 does not match `Graphite_ort.rsp` (dstart's ZZ
  is 6.144 instead of 6). The SCF density, `vcoul_DFT`, every IAM and
  `Graphite_ort_IAM_total.xsf` (negative densities, 49.6 e) inherit this.
  Regenerate the struct (`x sgroup`, `x symmetry`), run `x lstart` and the SCF
  again; the former r ≈ 0.94 open item cannot be judged before that.
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
