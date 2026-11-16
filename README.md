# wien2k-hrtem-potentials

Ab-initio scattering potentials from [WIEN2k](http://www.wien2k.at) for HRTEM
multislice simulations with **ixchel**, instead of the usual Independent Atom
Model (IAM).

Master's thesis project, TU Wien (Technical Physics), supervised by
Assoc. Prof. Stefan Löffler (USTEM).

## What this does

A multislice code needs, for every slice, the projected electrostatic
potential V_z(x, y) = ∫ V(x, y, z) dz. ixchel builds it from Kirkland's
parametrised isolated-atom potentials. Here it is taken from a self-consistent
WIEN2k calculation instead, with the method of Kurasch et al. (2011):

1. evaluate the Coulomb potential of the SCF density (`case.vcoul`) on a
   regular 3D grid of about 30 points per Å;
2. replace the grid values at the nuclei, where the potential diverges, by the
   value of a neighbouring grid point (nearest-neighbour cutoff, −50 kV);
3. shift the potential so that its smallest value (the vacuum level) is zero;
4. sum the grid planes of every layer along the beam direction;
5. hand the layers to ixchel as TIFF files in its own unit (1/Å).

The Coulomb potential of the superposed free atoms (`x dstart`) goes through
exactly the same steps and gives the IAM for comparison.

## Scripts

In the order of use:

* `src/run_graphene.sh` — the complete list of commands for graphene, from the
  structure file to the DFT-vs-IAM figure (steps 0–7 below).
* `src/check_dstart.py` — checks a `x dstart` run: standard code path, charge
  renormalisation, `case.rsp` consistent with `case.struct`.
* `src/lapw5grid.py` — evaluates `case.vcoul` (or `case.clmsum`, …) with lapw5c
  on a 3D grid and writes an XSF file. `--edges` and `--origin` choose the grid
  box, e.g. the rectangular cell a₁, a₁ + 2a₂ of a hexagonal lattice:
  `--edges 1 0 0  1 2 0  0 0 1`.
* `src/vcoul2tiff_ulm.py` — steps 2 to 5 of the method: cutoff, shift,
  projection per layer, conversion to 1/Å. Writes one TIFF per layer, each
  holding one unit cell.
* `src/tiff2ixchel.py` — repeats a one-unit-cell layer TIFF NCELL times and
  resamples it onto the ixchel `dim` grid (Fourier interpolation), with
  ixchel's cfg checksum. Required before `ixchel main`.
* `src/ixchel_tiff.py` — the TIFF layout of ixchel in one place (page 0 preview,
  page 1 float32 data, page 2 checksums). Imported by the scripts that read or
  write layer files.
* `src/check_transmission.py` — checks ixchel's transmission function
  `trans_<label>.tif` against exp(iσV) of the layer potential.
* `analysis/mip_compare.py` — mean inner potential, DFT vs IAM, with and
  without the cutoff; flatness of the vacuum potential.
* `analysis/plot_dft_iam_maps.py` — maps of the projected DFT and IAM
  potential, their difference and the integrated changes, in the style of the
  paper's Fig. 6.
* `analysis/sym_check.py` — density and projected potential at two bond
  centres (three-fold symmetry of the SCF).
* `src/plot_kurasch_fig2.py`, `src/plot_kurasch_fig3.py` — Figs. 2 and 3 of the
  paper with the paper's definitions, plus Doyle–Turner and Kirkland curves.

Every script explains its arguments, and where each input file comes from, in
the comment block at its top.

## Requirements

* WIEN2k 24.1 with `$WIENROOT` on `PATH` (`makestruct_lapw`, `init_lapw`,
  `run_lapw`, `x`, `lapw0`, `dstart`, `lapw5c`); `makestruct_lapw` needs tcsh
  and bc
* crystruct and ixchel (`ixchel main`, `ixchel image`)
* Python ≥ 3.10 with numpy, scipy, tifffile, matplotlib

```bash
python3 -m venv ~/venvs/hrtem
~/venvs/hrtem/bin/pip install numpy scipy tifffile matplotlib
```

## Workflow

The example is graphene. Steps 0 to 7 are the commands of
`src/run_graphene.sh` (about two minutes on 16 cores):

```bash
mkdir Graphene && cd Graphene && bash ~/wien2k-hrtem-potentials/src/run_graphene.sh
```

WIEN2k takes the case name, and so all file names, from the name of the
directory. In the commands below

```bash
PY=~/venvs/hrtem/bin/python
REPO=~/wien2k-hrtem-potentials
```

### 0. Structure file (optional)

`makestruct_lapw` is WIEN2k's structure generator. It asks its questions on
the terminal; here the answers come from the lines between `<<EOF` and `EOF`:
title, `S` = give a space group, 191 = P6/mmm, `b` = bohr, a b c, the default
angles, one atom, its element, its position, sphere radii not reduced, no
second `setrmt` run.

```bash
makestruct_lapw > makestruct.log <<EOF
Graphene
S
191
b
4.650616 4.650616 28.345892

1
C
1/3 2/3 0
0
N
EOF
mv init.struct Graphene.struct
rm -f init.* nn.def
```

The result is the hexagonal cell with a = 2.461 Å and c = 15 Å (layer plus
vacuum), C at (1/3, 2/3, 0) and (2/3, 1/3, 0), RMT = 1.33 bohr. The list of
symmetry operations is still empty; `init_lapw` fills it in step 1.

`reference/graphene_hex/Graphene.struct` is the structure file of the reference
case, written by `x sgroup`: the same cell, atoms and 24 symmetry operations.
The two files differ in labels, in the order of the operations and in the
eighth decimal of some translations; total energy and figure are the same.

### 1. WIEN2k input files and starting density

```bash
echo F > inlebreduce                                   # local lapw0 build: no Lebedev-point reduction
init_lapw -b -rkmax 7.0 -numk 200 -vxc 13
sed -i '2s/^NR2V/R2V /' Graphene.in0                   # lapw0 shall write Graphene.vcoul
until $PY $REPO/src/check_dstart.py Graphene; do x dstart; done
```

`-numk 200` gives an 11 × 11 × 1 mesh (16 k-points), `-vxc 13` is PBE. The
`until` line repeats `x dstart` until the check passes (see *Pitfalls*).

### 2. SCF and the Coulomb potential of the converged density (DFT)

```bash
run_lapw -ec 0.000001 -cc 0.00001
x lapw0
cp Graphene.vcoul Graphene.vcoul_DFT
cp Graphene.clmsum Graphene.clmsum_scf_backup
```

### 3. The same for superposed free atoms (IAM)

`x dstart` overwrites `Graphene.clmsum` with the atomic densities and `x lapw0`
gives their Coulomb potential. Afterwards the SCF files are put back.

```bash
x dstart
until $PY $REPO/src/check_dstart.py Graphene; do x dstart; done
x lapw0
cp Graphene.vcoul Graphene.vcoul_IAM
cp Graphene.clmsum_scf_backup Graphene.clmsum
cp Graphene.vcoul_DFT Graphene.vcoul
```

### 4. crystruct: layer geometry

ixchel needs a rectangular cell. For the hexagonal lattice it is spanned by
a₁, a₁ + 2a₂ and c (2.461 × 4.263 × 15 Å, four atoms):

```bash
crystruct --ixchel "1 0 0  1 2 0  0 0 1" Graphene.struct > Graphene_ixchel_001.cfg
```

### 5 and 6. Potential on a 3D grid; cutoff, shift and projection

Once for DFT and once for IAM. `--edges` takes the same three vectors as
crystruct, so that the grid box is the cell of the cfg.

```bash
for K in DFT IAM; do
    $PY $REPO/src/lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul_$K \
        --grid 73 127 451 --edges 1 0 0  1 2 0  0 0 1 -o Graphene_vcoul_${K}_ongrid.xsf
    $PY $REPO/src/vcoul2tiff_ulm.py --xsf Graphene_vcoul_${K}_ongrid.xsf \
        --cfg Graphene_ixchel_001.cfg -o ulm_${K}_001
done
```

`--grid` counts both end points of every edge, so one period has 72 × 126 × 450
points: about 30 points per Å, all four nuclei on grid points, and only small
prime factors. Both `vcoul2tiff_ulm.py` runs must report `replaced: 4 voxels`.
Output: `ulm_DFT_001/atompota.tif` and `ulm_IAM_001/atompota.tif`, one unit
cell each, in 1/Å.

### 7. Numbers and figure

```bash
$PY $REPO/analysis/mip_compare.py .
$PY $REPO/analysis/plot_dft_iam_maps.py . graphene_dft_iam_maps.png
```

### 8. ixchel

Not part of `run_graphene.sh`. `ixchel main` expects every layer file to cover
the whole supercell on exactly `dim` pixels, so `dim` and `NCELL` are set in
the crystal cfg first and `tiff2ixchel.py` brings the layer onto that grid.
With `NCELL = [14, 8, 1]` and `dim = [1008, 1008]` one ixchel pixel is one
WIEN2k grid point (72 · 14 = 126 · 8 = 1008).

```bash
cat > sim.cfg <<EOF
dim = [ 1008, 1008];
v0 = 80.;
cmode = 0;
inelastic_scattering = 0;
writeTrans = 1;
EOF
for K in DFT IAM; do
    mkdir -p sim_$K && cd sim_$K
    sed -e 's/^dim = .*/dim = [ 1008, 1008];/' -e 's/^NCELL = .*/NCELL = [ 14, 8, 1];/' \
        ../Graphene_ixchel_001.cfg > crystal.cfg
    cp ../sim.cfg sim.cfg
    $PY $REPO/src/tiff2ixchel.py --cfg crystal.cfg -i ../ulm_${K}_001 -o .
    ixchel main -c crystal.cfg -s sim.cfg
    cd ..
done
```

Do **not** run `ixchel atompot` in these directories: it would overwrite the
WIEN2k layer files with ixchel's own atomic potentials.

`writeTrans = 1` makes `ixchel main` write its transmission function, which
shows what ixchel made of the layer file:

```bash
cd sim_DFT
$PY $REPO/src/check_transmission.py --crystal crystal.cfg --sim sim.cfg \
    --pot atompota.tif --trans trans_a.tif \
    --ref-pot ../sim_IAM/atompota.tif --ref-trans ../sim_IAM/trans_a.tif
```

The image is then computed with `ixchel image -c crystal.cfg -s sim.cfg`. The
`sim.cfg` above holds no imaging parameters, and that step is not covered by
the results below.

### Figures 2 and 3 of the paper

Fig. 3 is a plane through a nucleus, evaluated by lapw5c directly (in the
graphene case directory; the plane definitions are in
`reference/lapw5_fig3_hex/`):

```bash
cp $REPO/reference/lapw5_fig3_hex/* .
lapw5c lapw5_fig3_3000.def                 # x-z plane through a nucleus, 3000 points per Å laterally
$PY $REPO/src/plot_kurasch_fig3.py Graphene.rho_fig3_3000 kurasch_fig3_graphene.png
```

Fig. 2 needs its own case: one C atom in a 10 Å cube and its *starting*
potential, before the first SCF cycle:

```bash
mkdir -p ~/wien2k_tests/C_single && cd ~/wien2k_tests/C_single
cp $REPO/reference/fig2_C_single/C_single.struct . && echo F > inlebreduce
init_lapw -b -rkmax 7.0 -numk 1 -vxc 13
sed -i '2s/^NR2V/R2V /' C_single.in0
until $PY $REPO/src/check_dstart.py C_single; do x dstart; done
x lapw0 && cp C_single.vcoul C_single.vcoul_start
$PY $REPO/src/plot_kurasch_fig2.py --vcoul C_single.vcoul_start --struct C_single.struct -o kurasch_fig2_carbon.png
```

### Symmetry check

The two bond centres (1/2, 0) and (1/2, 1/2) of the hexagonal cell are
symmetry equivalent, so density and projected potential must agree there:

```bash
$PY $REPO/analysis/sym_check.py . Graphene.clmsum Graphene.vcoul_DFT --p1 1/2 0 --p2 1/2 1/2
```

## Results for graphene

Hexagonal cell P6/mmm, a = 2.461 Å, 15 Å cell height, PBE, RKmax 7,
11 × 11 × 1 k-points, GMAX 16, scalar-relativistic; grid 73 × 127 × 451
(72 × 126 × 450 per period); 80 kV for the ixchel rows. Paper values for
Figs. 2 and 3 are read off the figures.

| check | result | paper |
|---|---|---|
| SCF total energy | −152.41591 Ry | SI File 1: −152.4134 … −152.4152 Ry |
| density / V_z at the two bond centres | equal (symmetry equivalent in P6/mmm) | equal (hexagonal) |
| vacuum: plane-averaged potential at z = 7.5 Å and 1 Å above and below, DFT / IAM | constant to 1e-5 V / 1e-5 V | — |
| nucleus grid points replaced by the cutoff, DFT / IAM | 4 / 4 | — |
| range of V_z in the layer, DFT / IAM | 6.75 … 445.71 / 8.66 … 453.45 V·Å | — |
| mean inner potential, DFT / IAM | 2.5647 / 3.0460 V | — |
| mean inner potential change, DFT vs IAM | −15.80 % (without the cutoff −10.36 %) | 15.5 % |
| change inside the paper's ring (diameter = bond / 2) around the atoms / outside | −6.69 % / −22.48 % | — (Fig. 6 e gives it for the B/O cells only) |
| (V_z,DFT − V_z,IAM) / V_z,IAM over the cell | −30.5 … −1.7 % | 10–30 % |
| `ixchel main` transmission function vs exp(iσV), DFT / IAM | 6e-8 / 6e-8 rad | — |
| phase of the layer, DFT − IAM | −0.0111 … −0.0019 rad | — |
| Fig. 3, centre column without / with the cutoff | 54 781 / 446 V·Å | ≈ 57 000 / ≈ 453 V·Å |
| Fig. 3, mean of the green line, pixel 1 / pixel 2 | 1671 (green sampled at 3000 points per Å) / 392 V·Å | ≈ 1660 / ≈ 396 V·Å |
| Fig. 3, Doyle–Turner projected at the core | 390 V·Å | ≈ 386 V·Å |
| Fig. 2, starting potential vs Kirkland | ≤ 0.3 % at 0.01–0.2 Å, ≤ 1.2 % up to 1 Å, +4…6 % at 1.5–3 Å (< 1 V) | "very good agreement" |
| Fig. 2, Doyle–Turner at the core | 1589 V | ≈ 1550–1600 V |

`run_graphene.sh` in an empty directory reproduces the total energy, the
−15.80 % and, byte for byte, the figure of the reference run.

Checks of the tools:

| check | result |
|---|---|
| `lapw5grid.py` vs WIEN2k's 3ddens, rutile (P lattice), `vcoul` on 138 × 138 × 89 | max 1e-9 Ry (output rounding) |
| the same for `clmsum` on 100³ | identical |
| `lapw5grid.py` vs 3ddens, graphene described as a CXY lattice | identical except inside the spheres of the centring atoms, which 3ddens leaves out |
| lapw5c vs lapw5 | identical for rutile (centrosymmetric); GaAs (no inversion) needs lapw5c (lapw5 off by up to 1 Ry) |
| `x lapw0` run again on the SCF density vs `case.vcoul_DFT` | 1e-15 |

## Pitfalls

* **WIEN2k 24.1 `dstart` is non-deterministic.** `tau_mode` is declared in
  `SRC_dstart/dstart.F` but never initialised (`init.F` only ever sets it to
  `.true.`), so a run can take the tau/potential path by accident: no density
  smoothing, no renormalisation, a charged IAM density (graphene: 12.027
  instead of 12 electrons, sloped vacuum potential). Check every run with
  `check_dstart.py` and repeat `x dstart` until it passes. If the check reports
  `ZZ != Z`, `case.rsp` was made for another RMT: run `x lstart` first. The
  permanent fix would be `tau_mode=.false.` in `init.F` (not applied: the
  WIEN2k tree is left untouched).
* **Lebedev-point reduction of the local lapw0 build.** It is on by default
  and assumes local mirror planes at every non-cubic site, which distorts the
  density of sites without them. Put a file `inlebreduce` containing `F` into
  the case directory *before* `run_lapw`; `grep -c LEBREDUCE case.output0`
  must give 0. Any SCF of this build without that file is suspect.
* **Mode line of `case.struct`.** WIEN2k reads the calculation mode from
  columns 14–17 of line 3. The line must read `MODE OF CALC=RELA unit=bohr`; a
  bare `RELA` at the start of the line is read as blank, and lapw1/lapw2 then
  treat the valence states non-relativistically. Check
  `grep 'MODE OF CALCULATION' case.output2` after the first cycle.
* **`case.in0`.** Line 2 must read `R2V`, not `NR2V`, and must not contain
  `NONORM` (lapw5 needs the standard r² normalisation). A zero-byte
  `case.vcoul` means `NR2V` was still set.
* **Graphene has to be a hexagonal (H) lattice.** `init_lapw` rejects the
  rectangular CXY description with the exact ratio b = a·√3 (`x symmetry`
  finds 24 operations instead of 8). The rectangular cell that ixchel needs is
  therefore sampled from the hexagonal case, with `--edges 1 0 0  1 2 0  0 0 1`
  in `lapw5grid.py` and the same triplet in crystruct.
* **3ddens is not used.** WIEN2k 24.1 `3ddens` adds the centring translations
  only for F, B, CXZ and R lattices (`SRC_3ddens/3ddens.F:403`), so for CXY and
  CYZ lattices the sphere contributions of the centring atoms are missing.
  `lapw5grid.py` drives lapw5c instead, which has no such restriction.

## Limits and open items

* **Beam directions.** `vcoul2tiff_ulm.py` projects only along an edge of the
  grid box ([001], [100] or [010] of the box) and takes the box as
  rectangular: it reads the lengths of the box edges, not their directions.
* **No check of the cell.** `vcoul2tiff_ulm.py` takes the lateral cell size
  from the XSF file and does not compare it with the cfg. `lapw5grid.py
  --edges` and the crystruct triplet have to be kept equal by hand.
* **Lattice types.** `lapw5grid.py` writes grids for hexagonal (H) cells and
  for cells with three right angles (P, B, F, CXY, CXZ, CYZ); rhombohedral (R),
  monoclinic and triclinic cells are not supported.
* **Nyquist term.** `tiff2ixchel.py` keeps only frequencies strictly below the
  Nyquist limit, so the term at the limit is dropped even when the cell fits
  the ixchel grid exactly (graphene: 2e-4 % of the spectral power). It lies
  above the 2/3 band limit that `ixchel main` applies anyway.
* **Paper values of Fig. 3** in the table above are read off the figure by
  eye.
* **Other systems of the paper** (B and O substitution, O adatom) are not
  computed yet.

## Test data (not in this repository)

WIEN2k cases hold multi-MB `vcoul`, XSF and TIFF files; none of that is
committed (`.gitignore`).

* `~/wien2k_tests/Graphene/hex/Graphene` — the graphene reference case (SCF and
  both potentials), with its run log `~/wien2k_tests/Graphene/hex/RUN.md`.
* `~/wien2k_tests/Graphene/hex/grid72` — grids, layer files, figure and ixchel
  runs of the table above (`run.sh`, `run_ixchel.sh`).
* `~/wien2k_tests/C_single` — the single C atom of Fig. 2.

In `reference/`: the structure files of both cases, the lapw5 inputs of
Fig. 3, the two reproduced figures and `kurasch2011_review.md`, notes on what
the paper specifies and how the scripts read it.

## Reference

S. Kurasch, J. C. Meyer, D. Künzel, A. Groß, U. Kaiser, *Simulation of bonding
effects in HRTEM images of light element materials*, Beilstein J. Nanotechnol.
**2**, 394–404 (2011), doi:[10.3762/bjnano.2.45](https://doi.org/10.3762/bjnano.2.45)
(open access, CC-BY). Their Supporting Information File 2 is a modified
`wien2venus.py` that only drives `lapw5c`; it contains no cutoff code.
