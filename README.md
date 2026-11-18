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

A second route (`src/vcoul2tiff_lapw3.py`, section *Second route* below) builds
the projected potential from the Fourier coefficients of the same potential
file instead: no 3D grid, no cutoff, and the beam along any zone axis whose
projected lattice has a rectangular cell.

## Scripts

In the order of use:

* `src/run_graphene.sh` — the complete list of commands for graphene, from the
  structure file to the DFT-vs-IAM figure (steps 0–7 below).
* `src/check_dstart.py` — checks a `x dstart` run: standard code path, charge
  renormalisation, `case.rsp` consistent with `case.struct`.
* `src/lapw5grid.py` — evaluates `case.vcoul` with lapw5c on a 3D grid and
  writes the values as an XSF data grid. `--edges` and `--origin` choose the grid
  box, e.g. the rectangular cell a₁, a₁ + 2a₂ of a hexagonal lattice:
  `--edges 1 0 0  1 2 0  0 0 1`.
* `src/vcoul2tiff_ulm.py` — steps 2 to 5 of the method: cutoff, shift,
  projection per layer, conversion to 1/Å. Writes one TIFF per layer, each
  holding one unit cell.
* `src/vcoul2tiff_lapw3.py` — the second route: Fourier coefficients of
  `case.vcoul` from WIEN2k's lapw3, summed on the ixchel grid for a chosen zone
  axis; writes the layer TIFF ready for `ixchel main` and a one-layer cfg.
* `src/run_graphene_lapw3.sh` — the complete list of commands for graphene by
  the second route: WIEN2k, the line scan below, the layer files, `ixchel main`.
* `src/plot_kurasch_fig2_lapw3.py` — the paper's Fig. 2 for a crystal: the
  potential on a line from a nucleus, summed from the lapw3 coefficients.
* `src/tiff2ixchel.py` — copies a one-unit-cell layer TIFF NCELL times side by
  side, so that it covers the ixchel `dim` grid. Required before `ixchel main`.
* `src/ixchel_tiff.py` — the TIFF layout of ixchel in one place (page 0 preview,
  page 1 float32 data, page 2 checksums). Imported by the scripts that read or
  write layer files.
* `src/check_transmission.py` — checks ixchel's transmission function
  `trans_<label>.tif` against exp(iσV) of the layer potential.
* `analysis/mip_compare.py` — mean inner potential, DFT vs IAM.
* `analysis/plot_dft_iam_maps.py` — maps of the projected DFT and IAM
  potential, their difference and the integrated changes, in the style of the
  paper's Fig. 6.
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
```

`-numk 200` gives an 11 × 11 × 1 mesh (16 k-points), `-vxc 13` is PBE.

### 2. SCF and the Coulomb potential of the converged density (DFT)

```bash
run_lapw -ec 0.000001 -cc 0.00001
x lapw0
cp Graphene.vcoul Graphene.vcoul_DFT
cp Graphene.clmsum Graphene.clmsum_scf_backup
```

### 3. The same for superposed free atoms (IAM)

`x dstart` overwrites `Graphene.clmsum` with the atomic densities and `x lapw0`
gives their Coulomb potential. The `until` line repeats `x dstart` until the
check passes (see *Pitfalls*). Afterwards the SCF files are put back.

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
the crystal cfg first and `tiff2ixchel.py` copies the unit cell `NCELL` times.
No value is changed: one ixchel pixel is one WIEN2k grid point. The numbers
have to fit for that, points per unit cell × `NCELL` = `dim`; here
72 · 14 = 126 · 8 = 1008 with `NCELL = [14, 8, 1]` and `dim = [1008, 1008]`.
If they do not fit, `ixchel main` stops with "Dimension mismatch".

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

`ixchel main` prints the warning that the layer file "was created with a
different crystal structure config file": the layer files carry no checksum
of the cfg. The calculation is not affected.

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

## Second route: projected potential from Fourier coefficients

`src/vcoul2tiff_lapw3.py` does not sample the potential. WIEN2k's `lapw3`,
which normally Fourier-transforms the electron density (X-ray structure
factors), is run on `case.vcoul`; the file has the same form as a density file.
For a beam along the lattice vector t = [u v w] the projection over one period
|t| contains only the coefficients with G ⟂ t, and their sum on the ixchel grid
is the layer:

```bash
$PY $REPO/src/vcoul2tiff_lapw3.py --struct Graphene.struct --vcoul Graphene.vcoul_DFT \
    --in2 Graphene.in2 --beam 0 0 1 --ncell 14 8 --dim 1008 1008 -o sim_lapw3
cd sim_lapw3 && cp ../sim.cfg . && ixchel main -c crystal.cfg -s sim.cfg
```

The script finds the rectangular image cell itself, runs `lapw3` in a temporary
directory, and writes `atompota.tif` on the `dim` grid together with
`crystal.cfg` (one layer of thickness |t|, with the atoms of the image cell).
It prints the lattice vectors that span the image cell.
`src/run_graphene_lapw3.sh` is the complete list of commands for graphene by
this route, from the structure file to `ixchel main`.

What it covers and what not:

* The beam must be a zone axis, and the lattice projected along it must have a
  rectangular cell; the script stops otherwise.
* One period along the beam is one slice. For a long period that is a thick
  slice; thinner slices are not implemented.
* Lattice types H and P, structures with an inversion centre at the origin.
* The mean value of the layer is relative to lapw0's zero of the potential, not
  to the vacuum level as in the cutoff route. The difference is a constant,
  i.e. a global phase of the wave.
* The layer is the Fourier sum cut at the Nyquist limit of the grid. It rings
  around the nuclei and can be negative there; `ixchel main` applies its own
  band limit to every layer file in the same way.
* `lapw3` of WIEN2k 24.1 is used unchanged, but for a purpose it was not written
  for. The checks below are the evidence that it works on a potential file.
* The scripts run `lapw3` on ONE thread. Its routine `USTPHX` multiplies the
  atom positions by 2π at its first call, and that first call sits inside a
  parallel loop (`fourir.frc`). Several threads can do it at the same time, and
  then every coefficient is wrong. Measured on graphene with 16 threads and the
  full list of reflections: 2 of 5 repeated runs at sin θ/λ ≤ 6 1/Å and 2 of 15
  at ≤ 4 1/Å gave F(000) = −60.6 or −51.5 instead of −9.17 Ry·bohr³; with the
  reflections of one plane only (the layer file) 29 repeated runs were equal.
  On one thread the loop cannot race. A plain `x lapw3` on a density runs the
  same code; 30 repeated runs on `Graphene.clmsum` (≤ 3 1/Å, 16 threads) were
  equal, though.

Checks (the difference DFT − IAM is smooth, so a plain grid sum is an accurate
reference for it):

| check | result |
|---|---|
| graphene [001], DFT − IAM, against the cutoff route (after removing the constant offset) | largest deviation 1.5e-4 1/Å, rms 5e-6 (range 0.19) |
| graphene: coefficients of the layer up to 1/d = 2.9 1/Å against the FFT of the cutoff layer | equal to 3e-4 V·Å once a point deficit of 154 V·Å at each nucleus pixel of the cutoff layer is removed (DFT and IAM alike) |
| graphene: `lapw3` on `vcoul` against the Poisson relation from the density's structure factors | median relative difference 5e-4 |
| graphene: mean inner potential from the G = 0 coefficient, DFT / IAM | 2.5693 / 3.0505 V, change −15.78 % (cutoff route −15.80 %) |
| graphene: `ixchel main` on the written cfg and layer; transmission function vs exp(iσV) | 7e-8 rad |
| graphene: `ixchel atompot` on the written cfg | its atoms sit on the peaks of the layer |
| rutile [001], DFT − IAM, against a `lapw5grid.py` box summed along the beam | largest deviation 3.3e-4 1/Å, rms 2e-5 (range 0.26) |
| rutile [110] (image cell 2.958 × 3.248 Å, sheared box) | largest deviation 1.4e-3 1/Å, rms 5e-5 (range 0.38) |
| rutile [111] (image cell 2.692 × 6.497 Å; no orthogonal box of lattice vectors exists) | largest deviation 1.2e-3 1/Å, rms 1.5e-4 (range 0.41) |

Rutile (P4₂/mnm) has symmetry operations with translations, so these rows also
test the phase factors between equivalent reflections.

### The paper's Fig. 2 for graphene, from the same coefficients

The paper's Fig. 2 is a line scan of the starting potential of a single carbon
atom (section *Figures 2 and 3 of the paper*). `src/plot_kurasch_fig2_lapw3.py`
draws the same figure for the crystal, along a line from a nucleus, with the
WIEN2k curve summed from the lapw3 coefficients of ALL reflections (the layer
file uses only those perpendicular to the beam):

```bash
$PY $REPO/src/plot_kurasch_fig2_lapw3.py --struct Graphene.struct --in2 Graphene.in2 \
    --vcoul Graphene.vcoul_IAM Graphene.vcoul_DFT --direction 0 0 1 -o kurasch_fig2_graphene_lapw3.png
```

A finite Fourier sum cannot give −2Z/r at the nucleus. The script therefore
subtracts the coefficients of point nuclei with Gaussian clouds of charge −Z,
sums the smooth remainder, and adds that reference again in real space. The
comparison curves are Kirkland and Doyle–Turner atoms summed over all atom
positions of the crystal. Result (`reference/kurasch_fig2_graphene_lapw3.png`),
line along c from the nucleus into the vacuum, smallest value set to zero as in
the paper:

| distance [Å] | WIEN2k start (IAM) [V] | WIEN2k SCF [V] | Kirkland atoms [V] | Doyle–Turner atoms [V] |
|---|---|---|---|---|
| 0.004 | 21 206 | 21 202 | 21 347 | 1 589 |
| 0.02 | 3 932 | 3 928 | 3 930 | 1 532 |
| 0.1 | 562.8 | 557.6 | 562.9 | 671.2 |
| 0.2 | 212.9 | 207.6 | 213.5 | 198.7 |
| 0.5 | 41.76 | 38.52 | 42.12 | 41.48 |
| 1.0 | 5.35 | 4.15 | 5.34 | 5.28 |
| 2.0 | 0.196 | 0.072 | 0.183 | 0.132 |

Against a lapw5c line scan of the same file along the same line (before the
shift), the curve differs by at most 3.1 V up to 0.02 Å (of 5 000 to 21 000 V),
1.3 V up to 0.1 Å, 0.10 V up to 0.4 Å, 0.05 V up to 1 Å and 0.006 V beyond;
IAM and DFT file alike. The last number is the ripple of the cut sum, so the
figure ends at 0.01 V (the paper's axis goes to 1e-6 V). Another width of the
Gaussian clouds changes the curve by 1e-9 V.

## Results for graphene

Hexagonal cell P6/mmm, a = 2.461 Å, 15 Å cell height, PBE, RKmax 7,
11 × 11 × 1 k-points, GMAX 16, scalar-relativistic; grid 73 × 127 × 451
(72 × 126 × 450 per period); 80 kV for the ixchel rows. Paper values for
Figs. 2 and 3 are read off the figures.

| check | result | paper |
|---|---|---|
| SCF total energy | −152.41591 Ry | SI File 1: −152.4134 … −152.4152 Ry |
| nucleus grid points replaced by the cutoff, DFT / IAM | 4 / 4 | — |
| range of V_z in the layer, DFT / IAM | 6.75 … 445.71 / 8.66 … 453.45 V·Å | — |
| mean inner potential, DFT / IAM | 2.5647 / 3.0460 V | — |
| mean inner potential change, DFT vs IAM | −15.80 % | 15.5 % |
| change inside the paper's ring (diameter = bond / 2) around the atoms / outside | −6.69 % / −22.48 % | — (Fig. 6 e gives it for the B/O cells only) |
| (V_z,DFT − V_z,IAM) / V_z,IAM over the cell | −30.5 … −1.7 % | 10–30 % |
| `ixchel main` transmission function vs exp(iσV), DFT / IAM | 6e-8 / 8e-8 rad | — |
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
  instead of 12 electrons, sloped vacuum potential). Check every run whose
  density is used directly (the IAM potential, the starting potential of
  Fig. 2) with `check_dstart.py` and repeat `x dstart` until it passes. The
  starting density of an SCF needs no check: from a failed start the graphene
  SCF converges in the same 10 cycles to the same energy within 4e-8 Ry. If
  the check reports `ZZ != Z`, `case.rsp` was made for another RMT: run
  `x lstart` first. The permanent fix would be `tau_mode=.false.` in `init.F`
  (not applied: the WIEN2k tree is left untouched).
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

* **Other beam directions — open.** The route works when three mutually
  perpendicular lattice vectors exist with the beam along the third: they go
  to `lapw5grid.py --edges` and to crystruct. Graphene [001] is such a case;
  no other direction has been run yet. `vcoul2tiff_ulm.py` projects only along
  an edge of the grid box and reads the lengths of the box edges, not their
  directions, so a box that is not rectangular gives wrong layers without a
  warning.
* **No check of the cell.** `vcoul2tiff_ulm.py` takes the lateral cell size
  from the XSF file and does not compare it with the cfg. `lapw5grid.py
  --edges` and the crystruct triplet have to be kept equal by hand.
* **Oblique lattices — open.** `lapw5grid.py` writes grids for hexagonal (H)
  cells and for cells with three right angles (P, B, F, CXY, CXZ, CYZ);
  rhombohedral (R), monoclinic and triclinic cells are not supported. crystruct
  writes an ixchel cfg only for three perpendicular vectors, and an ixchel
  layer is a rectangle (`ax`, `by`). How to treat cells whose projection along
  the beam is not rectangular is not decided.
* **The grids have to fit.** `tiff2ixchel.py` only copies; it does not
  resample. The grid of `lapw5grid.py`, `NCELL` and `dim` must be chosen so
  that the points per unit cell times `NCELL` give `dim` in both directions.
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
