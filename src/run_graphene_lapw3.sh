#!/bin/bash
# run_graphene_lapw3.sh — graphene by the second route (Fourier coefficients from lapw3):
# from the structure file to kurasch_fig2_graphene_lapw3.png and to the ixchel run
#
# The plain list of commands, in order.  Steps 0 to 3 are those of run_graphene.sh.
# Run it inside a directory named Graphene (WIEN2k takes the case name from the directory name):
#     mkdir Graphene && cd Graphene && bash ~/wien2k-hrtem-potentials/src/run_graphene_lapw3.sh
set -e
PY=~/venvs/hrtem/bin/python
REPO=~/wien2k-hrtem-potentials
export OMP_NUM_THREADS=4

# 0 (optional, only without Graphene.struct): the structure file from WIEN2k's generator.
#   Answers: title, S = space group, 191 = P6/mmm, b = bohr, a b c, angles (default),
#   1 atom, element, position, RMT reduction 0 %, no second setrmt run.
if [ ! -f Graphene.struct ]; then
makestruct_lapw > makestruct.log <<END
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
END
mv init.struct Graphene.struct
rm -f init.* nn.def
fi

# 1: WIEN2k input files and starting density
echo F > inlebreduce                                   # local lapw0 build: no Lebedev-point reduction
init_lapw -b -rkmax 7.0 -numk 200 -vxc 13
sed -i '2s/^NR2V/R2V /' Graphene.in0                   # lapw0 shall write Graphene.vcoul

# 2: SCF, then the Coulomb potential of the converged density (DFT)
run_lapw -ec 0.000001 -cc 0.00001
x lapw0
cp Graphene.vcoul Graphene.vcoul_DFT
cp Graphene.clmsum Graphene.clmsum_scf_backup

# 3: the same for superposed free atoms (IAM), then put the SCF files back.
#    x dstart of WIEN2k 24.1 fails at random; it is repeated until the check passes.
x dstart
until $PY $REPO/src/check_dstart.py Graphene; do x dstart; done
x lapw0
cp Graphene.vcoul Graphene.vcoul_IAM
cp Graphene.clmsum_scf_backup Graphene.clmsum
cp Graphene.vcoul_DFT Graphene.vcoul

# 4: the paper's Fig. 2 for graphene: the potential on a line from a nucleus along c.
#    The script runs lapw3 on each potential file (all reflections up to sin(theta)/lambda = 6 1/Å).
$PY $REPO/src/plot_kurasch_fig2_lapw3.py --struct Graphene.struct --in2 Graphene.in2 \
    --vcoul Graphene.vcoul_IAM Graphene.vcoul_DFT --direction 0 0 1 -o kurasch_fig2_graphene_lapw3.png

# 5: the layer file for ixchel, beam along c (DFT and IAM).
#    The script runs lapw3 again, now only for the reflections perpendicular to the beam,
#    and writes atompota.tif and crystal.cfg.
for K in DFT IAM; do
    $PY $REPO/src/vcoul2tiff_lapw3.py --struct Graphene.struct --vcoul Graphene.vcoul_$K \
        --in2 Graphene.in2 --beam 0 0 1 --ncell 14 8 --dim 1008 1008 -o sim_lapw3_$K
done

# 6: ixchel on these layer files (not ixchel atompot: it would overwrite atompota.tif)
cat > sim.cfg <<END
dim = [ 1008, 1008];
v0 = 80.;
cmode = 0;
inelastic_scattering = 0;
writeTrans = 1;
END
for K in DFT IAM; do
    cp sim.cfg sim_lapw3_$K/
    cd sim_lapw3_$K
    ixchel main -c crystal.cfg -s sim.cfg
    cd ..
done
