#!/bin/bash
# run_graphene.sh — graphene from the structure file to graphene_dft_iam_maps.png
#
# The plain list of commands, in order.  Run it inside a directory named Graphene
# (WIEN2k takes the case name from the directory name):
#     mkdir Graphene && cd Graphene && bash ~/wien2k-hrtem-potentials/src/run_graphene.sh
set -e
PY=~/venvs/hrtem/bin/python
REPO=~/wien2k-hrtem-potentials
export OMP_NUM_THREADS=4

# 0 (optional, only without Graphene.struct): the structure file from WIEN2k's generator.
#   Answers: title, S = space group, 191 = P6/mmm, b = bohr, a b c, angles (default),
#   1 atom, element, position, RMT reduction 0 %, no second setrmt run.
if [ ! -f Graphene.struct ]; then
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
fi

# 1: WIEN2k input files and starting density
echo F > inlebreduce                                   # local lapw0 build: no Lebedev-point reduction
init_lapw -b -rkmax 7.0 -numk 200 -vxc 13
sed -i '2s/^NR2V/R2V /' Graphene.in0                   # lapw0 shall write Graphene.vcoul
until $PY $REPO/src/check_dstart.py Graphene; do x dstart; done

# 2: SCF, then the Coulomb potential of the converged density (DFT)
run_lapw -ec 0.000001 -cc 0.00001
x lapw0
cp Graphene.vcoul Graphene.vcoul_DFT
cp Graphene.clmsum Graphene.clmsum_scf_backup

# 3: the same for superposed free atoms (IAM), then put the SCF files back
x dstart
until $PY $REPO/src/check_dstart.py Graphene; do x dstart; done
x lapw0
cp Graphene.vcoul Graphene.vcoul_IAM
cp Graphene.clmsum_scf_backup Graphene.clmsum
cp Graphene.vcoul_DFT Graphene.vcoul

# 4: layer geometry of the rectangular cell a1, a1 + 2 a2, c
crystruct --ixchel "1 0 0  1 2 0  0 0 1" Graphene.struct > Graphene_ixchel_001.cfg

# 5 and 6: potential on a 3D grid, then cutoff, shift and projection (DFT and IAM)
for K in DFT IAM; do
    $PY $REPO/src/lapw5grid.py --struct Graphene.struct --clm Graphene.vcoul_$K \
        --grid 73 127 451 --edges 1 0 0  1 2 0  0 0 1 -o Graphene_vcoul_${K}_ongrid.xsf
    $PY $REPO/src/vcoul2tiff_ulm.py --xsf Graphene_vcoul_${K}_ongrid.xsf \
        --cfg Graphene_ixchel_001.cfg -o ulm_${K}_001
done

# 7: the numbers and the figure
$PY $REPO/analysis/mip_compare.py .
$PY $REPO/analysis/plot_dft_iam_maps.py . graphene_dft_iam_maps.png
