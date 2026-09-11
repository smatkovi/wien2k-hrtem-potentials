#!/bin/bash
# Regenerates all WIEN2k potentials with subslicing and runs the comparisons.
# Usage: cd ~/wien2k_tests && bash <repo>/src/run_subsliced.sh

set -e
SRC_DIR=$(dirname "$(readlink -f "$0")")
XSF2TIFF=$SRC_DIR/xsf2tiff_dwFactor.py
COMPARE=$SRC_DIR/compare_all_subsliced.py

echo "════════════════════════════════════════════════════════════════════════"
echo "  xsf2tiff_dwFactor.py with subslicing — all materials"
echo "════════════════════════════════════════════════════════════════════════"

# ──────────────────────────────────────────────────────────────────────────
# RUTILE (P lattice, orthogonal)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ RUTILE ═══"
cd ~/wien2k_tests/Rutile

for DIR in 001 100 110; do
    HKL=$(echo $DIR | sed 's/\(.\)/\1 /g')
    CFG=Rutile_ixchel_${DIR}.cfg

    if [ ! -f "$CFG" ]; then
        # cfg may be located in the ixchel folder
        if [ -f "ixchel_${DIR}/${CFG}" ]; then
            CFG="ixchel_${DIR}/${CFG}"
        elif [ -f "ixchel_${DIR}/ixchel.cfg" ]; then
            CFG="ixchel_${DIR}/ixchel.cfg"
        else
            echo "WARNING: No cfg found for Rutile [$DIR], skipping."
            continue
        fi
    fi

    echo ""
    echo "--- Rutile [$DIR] ---"
    python3 $XSF2TIFF \
        --dft Rutile_DFT_total.xsf \
        --iam Rutile_IAM_total.xsf \
        --struct Rutile.struct \
        --fparams fparams.dat \
        --cfg "$CFG" \
        --miller $HKL \
        -o wien2k_${DIR}
done

echo ""
echo "--- Rutile comparison ---"
python3 $COMPARE -p Rutile -d 001 100 110

# ──────────────────────────────────────────────────────────────────────────
# Si (F→P supercell, orthogonal)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ Si ═══"
cd ~/wien2k_tests/Si

# Si needs DFT+IAM XSFs — check whether they exist
if [ -f "Si_DFT_total.xsf" ] && [ -f "Si_IAM_total.xsf" ]; then
    FPARAMS=fparams.dat
    [ ! -f "$FPARAMS" ] && FPARAMS=../Rutile/fparams.dat

    CFG_001=Si_ixchel_001.cfg
    [ ! -f "$CFG_001" ] && CFG_001=ixchel_001/Si_ixchel_001.cfg

    echo ""
    echo "--- Si [001] ---"
    python3 $XSF2TIFF \
        --dft Si_DFT_total.xsf \
        --iam Si_IAM_total.xsf \
        --struct Si.struct \
        --fparams "$FPARAMS" \
        --cfg "$CFG_001" \
        --miller 0 0 1 \
        -o wien2k_001

    echo ""
    echo "--- Si comparison ---"
    python3 $COMPARE -p Si -d 001
else
    echo "WARNING: Si_DFT_total.xsf or Si_IAM_total.xsf is missing."
    echo "  Generate them with:"
    echo "    # DFT (SCF-converged):"
    echo "    cp Si.clmsum Si.clmval   # if necessary"
    echo "    \$WIENROOT/3ddens 3ddens_dft.def"
    echo "    mv Si.xsf Si_DFT_total.xsf"
    echo ""
    echo "    # IAM (dstart):"
    echo "    x dstart"
    echo "    \$WIENROOT/3ddens 3ddens_iam.def"
    echo "    mv Si.xsf Si_IAM_total.xsf"
fi

# ──────────────────────────────────────────────────────────────────────────
# GRAPHITE_ORT (orthogonalized hexagonal cell)
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "═══ GRAPHITE_ORT ═══"
cd ~/wien2k_tests/Graphite_ort

if [ -f "Graphite_ort_DFT_total.xsf" ] && [ -f "Graphite_ort_IAM_total.xsf" ]; then
    FPARAMS=fparams.dat
    [ ! -f "$FPARAMS" ] && FPARAMS=../Rutile/fparams.dat

    CFG=Graphite_ixchel_001.cfg
    [ ! -f "$CFG" ] && CFG=ixchel_001/Graphite_ixchel_001.cfg

    # create wien2k_001 if it does not exist yet
    mkdir -p wien2k_001

    echo ""
    echo "--- Graphite_ort [001] ---"
    python3 $XSF2TIFF \
        --dft Graphite_ort_DFT_total.xsf \
        --iam Graphite_ort_IAM_total.xsf \
        --struct Graphite_ort.struct \
        --fparams "$FPARAMS" \
        --cfg "$CFG" \
        --miller 0 0 1 \
        -o wien2k_001

    echo ""
    echo "--- Graphite_ort comparison ---"
    python3 $COMPARE -p Graphite_ort -d 001
else
    echo "WARNING: Graphite_ort DFT/IAM XSFs are missing."
fi

# ──────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════════════"
echo "  DONE — plots:"
echo "    ~/wien2k_tests/Rutile/Rutile_subslicing_comparison.png"
echo "    ~/wien2k_tests/Si/Si_subslicing_comparison.png"
echo "    ~/wien2k_tests/Graphite_ort/Graphite_ort_subslicing_comparison.png"
echo "════════════════════════════════════════════════════════════════════════"
