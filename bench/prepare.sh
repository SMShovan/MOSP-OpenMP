#!/usr/bin/env bash
# Prepare one benchmark graph from a SuiteSparse Matrix Market file:
#   <outDir>/csr/graphCsr*.txt   K random weights in [1,100] per edge
#   <outDir>/init/obj<k>/        initial SOSP trees (Dijkstra, source 0)
#   <outDir>/changes_<N>_<P>/    uniform batch, N changes, P% insertions
#                                (the repository's generator, seed 777)
#   <outDir>/changes_<N>_<P>_safe/  same insertions, deletions filtered so
#                                no vertex becomes unreachable
#
# usage: bench/prepare.sh <in.mtx> <outDir> [K=3] [N=50000] [P=50]
#
# The graphs used in results/ are roadNet-PA, roadNet-CA (SNAP) and
# rgg_n_2_20_s0, road_usa (DIMACS10) from https://sparse.tamu.edu.
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
prep=$here/bin/mospPrep
mtx=${1:?in.mtx}; out=${2:?outDir}; K=${3:-3}; N=${4:-50000}; P=${5:-50}
"$prep" mtx2csr "$mtx" "$out/csr/graphCsr" "$K" 1 100 12345
"$prep" init "$out/csr/graphCsr" "$out/init"
"$prep" changes "$out/csr/graphCsr" "$out/changes_${N}_${P}" \
    --changes "$N" --ins "$P" --seed 777
"$prep" changes "$out/csr/graphCsr" "$out/changes_${N}_${P}_safe" \
    --changes "$N" --ins "$P" --seed 777 --safe
