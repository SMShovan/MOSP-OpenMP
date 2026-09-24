#!/usr/bin/env bash
# Prepare one benchmark graph from a SuiteSparse Matrix Market file:
#   <outDir>/csr/graphCsr*.txt   K random weights in [1,100] per edge
#   <outDir>/init/obj<k>/        initial SOSP trees (Dijkstra, source 0)
#   <outDir>/changes_<N>_<P>/    uniform batch, N changes, P% insertions
#                                (the repository's generator, seed 777)
#   <outDir>/changes_<N>_<P>_safe/  same insertions, deletions filtered so
#                                no vertex becomes unreachable
#   <outDir>/changes_local_10000_<P>_safe/  (only with HOPS) 10K changes,
#                                P% insertions, every endpoint within HOPS
#                                undirected hops of a random centre,
#                                connectivity-safe (seed 777)
#
# usage: bench/prepare.sh <in.mtx> <outDir> [K=3] [N=50000] [P=50] [HOPS]
#
# The graphs used in results/ are roadNet-PA, roadNet-CA (SNAP) and
# rgg_n_2_20_s0, road_usa (DIMACS10) from https://sparse.tamu.edu; their
# local batches used HOPS = 110 (roadNet-PA), 160 (roadNet-CA),
# 110 (rgg_n_2_20_s0) and 200 (road_usa).
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
prep=$here/bin/mospPrep
mtx=${1:?in.mtx}; out=${2:?outDir}; K=${3:-3}; N=${4:-50000}; P=${5:-50}
hops=${6:-}
"$prep" mtx2csr "$mtx" "$out/csr/graphCsr" "$K" 1 100 12345
"$prep" init "$out/csr/graphCsr" "$out/init"
"$prep" changes "$out/csr/graphCsr" "$out/changes_${N}_${P}" \
    --changes "$N" --ins "$P" --seed 777
"$prep" changes "$out/csr/graphCsr" "$out/changes_${N}_${P}_safe" \
    --changes "$N" --ins "$P" --seed 777 --safe
if [[ -n "$hops" ]]; then
  "$prep" changes "$out/csr/graphCsr" "$out/changes_local_10000_${P}_safe" \
      --changes 10000 --ins "$P" --seed 777 --local "$hops" --safe
fi
