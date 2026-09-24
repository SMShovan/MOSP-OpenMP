#!/usr/bin/env bash
# Time the original code (built by bench/baseline/build.sh) on one prepared
# input and report the median (a) and (b), as defined in results/README.md:
#   (a) the propagation loops and BFS post-passes of the K + 1 SOSP calls
#       (stages */2d_propagate_loop_omp, */3_bfs_reachability_omp) plus
#       the OpenMP membership map of the combined graph
#       (comb/1_membership_std_map_omp)
#   (b) TOTAL_WALL_MS: the K SOSP updates and the combined graph, from
#       reading the text inputs to writing the outputs
#
# usage: bench/baseline/run.sh <mospBench> <graphDir> <changesDir> [reps] [K]
#
#   <mospBench>   mospBench_asis or mospBench_O3
#   <graphDir>    as for bench/run.sh (csr/graphCsr*.txt, init/obj<k>/)
#   reps          number of runs (default 3); K objectives (default 3)
#
# Environment: OMP_NUM_THREADS, OMP_PROC_BIND, OMP_PLACES (default 28
# threads pinned to cores), OUT_DIR (logs; default
# bench-output/baseline-<date>). If <changesDir>/expected exists (from
# `mospPrep expected`), the driver also compares the updated distances.
set -euo pipefail
here=$(cd "$(dirname "$0")/../.." && pwd)
bin=${1:?mospBench}; graph=${2:?graphDir}; changes=${3:?changesDir}
reps=${4:-3}; K=${5:-3}
[[ "$reps" =~ ^[1-9][0-9]*$ && "$K" =~ ^[1-9][0-9]*$ ]] ||
  { echo "reps and K must be positive integers" >&2; exit 2; }
out=${OUT_DIR:-$here/bench-output/baseline-$(date +%Y%m%d)}
mkdir -p "$out"
name="$(basename "$bin")_$(basename "$graph")_$(basename "$changes")_K$K"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-28}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}
expected=()
[[ -d "$changes/expected" ]] && expected=(--expected "$changes/expected")

a=(); b=()
for (( r = 1; r <= reps; r++ )); do
  log="$out/${name}_r$r.log"
  run_out="$out/${name}_r${r}_out"
  "$bin" "$graph/csr/graphCsr" "$K" 0 "$changes" "$graph/init" "$run_out" \
      "${expected[@]}" > "$log" 2>&1 || true
  rm -rf "$run_out"
  grep -q '^TOTAL_WALL_MS' "$log" || { echo "run $r failed (see $log)" >&2; exit 1; }
  ta=$(awk -v re='(2d_propagate_loop_omp|3_bfs_reachability_omp|1_membership_std_map_omp)$' \
      '$1 == "STAGE" && $2 ~ re { s += $3 } END { printf "%.3f", s }' "$log")
  tb=$(awk '$1 == "TOTAL_WALL_MS" { print $2 }' "$log")
  a+=("$ta"); b+=("$tb")
  pass=$(grep -c 'VALIDATE.*PASS' "$log" || true)
  echo "$name run $r: a_ms=$ta b_ms=$tb validate_pass=$pass"
done
median() { printf '%s\n' "$@" | sort -g | awk '{v[NR]=$1} END {print (NR%2 ? v[(NR+1)/2] : (v[NR/2]+v[NR/2+1])/2)}'; }
echo "MEDIAN $name a_ms=$(median "${a[@]}") b_ms=$(median "${b[@]}") reps=$reps threads=$OMP_NUM_THREADS"
