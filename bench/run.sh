#!/usr/bin/env bash
# Run bin/mosp repeatedly on one prepared input and report the median of
# the parallel-compute (a) and end-to-end (b) times.
#
# usage: bench/run.sh <graphDir> <changesDir> [reps] [-- extra mosp options]
#
#   <graphDir>    directory with csr/graphCsr{RowPtr,ColInd,Values}.txt and
#                 init/obj<k>/{distancesOriginal,SSSPTreeOriginal}.txt
#   <changesDir>  directory with insert.txt and delete.txt
#   reps          number of runs (default 3)
#
# Environment:
#   OMP_NUM_THREADS, OMP_PROC_BIND, OMP_PLACES
#              default 28 threads pinned to cores (close); unpinned runs
#              on a shared host varied by more than 10x
#   OUT_DIR    where logs and outputs go (default bench-output/<date>);
#              outputs of each run are deleted after the run
#   MOSP_BIN   driver to run (default bin/mosp next to this script)
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
graph=${1:?graphDir}; changes=${2:?changesDir}; shift 2
# reps is optional: take the next argument only if it is a number.
reps=3
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then reps=$((10#$1)); shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
if (( reps < 1 )); then echo "reps must be >= 1" >&2; exit 2; fi
if [[ $# -gt 0 && "$1" != -* ]]; then
  echo "usage: bench/run.sh <graphDir> <changesDir> [reps] [-- mosp options]" >&2
  exit 2
fi
extra=("$@")
bin=${MOSP_BIN:-$here/bin/mosp}
out=${OUT_DIR:-$here/bench-output/$(date +%Y%m%d)}
mkdir -p "$out"
name="$(basename "$graph")_$(basename "$changes")"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-28}
export OMP_PROC_BIND=${OMP_PROC_BIND:-close}
export OMP_PLACES=${OMP_PLACES:-cores}

compute=(); e2e=()
for (( r = 1; r <= reps; r++ )); do
  log="$out/${name}_r$r.log"
  run_out="$out/${name}_r${r}_out"
  "$bin" --graph "$graph/csr/graphCsr" --changes "$changes" \
      --init "$graph/init" --out "$run_out" "${extra[@]}" > "$log" 2>&1
  rm -rf "$run_out"
  if ! line=$(grep '^RESULT' "$log"); then
    echo "$name run $r: no RESULT line (see $log)" >&2; exit 1
  fi
  c=$(sed -E 's/.*compute_ms=([0-9.eE+-]+).*/\1/' <<< "$line")
  t=$(sed -E 's/.*end_to_end_ms=([0-9.eE+-]+).*/\1/' <<< "$line")
  compute+=("$c"); e2e+=("$t")
  echo "$name run $r: compute_ms=$c end_to_end_ms=$t"
done
if (( ${#compute[@]} != reps )); then echo "missing timings" >&2; exit 1; fi
median() { printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END {print (NR%2 ? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}'; }
echo "MEDIAN $name compute_ms=$(median "${compute[@]}") end_to_end_ms=$(median "${e2e[@]}") reps=$reps threads=$OMP_NUM_THREADS"
