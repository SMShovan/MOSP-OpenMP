#!/usr/bin/env bash
# Build the timing driver for the original code (tag baseline-2026-09):
#   <outDir>/src-tree/          the original sources (git archive of the tag)
#                               plus stage-timers.patch, prof.h, mospBench.cpp
#   <outDir>/mospBench_asis     original flags (no optimization)
#   <outDir>/mospBench_O3       the same with -O3
#
# usage: bench/baseline/build.sh <outDir> [CXX=g++]
#
# The patch only adds stage timers (prof.h) to parallelSOSPUpdate() and
# parallelCombinedGraph(); the algorithms are unchanged. bench/baseline/
# run.sh runs the drivers and reports the original's (a) and (b).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../.." && pwd)
out=${1:?outDir}; cxx=${2:-g++}
tree="$out/src-tree"
rm -rf "$tree" && mkdir -p "$tree"
git -C "$repo" archive baseline-2026-09 src headers | tar -x -C "$tree"
patch -d "$tree" -p1 --quiet < "$here/stage-timers.patch"
cp "$here/prof.h" "$tree/headers/"
cp "$here/mospBench.cpp" "$tree/src/"
srcs=(src/mospBench.cpp src/parallelSOSPUpdate.cpp src/parallelCombinedGraph.cpp
      src/read.cpp src/Dijkstra.cpp)
flags=(-std=c++17 -Wall -Wextra -Iheaders -fopenmp)   # baseline Makefile
(cd "$tree" && "$cxx" "${flags[@]}" -o ../mospBench_asis "${srcs[@]}")
(cd "$tree" && "$cxx" "${flags[@]}" -O3 -o ../mospBench_O3 "${srcs[@]}")
echo "built $out/mospBench_asis and $out/mospBench_O3"
