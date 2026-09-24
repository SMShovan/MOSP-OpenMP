# MOSPOpenMP

Shared-memory (OpenMP) implementation of the Multi-Objective Shortest Path
(MOSP) update algorithm for dynamic networks (DynaMOSP). MOSP-CUDA is the
GPU implementation of the same algorithm; both produce identical trees.

Given a multi-objective graph, one SOSP (single-objective shortest path)
tree per objective and a batch of edge insertions/deletions, it

1. updates the K SOSP trees (Steps 1-2 of the SOSP update),
2. builds the combined graph of the K trees with preference weights
   W(e) = K + 1 - sum_{i: e in T_i} 1/Pref_i, and
3. finds the SOSP tree of the combined graph: the MOSP tree, whose paths
   are reported with their K objective values.

See [CHANGES.md](CHANGES.md) for the fixes and optimizations of the
`fix/correctness-perf` branch and [results/](results/README.md) for the
measurements.

## Requirements

- C++17 compiler with OpenMP (tested with g++ 12)

## Build

From the project root:

```
make                        # bin/main, bin/mosp, bin/mospPrep, bin/mospTest
make CXX=g++-15             # a specific compiler (default: the system g++)
make OPT=                   # original flags: no optimization
```

Everything is compiled at `-O3` with `-fopenmp`; header changes trigger
rebuilds. For clangd and other IDE tooling, generate an (untracked)
`compile_commands.json` with `bear -- make`.

## Threads and pinning

Pin the threads for any timing: on a shared host, unpinned runs varied by
more than 10x.

```
OMP_NUM_THREADS=28 OMP_PROC_BIND=close OMP_PLACES=cores bin/mosp ...
```

`bench/run.sh` sets these by default (28 threads = the physical cores of
the test machine). `make test` uses 4 threads (`TEST_THREADS`), because
the test graphs are tiny and more threads than free cores make OpenMP
barriers crawl.

## Tests

```
make test                   # everything below, inside test-output/
make test TEST_SEED=0       # random seed for the stress tests (printed)
make test TEST_THREADS=16   # more OpenMP threads for the tests
```

- `bin/main`: the demo pipeline plus 10 generated test cases.
- `bin/stressTest [seed] [runs]`, `bin/parallelStressTest [seed] [runs]`:
  100 random graphs each; distances *and* SSSP trees must equal Dijkstra.
- `bin/mospTest [--seed S] [--only GROUP]`: oracle suite. Seeded random
  graphs and road-like grids with every kind of batch (uniform
  connectivity-safe and disconnecting, deletions only, disconnecting
  deletions, insertions only, tree-edge weight increases, re-weighting,
  thesis-style targeted, local), run through the file-based APIs and the
  in-memory pipeline, and checked against host Dijkstra (distances, parent
  consistency, identical canonical parents, identical output of two runs)
  and a host reference of the combined graph (default and skewed Pref).
  Also: the worked example of the thesis (Ch. 4, "Finding a single MOSP"),
  count-to-infinity regressions, a cyclic input tree, the distance-only
  fallback for large weights (random weights, and equal weights whose
  many ties exercise the lowest-id parent recovery), the boundary of the
  packed format, the binary cache (source identity, damaged files),
  generator and batch-application equivalence checks.

## Running on real graphs

`bin/mospPrep` prepares inputs, `bin/mosp` runs the MOSP update on them.

```
bin/mospPrep mtx2csr roadNet-CA.mtx g/csr/graphCsr 3 1 100 12345  # K=3, w in [1,100]
bin/mospPrep init g/csr/graphCsr g/init                           # initial trees
bin/mospPrep changes g/csr/graphCsr g/changes --changes 50000 --ins 50 --seed 777 --safe
bin/mosp --graph g/csr/graphCsr --changes g/changes --init g/init --out out --validate
```

`bench/prepare.sh` wraps the preparation (optionally with a local
batch), `bench/run.sh` repeats runs with pinned threads and reports
medians, and `bench/baseline/` builds and times the original code of the
tag `baseline-2026-09` the same way (see [results/](results/README.md)).

### bin/mosp

```
mosp --graph <csrPrefix> --changes <dir> --init <dir> [options]
  -k K               use the first K objectives (default: all; required
                     for a graph without edges, whose K cannot be inferred)
  --source s         source vertex (default 0)
  --pref p1,..,pK    preference vector (default all 1s; lower = higher priority)
  --delta D          near-far bucket width (default 32 * avg weight / avg degree)
  --cache file       binary cache of the graph (see below)
  --canonicalize     normalize initial trees from other tools to the tie rule
  --out dir          output directory; --no-output to skip writing
  --validate         check all trees against host Dijkstra
  --timing file.csv  per-stage timings and counters
  --quiet            only the summary line
```

`--cache` keeps a binary copy of the CSR next to the text. The cache
records the canonical path, size and modification time of the three text
files it was built from and is used only when they still match; a cache
that is stale, damaged or was written for another graph is rebuilt (with
a note on stderr). Rewriting a text file with identical size and
modification time is not detected; delete the cache in that case.

The summary line `RESULT compute_ms=<a> end_to_end_ms=<b> threads=<t>`
reports (a) the parallel compute of the K SOSP updates and of Steps 2-3
(the region the papers time) and (b) the wall time from reading the
inputs to writing the outputs. Outputs: `obj<k>/distancesUpdated.txt`, `obj<k>/SSSPTreeUpdated.txt`,
`combinedGraph/distancesCsr.txt` (in units of 1/L, L = lcm(Pref)),
`combinedGraph/SSSPTreeCsr.txt`, `combinedGraph/mospCosts.txt` (the K
objective values of the MOSP path to every vertex).

### bin/mospPrep

```
mospPrep mtx2csr <in.mtx> <outPrefix> <K> <wmin> <wmax> <seed>
mospPrep widen <inPrefix> <outPrefix> <K> <wmin> <wmax> <seed>
mospPrep cache <csrPrefix> <binaryPath>
mospPrep changes <csrPrefix> <outDir> [--changes N] [--ins PCT]
                 [--mode uniform|targeted|reweight|increase] [--local HOPS]
                 [--safe] [--seed S] [--source s] [--wmin a] [--wmax b] [-k K]
mospPrep init <csrPrefix> <outDir> [--source s] [-k K]
mospPrep expected <csrPrefix> <changesDir> <outDir> [--source s] [-k K]
```

`-k K` gives the number of objectives of a graph without edges (its
empty Values file does not tell it); weights must satisfy
1 <= wmin <= wmax <= 2^31-1.

Change modes: `uniform` (the original generator, identical output for the
same seed), `targeted` (thesis workload: below-average inserted weights,
deletions of SOSP-tree edges), `reweight` (new weights on existing edges),
`increase` (weight increases on tree edges); `--local HOPS` confines a
batch to a ball of HOPS undirected hops, `--safe` drops deletions that
would disconnect a vertex from the source.

## Semantics worth knowing

- **Tie-break:** among in-neighbours u with equal distance d[u] + w(u,v),
  the parent of v is the lowest u, everywhere (Dijkstra, sequential,
  CUDA, OpenMP). Trees are therefore deterministic and identical across
  implementations, and the combined graph no longer depends on the order
  of operations.
- **Disconnection:** vertices that the batch cuts off from the source get
  distance INF and parent -1 (the update invalidates the subtrees of
  deleted or weight-increased tree edges instead of counting to infinity).
- **Weights** must be integers in [1, 2^31-1] (the loaders reject other
  values, and tree files must list every vertex once); distances are
  64-bit.
- **Scale:** the search packs (distance, parent) into 64 bits when
  n * maxWeight (the longest path plus one edge) fits next to the parent
  ids and otherwise falls back to 64-bit distances with a
  parent-recovery pass.
- **Tracked files:** `data/`, `output/`, `tests/` and `parallelStressTest/`
  hold outputs of earlier runs; `make test` runs everything inside
  `test-output/` and leaves them untouched, but `./bin/main` and `make run`
  (below) rewrite the tracked files in `data/`, `output/` and `tests/`
  (restore them with `git checkout -- data output tests`).

## Stock pipeline (bin/main)

From the project root:

```
./bin/main          # or: make run
```

It runs in the current directory and overwrites the tracked sample files
listed below, so `git status` shows them as modified afterwards; run it
from a scratch directory (as `make test` does in `test-output/`) to keep
the checkout clean.

The app writes the output graph to:
- `data/graph.mtx`
- `data/originalGraph/graphCsrRowPtr.txt`
- `data/originalGraph/graphCsrColInd.txt`
- `data/originalGraph/graphCsrValues.txt`

Then it applies `output/changedEdges/*.txt` changes and writes:
- `data/updatedGraph/updatedGraphCsrRowPtr.txt`
- `data/updatedGraph/updatedGraphCsrColInd.txt`
- `data/updatedGraph/updatedGraphCsrValues.txt`

Then it runs Dijkstra and writes original-graph results to:
- `output/distancesTrees/distances.txt`
- `output/distancesTrees/SSSPTree.txt`
- `output/distancesTrees/distancesCsr.txt`
- `output/distancesTrees/SSSPTreeCsr.txt`

Then it runs Dijkstra on the updated CSR graph and writes:
- `output/updatedDistancesTrees/updatedDistancesCsr.txt`
- `output/updatedDistancesTrees/updatedSSSPTreeCsr.txt`

Then it runs the Sequential SOSP Update algorithm (incremental update without
recomputing Dijkstra from scratch) and writes:
- `output/sospUpdateDistancesTrees/distancesCsr.txt`
- `output/sospUpdateDistancesTrees/SSSPTreeCsr.txt`

It also generates edge-change files:
- `output/changedEdges/insert.txt`
- `output/changedEdges/delete.txt`

Then it computes the initial tree of every objective, runs the OpenMP SOSP
update per objective (`output/parallelSospObj<k>/`) and the combined graph
(`output/combinedGraph/`).

## Test Cases

The app also generates 10 deterministic test cases under `tests/testCaseN/`, each containing:
- `originalGraph/` (CSR files)
- `changedEdges/` (insert.txt, delete.txt)
- `updatedGraph/` (CSR files after applying changes)
- `expected/` (ground truth distances and SSSP trees for original and updated graphs,
  plus SOSP update algorithm output for comparison)

These are seeded for reproducibility and vary across graph size, objective count,
change ratio, and Dijkstra objective index.

## Doxygen

Generate docs from the project root:

```
doxygen Doxyfile
```

Open the HTML output:

```
open html/index.html
```
