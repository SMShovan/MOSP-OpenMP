# Changes on `fix/correctness-perf`

Branch point: `baseline-2026-09` (`7284f50`, "Code Refactored For Modularity
and Readability."). Every commit builds and passes `make test`. MOSP-CUDA has
the same fixes on its branch; both implementations now produce identical
trees. Measurements: see [results/README.md](results/README.md).

Two scopes are reported:

- **(a) compute**: the region the papers time, i.e. the parallel compute of
  the K SOSP updates and of Steps 2-3 of the MOSP update (combined graph +
  its SOSP). For the original code this is the propagation loops plus the
  BFS post-passes of the K + 1 SOSP calls plus the (OpenMP) membership map of
  the combined graph; its sequential Step 1 and its adjacency-list building
  are not counted. For the new code it includes Step 1 and the
  combined-graph construction but not applying the batch and building the
  reverse graph and weight columns ("prepare").
- **(b) end to end**: wall time from reading the inputs (text CSR, change
  batch, initial trees) to writing all outputs.

All timings use 28 threads pinned to the physical cores
(`OMP_NUM_THREADS=28 OMP_PROC_BIND=close OMP_PLACES=cores`) of a Xeon Gold
6258R; medians of 3 runs unless noted; K = 3; 50K changes with 50%
deletions. "Original" is `baseline-2026-09` built with the system g++ and
the original flags (no optimization); "original -O3" adds `-O3`.

## Summary

- **Correctness.** The SOSP update could return *too small distances for
  reachable vertices* (count to infinity cut off by an iteration cap; the
  stock OpenMP stress test failed intermittently) and needed n iterations
  whenever a batch disconnects vertices. Fixed by subtree invalidation plus
  a monotone update (M-a). Parent ties are broken by the lowest vertex id
  everywhere (M-c): the trees no longer depend on the OpenMP schedule, equal
  the Dijkstra trees exactly and are byte-identical to MOSP-CUDA's. The
  preference vector is implemented (M-d); batches can be generated
  reproducibly (M-e).
- **Performance** (connectivity-safe 50K batch, K = 3):

| graph | (a) original as-is / -O3 | (a) new | (a) speedup vs as-is / -O3 | (b) original as-is / -O3 | (b) new | (b) speedup vs as-is / -O3 |
|---|---:|---:|---:|---:|---:|---:|
| roadNet-PA | 3.49 s / 1.45 s | 63.2 ms | 55.3x / 22.9x | 41.6 s / 20.9 s | 645 ms | 64.6x / 32.5x |
| roadNet-CA | 6.07 s / 2.34 s | 110 ms | 55.4x / 21.3x | 74.7 s / 37.3 s | 1.17 s | 63.8x / 31.9x |
| rgg_n_2_20_s0 | – / 2.45 s | 166 ms | – / 14.7x | – / 57.2 s | 1.30 s | – / 44.0x |
| road_usa | – / 45.0 s | 1.32 s | – / 34.2x | – / 637 s | 12.9 s | – / 49.3x |

  The road_usa original ran once, under a host load of about 45 from other
  jobs (the other runs: 7-29); at low load its update took 7.47 s per
  objective instead of 10.1 s, so its ratios are probably overstated by up
  to ~30%. On the local 10K batch (a) improves 10.8-27.2x and (b) 33-36x
  against the -O3 build. On the unfiltered batch the original needs n
  iterations per objective (roadNet-PA: 106 s for one objective, against
  14.1 ms).

## Correctness

### M-a: count-to-infinity on disconnection, and wrong distances (abf4a34)

Same defect and fix as in MOSP-CUDA. After deleting (or raising the weight
of) a tree edge (u,v), Step 1 could pick a descendant of v as its new parent;
the resulting stale cycle "counts to infinity". The loop was capped at
`maxIterations = n` and followed by a BFS that only repairs unreachable
vertices, which

1. gives **wrong distances** for reachable vertices when the cycle needs more
   than n rounds: the stock OpenMP stress test failed intermittently (e.g.
   n = 6, graphSeed 621705, changeSeed 250813: vertex 1 gets 60 instead of
   90; the sequential reference failed too). Three such cases are now
   regression tests in `bin/mospTest`;
2. is **slow** when deletions disconnect vertices (n iterations).

Fix: the SOSP subtrees of the heads of deleted or weight-increased tree
edges are invalidated (in parallel: every vertex walks up its parent chain to
the first vertex with a known state and writes that state along the path),
and the update is monotone; the cap and the BFS post-pass are gone. The
sequential reference got the same fix.

Disconnecting batch (the 50K batch without the connectivity filter). The
original was run with K = 1 only (one objective takes minutes):

| graph | original -O3, one objective: SOSP update | original -O3, K = 1: (b) | new, per objective | new, K = 3: (a) / (b) |
|---|---:|---:|---:|---:|
| roadNet-PA | 106 s | 117 s | 14.1 ms | 63.6 ms / 645 ms |

That is 7,500x per objective. The new code's time is the same as on the
connectivity-safe batch (roadNet-CA 109 ms / 1.14 s, road_usa 1.31 s /
13.6 s for K = 3).

### M-c: defined tie-break (cd8f10b)

Among in-neighbours with equal distance the lowest vertex id becomes the
parent, in Dijkstra (all three variants), `findBestParent`, Step 1 and the
new engine (compare-and-swap minimum on the packed (distance, parent) word).
The parent no longer depends on the OpenMP schedule: every updated tree
equals the Dijkstra tree exactly, two runs give identical files, and the
results are byte-identical to MOSP-CUDA's (checked on roadNet-CA with the
disconnecting batch: all trees, the MOSP tree and the MOSP costs). The
tracked `tests/` cases regenerate byte for byte (they contain no ties).

### M-d: preference vector (f091a8d)

`parallelCombinedGraph(..., preferences)` and `bin/mosp --pref` implement
W(e) = K + 1 - sum_{i: e in T_i} 1/Pref_i (exact, scaled by L = lcm(Pref));
`mospPathCosts` returns the K objective values along the MOSP tree. The
worked example of the thesis (Ch. 4, "Finding a single MOSP": cost
(15, 3, 20) for Pref = {4, 1, 4}, (15, 24, 7) for Pref = {4, 4, 1}) is a test.

### M-e: seeded change generator (6ac28af)

Same generator as in MOSP-CUDA (`bin/mospPrep changes`): uniform (identical
to `generateChangedEdges`), thesis-style targeted, reweight, tree-edge weight
increases, `--local HOPS`, `--safe`.

### Smaller fixes

- `generateGraph` did not create `data/`, so `main` failed on a fresh
  checkout (c099516).
- `runDijkstraCSR` rejected every objective index for a graph whose batch
  deleted all edges (stress-test seed 53, run 26); the stress tests now
  print the parameters of a failing run (e5719f6).
- `make test` runs everything in `test-output/` with 4 threads
  (`TEST_THREADS`); with 56 threads on a busy host the tiny test graphs made
  OpenMP barriers crawl for more than 10 minutes (d2d759a).
- `bin/mosp --validate` requires canonical (lowest-id) parents only together
  with `--canonicalize` (fa6abe2).

## Performance

### Build (da40a13)

The Makefile hard-coded `CXX := g++-15` (not installed) and no optimization;
it now uses the system g++ at `-O3` with header dependencies. The committed
macOS arm64 objects in `bin/` and `build/` are no longer tracked. `-O3` alone
halves the original's end-to-end time and cuts its compute by 2.4-2.6x.

### M-f: allocation and merging in the propagation loop (ebf3674)

`isCandidate` (n bytes) was allocated and zeroed in every iteration and the
thread-local lists were merged under `omp critical`. The flags are now
allocated once and only the listed entries are reset; lists are merged at
prefix-sum offsets (`ListGather`). On roadNet-PA the effect was within noise
(the candidate re-evaluation dominated); the loop itself was replaced next.

### MP5: work-efficient push (608b2a5)

`sospUpdateCpu` is the CPU counterpart of MOSP-CUDA's engine: roots from the
change list, subtree invalidation, one pull pass over the invalidated
vertices and insertion heads, then a near-far push worklist with a
compare-and-swap minimum on the packed (distance << b | parent) word,
generation stamps instead of per-iteration resets, prefix-sum merges, and
the same run-time packing with a distance-only fallback plus parent recovery
for large n or weights. Per objective (original -O3 -> new): roadNet-PA 159
-> 14.0 ms, roadNet-CA 287 -> 23.1 ms, rgg 371 -> 50.1 ms, road_usa 10.1 s
(7.47 s at low host load) -> 308 ms.

### MP3: combined graph per vertex (ac83772)

The thread-local `std::map`s of tree edges merged under `omp critical` took
0.9 s (roadNet-PA) to 1.3 s (roadNet-CA, rgg), plus temporary text files and
a full file-based SOSP update. Now every vertex compares its K parents, a
count / prefix-sum / fill pass builds the out-edge CSR, and Step 3 is the
near-far SSSP of the MP5 engine. Steps 2-3 (original -O3 map + SOSP -> new):
roadNet-PA 882 + 95.5 -> 16.2 ms, roadNet-CA 1.27 s + 167 -> 25.8 ms, rgg
1.28 s + 76.2 -> 19.1 ms, road_usa 11.7 s + 3.21 s -> 367 ms.

### H-M1: in-memory pipeline (64e446a, 245e3b4)

`mospUpdate()` + `bin/mosp`: the text inputs are read once (the original
parsed the K-weight CSR K + 1 times), the batch is applied once, the reverse
graph and the K weight columns are built once in parallel and shared by all
objectives, buffers are allocated once, no temporary files, and the text
files are read and written concurrently on the OpenMP threads (so pinning is
respected). `main` computes the initial trees before the update loop.

Where the end-to-end time goes (K = 3, 50K safe batch):

| graph | original -O3: text reads / text writes / total | new: read / apply batch / prepare / (a) / write / total |
|---|---|---|
| roadNet-PA | 15.9 s / 2.44 s / 20.9 s | 285 ms / 58.1 ms / 42.0 ms / 63.2 ms / 196 ms / 645 ms |
| roadNet-CA | 28.5 s / 4.38 s / 37.3 s | 546 ms / 78.5 ms / 70.8 ms / 110 ms / 352 ms / 1.17 s |
| rgg_n_2_20_s0 | 48.9 s / 2.77 s / 57.2 s | 614 ms / 126 ms / 133 ms / 166 ms / 249 ms / 1.30 s |
| road_usa | 483 s / 76.6 s / 637 s | 4.53 s / 728 ms / 860 ms / 1.32 s / 5.45 s / 12.9 s |

The road_usa original ran under a high host load (see the summary). The new
end-to-end time is mostly text I/O; `--cache` (binary CSR) cuts the reading
part as in MOSP-CUDA.

## Tests

- `make test` (4 threads): stock pipeline + 10 tracked test cases, both
  stress tests (distances and trees must equal Dijkstra) and `bin/mospTest`
  (same suite as MOSP-CUDA, including the in-memory pipeline, the thesis
  example, the regressions and the large-weight fallback).
- 30 additional stress seeds with 4 threads and 10 with 16 threads pass; the
  oracle suite passes with 16 threads.
- Real graphs: `bin/mosp --validate --canonicalize` (28 threads) on
  roadNet-PA, roadNet-CA, rgg_n_2_20_s0 and road_usa with the safe,
  disconnecting and local batches: every tree and the MOSP tree identical
  to Dijkstra.

## Behaviour changes

Same as MOSP-CUDA: lowest-id parents on ties, INF/-1 for vertices cut off
by the batch, combined distances in units of 1/L with a Pref vector,
`parallelCombinedGraph` writes no temporary files (`workDir` unused).

## Not done / deviations

- Update-vs-recompute selection and a static recompute baseline: parked.
- The file-based `parallelSOSPUpdate` (used by `main`, the stress tests and
  the test cases) still builds adjacency lists per call; the fast path is
  `mospUpdate` (bin/mosp).
- The tracked output trees (`data/`, `output/`, `tests/`,
  `parallelStressTest/`, `html/`) were left in place; `make test` does not
  touch them.

## Known issues and risks

- **Small graphs and many threads.** With 56 threads the tiny graphs of the
  test suite spend their time in OpenMP barriers (hence `TEST_THREADS=4`);
  the engine has no sequential cut-off for small frontiers.
- **Local batches** gain less than large ones (10.8x on rgg, compute) since
  the near-far loop runs hundreds of barrier-separated iterations with
  little work each; the default Delta is a heuristic (see MOSP-CUDA's
  results for a sweep).
- **Measurements** were taken on a shared host (other jobs were running;
  the runs of this campaign did not overlap). The original was not run
  as-is on rgg, road_usa and the local batches, and on the disconnecting
  batch only with K = 1 on roadNet-PA; the road_usa K = 3 original ran once.
- **Index width.** Vertex ids, edge offsets and weights are 32-bit, as in the
  original.
- **Outputs differ from the original's** where distances tie (parents) and
  for vertices the original left with a wrong distance (M-a).
