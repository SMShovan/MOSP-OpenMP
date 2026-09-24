/**
 * @file mosp.cpp
 * @brief Command-line driver: run the MOSP update on prepared inputs and
 *        report timings (and optionally validate the result).
 *
 * Usage:
 *   mosp --graph <csrPrefix> --changes <dir> --init <dir> [options]
 *
 *   --graph <prefix>   CSR text files <prefix>RowPtr.txt/ColInd.txt/Values.txt
 *   --changes <dir>    directory with insert.txt and delete.txt
 *   --init <dir>       initial SOSP trees: <dir>/obj<k>/distancesOriginal.txt
 *                      and <dir>/obj<k>/SSSPTreeOriginal.txt
 *   -k <K>             number of objectives to use (default: all in the graph)
 *   --source <s>       source vertex (default 0)
 *   --pref p1,..,pK    preference vector of the combined graph (default 1s;
 *                      lower value = higher priority, thesis Ch. 4 Step 2)
 *   --delta <D>        near-far bucket width (default: 32 * average weight /
 *                      average out-degree, per objective)
 *   --cache <file>     binary cache of the graph (written if missing/stale)
 *   --canonicalize     normalize the initial trees to the lowest-id tie rule
 *                      (for trees from other tools; not timed)
 *   --out <dir>        output directory (default mosp-output)
 *   --no-output        do not write the result files
 *   --validate         check every SOSP tree and the MOSP tree against host
 *                      Dijkstra (distances, parent consistency, canonical
 *                      parents)
 *   --timing <csv>     record per-stage timings and counters to a CSV file
 *   --quiet            print only the summary line
 *
 * Outputs (in --out): obj<k>/distancesUpdated.txt, obj<k>/SSSPTreeUpdated.txt,
 * combinedGraph/distancesCsr.txt (units of 1/L, L = lcm(Pref)),
 * combinedGraph/SSSPTreeCsr.txt and combinedGraph/mospCosts.txt ("v c1 .. cK":
 * the objective values of the MOSP path to v, thesis Step 3).
 *
 * Summary line (stable format, parsed by bench/run.sh):
 *   RESULT compute_ms=<a> end_to_end_ms=<b> threads=<t>
 * where (a) is the parallel compute of the K SOSP updates plus Steps 2-3 of
 * the MOSP update (the region the papers time) and (b) the wall time of
 * the whole run, from reading the inputs to writing the outputs. Pin the
 * threads for stable numbers: OMP_PROC_BIND=close OMP_PLACES=cores.
 */

#include "csrGraph.h"
#include "dijkstra.h"
#include "mospUpdate.h"
#include "parallelCombinedGraph.h"
#include "stageTimer.h"
#include "validation.h"

#include <omp.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

struct Options {
  string graph, changes, init, out = "mosp-output", timingCsv, cache;
  vector<int> pref;
  int K = 0;
  int source = 0;
  long long delta = 0;
  bool canonicalize = false;
  bool writeOutput = true;
  bool validate = false;
  bool quiet = false;
};

void usage() {
  cerr << "usage: mosp --graph <csrPrefix> --changes <dir> --init <dir>\n"
          "            [-k K] [--source s] [--pref p1,..,pK] [--delta D]\n"
          "            [--cache file] [--canonicalize] [--out dir]\n"
          "            [--no-output] [--validate] [--timing file.csv]\n"
          "            [--quiet]\n";
}

bool parseOptions(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    string a = argv[i];
    auto next = [&](string &dst) {
      if (i + 1 >= argc) {
        return false;
      }
      dst = argv[++i];
      return true;
    };
    string value;
    if (a == "--graph") {
      if (!next(opt.graph)) return false;
    } else if (a == "--changes") {
      if (!next(opt.changes)) return false;
    } else if (a == "--init") {
      if (!next(opt.init)) return false;
    } else if (a == "--out") {
      if (!next(opt.out)) return false;
    } else if (a == "--timing") {
      if (!next(opt.timingCsv)) return false;
    } else if (a == "--cache") {
      if (!next(opt.cache)) return false;
    } else if (a == "-k") {
      if (!next(value)) return false;
      opt.K = atoi(value.c_str());
    } else if (a == "--source") {
      if (!next(value)) return false;
      opt.source = atoi(value.c_str());
    } else if (a == "--delta") {
      if (!next(value)) return false;
      opt.delta = atoll(value.c_str());
    } else if (a == "--pref") {
      if (!next(value)) return false;
      istringstream list(value);
      string item;
      while (getline(list, item, ',')) {
        opt.pref.push_back(atoi(item.c_str()));
      }
    } else if (a == "--canonicalize") {
      opt.canonicalize = true;
    } else if (a == "--no-output") {
      opt.writeOutput = false;
    } else if (a == "--validate") {
      opt.validate = true;
    } else if (a == "--quiet") {
      opt.quiet = true;
    } else {
      cerr << "unknown option: " << a << "\n";
      return false;
    }
  }
  return !opt.graph.empty() && !opt.changes.empty() && !opt.init.empty();
}

double msSince(chrono::steady_clock::time_point start) {
  return chrono::duration<double, milli>(chrono::steady_clock::now() - start)
      .count();
}

/// Objective k of a K * n objective-major array.
template <typename T>
vector<T> slice(const vector<T> &all, int k, int n) {
  return vector<T>(all.begin() + static_cast<size_t>(k) * n,
                   all.begin() + static_cast<size_t>(k + 1) * n);
}

bool writeCosts(const string &path, const vector<long long> &costs, int K) {
  FILE *file = fopen(path.c_str(), "w");
  if (file == nullptr) {
    return false;
  }
  const size_t n = costs.size() / K;
  for (size_t v = 0; v < n; ++v) {
    fprintf(file, "%zu", v);
    for (int k = 0; k < K; ++k) {
      long long c = costs[v * K + k];
      if (c >= DISTANCE_INF / 2) {
        fprintf(file, " INF");
      } else {
        fprintf(file, " %lld", c);
      }
    }
    fputc('\n', file);
  }
  return fclose(file) == 0;
}

/// Check every tree and the MOSP tree against host Dijkstra.
bool validate(const CsrGraph &updated, const MospResult &result, int source,
              const vector<int> &pref) {
  const int n = updated.numberOfNodes;
  const int K = result.numberOfObjectives;
  CsrGraph reverse;
  transposeCsrGraph(updated, reverse);
  bool allOk = true;
  for (int k = 0; k < K; ++k) {
    vector<long long> refDist;
    vector<int> refParent;
    dijkstraCsrGraph(updated, k, source, refDist, refParent);
    TreeCheck check = checkSospTree(reverse, k, source,
                                    slice(result.distances, k, n),
                                    slice(result.parents, k, n), refDist,
                                    &refParent);
    bool ok = check.ok(true);
    allOk = allOk && ok;
    cout << "VALIDATE obj" << k << " " << (ok ? "PASS" : "FAIL") << " ("
         << check.summary() << ")\n";
  }
  // MOSP tree: SOSP of the combined graph built from the updated trees.
  CsrGraph combined =
      combinedGraphReference(result.parents, n, K, source, pref);
  CsrGraph combinedReverse;
  transposeCsrGraph(combined, combinedReverse);
  vector<long long> refDist;
  vector<int> refParent;
  dijkstraCsrGraph(combined, 0, source, refDist, refParent);
  TreeCheck check =
      checkSospTree(combinedReverse, 0, source, result.combinedDistances,
                    result.combinedParent, refDist, &refParent);
  bool ok = check.ok(true);
  cout << "VALIDATE combined " << (ok ? "PASS" : "FAIL") << " ("
       << check.summary() << ")\n";
  return allOk && ok;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!parseOptions(argc, argv, opt)) {
    usage();
    return 2;
  }
  setInstrumentation(!opt.timingCsv.empty());

  auto start = chrono::steady_clock::now();
  double loadMs = 0, canonicalizeMs = 0, writeMs = 0;

  // --- Inputs (read once; shared by all objectives) -------------------------
  auto t = chrono::steady_clock::now();
  CsrGraph original;
  ChangeBatch batch;
  {
    ScopedStage stage("read_graph");
    if (!loadCsrGraph(opt.graph, original, opt.cache)) {
      return 1;
    }
  }
  const int n = original.numberOfNodes;
  const int KG = original.numberOfObjectives;
  const int K = opt.K > 0 ? min(opt.K, KG) : KG;
  vector<long long> distances(static_cast<size_t>(K) * n);
  vector<int> parents(static_cast<size_t>(K) * n);
  {
    ScopedStage stage("read_changes_and_trees");
    if (!readChangeBatch(opt.changes + "/insert.txt",
                         opt.changes + "/delete.txt", KG, n, batch)) {
      return 1;
    }
    for (int k = 0; k < K; ++k) {
      vector<long long> dist;
      vector<int> parent;
      string dir = opt.init + "/obj" + to_string(k);
      if (!readDistances(dir + "/distancesOriginal.txt", n, dist) ||
          !readParents(dir + "/SSSPTreeOriginal.txt", n, parent)) {
        return 1;
      }
      copy(dist.begin(), dist.end(),
           distances.begin() + static_cast<size_t>(k) * n);
      copy(parent.begin(), parent.end(),
           parents.begin() + static_cast<size_t>(k) * n);
    }
  }
  loadMs = msSince(t);
  if (opt.canonicalize) {
    auto tc = chrono::steady_clock::now();
    for (int k = 0; k < K; ++k) {
      canonicalizeTree(original, k, opt.source, slice(distances, k, n),
                       parents.data() + static_cast<size_t>(k) * n);
    }
    canonicalizeMs = msSince(tc);
  }

  // --- MOSP update ------------------------------------------------------------
  MospOptions options;
  options.source = opt.source;
  options.numberOfObjectives = K;
  options.preferences = opt.pref;
  options.delta = opt.delta;
  CsrGraph updated;
  MospResult result;
  if (!mospUpdate(original, batch, distances, parents, options, updated,
                  result)) {
    return 1;
  }

  // --- Outputs ----------------------------------------------------------------
  if (opt.writeOutput) {
    auto tw = chrono::steady_clock::now();
    ScopedStage stage("write_outputs");
    bool ok = true;
    for (int k = 0; k < K && ok; ++k) {
      string dir = opt.out + "/obj" + to_string(k);
      ok = writeDistances(dir + "/distancesUpdated.txt",
                          slice(result.distances, k, n)) &&
           writeParents(dir + "/SSSPTreeUpdated.txt",
                        slice(result.parents, k, n));
    }
    vector<long long> costs;
    ok = ok &&
         writeDistances(opt.out + "/combinedGraph/distancesCsr.txt",
                        result.combinedDistances) &&
         writeParents(opt.out + "/combinedGraph/SSSPTreeCsr.txt",
                      result.combinedParent) &&
         mospPathCosts(updated, result.combinedParent, opt.source, costs) &&
         writeCosts(opt.out + "/combinedGraph/mospCosts.txt", costs, KG);
    if (!ok) {
      cerr << "cannot write the outputs to " << opt.out << "\n";
      return 1;
    }
    writeMs = msSince(tw);
  }
  const double endToEnd = msSince(start);

  // --- Report -----------------------------------------------------------------
  const MospTimings &tm = result.timings;
  if (!opt.quiet) {
    printf("graph  n=%d m=%d K=%d; batch %d inserts, %d deletes\n", n,
           updated.numberOfEdges(), K, batch.numberOfInserts(),
           batch.numberOfDeletes());
    printf("host   threads %d, read inputs %.1f ms, canonicalize %.1f ms, "
           "apply batch %.1f ms, prepare %.1f ms, write %.1f ms\n",
           omp_get_max_threads(), loadMs, canonicalizeMs, tm.applyBatch,
           tm.prepare, writeMs);
    for (int k = 0; k < K; ++k) {
      const SospStats &s = result.objectiveStats[k];
      printf("obj%d   SOSP update %.3f ms (invalidated %d, iterations %d, "
             "epochs %d, pushes %lld)\n",
             k, tm.objectives[k], s.invalidated, s.iterations, s.epochs,
             s.pushes);
    }
    const CombineStats &c = result.combineStats;
    printf("comb   combined graph + SOSP %.3f ms (%d edges, L=%lld, delta "
           "%lld, iterations %d, pushes %lld)\n",
           tm.combined, c.numberOfEdges, c.scale, c.delta, c.search.iterations,
           c.search.pushes);
  }
  if (!opt.timingCsv.empty()) {
    recordStage("TOTAL_end_to_end", endToEnd);
    if (!writeInstrumentationCsv(opt.timingCsv)) {
      cerr << "cannot write " << opt.timingCsv << "\n";
    }
  }
  printf("RESULT compute_ms=%.3f end_to_end_ms=%.3f threads=%d\n",
         tm.compute(), endToEnd, omp_get_max_threads());
  fflush(stdout);

  if (opt.validate && !validate(updated, result, opt.source, opt.pref)) {
    return 1;
  }
  return 0;
}
