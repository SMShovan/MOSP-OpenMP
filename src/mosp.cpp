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
 *   --out <dir>        output directory (default mosp-output)
 *   --validate         check every SOSP tree against host Dijkstra on the
 *                      updated graph (distances + parent consistency)
 *   --timing <csv>     record per-stage timings and counters to a CSV file
 *   --quiet            print only the summary lines
 *
 * Summary lines (stable format, parsed by bench/run.sh):
 *   RESULT compute_ms=<a> end_to_end_ms=<b> threads=<t>
 * where (a) is the parallel compute of the K SOSP updates plus the SOSP on
 * the combined graph (the region the papers time) and (b) the wall time of
 * the whole run, from reading the inputs to writing the outputs. Pin the
 * threads for stable numbers: OMP_PROC_BIND=close OMP_PLACES=cores.
 */

#include "csrGraph.h"
#include "dijkstra.h"
#include "parallelCombinedGraph.h"
#include "parallelSOSPUpdate.h"
#include "stageTimer.h"
#include "validation.h"

#include <omp.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

struct Options {
  string graph, changes, init, out = "mosp-output", timingCsv;
  int K = 0;
  int source = 0;
  bool validate = false;
  bool quiet = false;
};

void usage() {
  cerr << "usage: mosp --graph <csrPrefix> --changes <dir> --init <dir>\n"
          "            [-k K] [--source s] [--out dir] [--validate]\n"
          "            [--timing file.csv] [--quiet]\n";
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
    } else if (a == "-k") {
      if (!next(value)) return false;
      opt.K = atoi(value.c_str());
    } else if (a == "--source") {
      if (!next(value)) return false;
      opt.source = atoi(value.c_str());
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

} // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!parseOptions(argc, argv, opt)) {
    usage();
    return 2;
  }
  if (!opt.quiet) {
    cout.setf(ios::unitbuf);
  }
  // Stage timers are always on in this driver: the compute scope is the sum
  // of the stages whose name ends in "_compute".
  setInstrumentation(true);
  const string csv =
      opt.timingCsv.empty() ? opt.out + "/stages.csv" : opt.timingCsv;

  auto start = chrono::steady_clock::now();
  int K = opt.K;
  if (K <= 0) {
    // Count the weights on the first line of the values file.
    FILE *file = fopen((opt.graph + "Values.txt").c_str(), "r");
    char line[4096];
    if (file == nullptr || !fgets(line, sizeof(line), file)) {
      cerr << "cannot read " << opt.graph << "Values.txt\n";
      return 1;
    }
    fclose(file);
    istringstream tokens(line);
    int w;
    K = 0;
    while (tokens >> w) {
      ++K;
    }
  }

  vector<string> trees;
  for (int k = 0; k < K; ++k) {
    string objIn = opt.init + "/obj" + to_string(k);
    string objOut = opt.out + "/obj" + to_string(k);
    setStagePrefix("obj" + to_string(k) + "/");
    auto t = chrono::steady_clock::now();
    bool ok = parallelSOSPUpdate(
        opt.graph, objIn + "/distancesOriginal.txt",
        objIn + "/SSSPTreeOriginal.txt", opt.changes + "/insert.txt",
        opt.changes + "/delete.txt", k, opt.source,
        objOut + "/distancesUpdated.txt", objOut + "/SSSPTreeUpdated.txt");
    recordStage("TOTAL_sosp_update", msSince(t));
    setStagePrefix("");
    if (!ok) {
      cerr << "parallelSOSPUpdate failed for objective " << k << "\n";
      return 1;
    }
    trees.push_back(objOut + "/SSSPTreeUpdated.txt");
  }
  {
    auto t = chrono::steady_clock::now();
    bool ok = parallelCombinedGraph(opt.graph, trees, K, opt.source,
                                    opt.out + "/combinedGraph",
                                    opt.out + "/combinedGraph/distancesCsr.txt",
                                    opt.out + "/combinedGraph/SSSPTreeCsr.txt");
    recordStage("TOTAL_combined_graph", msSince(t));
    if (!ok) {
      cerr << "parallelCombinedGraph failed\n";
      return 1;
    }
  }
  const double endToEnd = msSince(start);

  if (!writeInstrumentationCsv(csv)) {
    cerr << "cannot write " << csv << "\n";
  }
  if (!opt.quiet) {
    printInstrumentation(cout);
  }
  const double compute = totalStageTime("_compute");
  cout << "RESULT compute_ms=" << compute << " end_to_end_ms=" << endToEnd
       << " threads=" << omp_get_max_threads() << "\n";

  if (opt.validate) {
    CsrGraph original, updated, reverse;
    ChangeBatch batch;
    if (!readCsrGraph(opt.graph, original) ||
        !readChangeBatch(opt.changes + "/insert.txt",
                         opt.changes + "/delete.txt",
                         original.numberOfObjectives, original.numberOfNodes,
                         batch) ||
        !applyChangeBatch(original, batch, updated)) {
      cerr << "validation: cannot rebuild the updated graph\n";
      return 1;
    }
    transposeCsrGraph(updated, reverse);
    bool allOk = true;
    for (int k = 0; k < K; ++k) {
      vector<long long> dist, refDist;
      vector<int> parent, refParent;
      string objOut = opt.out + "/obj" + to_string(k);
      if (!readDistances(objOut + "/distancesUpdated.txt",
                         updated.numberOfNodes, dist) ||
          !readParents(objOut + "/SSSPTreeUpdated.txt", updated.numberOfNodes,
                       parent)) {
        return 1;
      }
      dijkstraCsrGraph(updated, k, opt.source, refDist, refParent);
      TreeCheck check = checkSospTree(reverse, k, opt.source, dist, parent,
                                      refDist);
      bool ok = check.ok(false);
      allOk = allOk && ok;
      cout << "VALIDATE obj" << k << " " << (ok ? "PASS" : "FAIL") << " ("
           << check.summary() << ")\n";
    }
    if (!allOk) {
      return 1;
    }
  }
  return 0;
}
