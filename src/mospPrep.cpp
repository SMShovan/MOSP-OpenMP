/**
 * @file mospPrep.cpp
 * @brief Input preparation for bin/mosp and the benchmarks.
 *
 * Subcommands (all seeded, all outputs in the repository's text formats):
 *
 *   mtx2csr <in.mtx> <outPrefix> <K> <wmin> <wmax> <seed>
 *       SuiteSparse Matrix Market file -> CSR text. Symmetric matrices get
 *       both edge directions; self-loops and duplicate edges are dropped;
 *       every edge gets K uniform random weights in [wmin, wmax]
 *       (1 <= wmin <= wmax <= 2^31-1, 1 <= K <= 32).
 *
 *   widen <inPrefix> <outPrefix> <K> <wmin> <wmax> <seed>
 *       Copy a graph and append random objectives until it has K (the
 *       existing objectives are kept unchanged).
 *
 *   cache <csrPrefix> <binaryPath>
 *       Write the binary cache read by `mosp --cache`.
 *
 *   changes <csrPrefix> <outDir> [--changes N] [--ins PCT]
 *           [--mode uniform|targeted|reweight|increase] [--local HOPS]
 *           [--safe] [--seed S] [--source s] [--wmin a] [--wmax b]
 *       Generate <outDir>/insert.txt and <outDir>/delete.txt
 *       (see changeGenerator.h for the modes).
 *
 *   init <csrPrefix> <outDir> [--source s]
 *       Initial SOSP trees: Dijkstra per objective ->
 *       <outDir>/obj<k>/distancesOriginal.txt, SSSPTreeOriginal.txt.
 *
 *   expected <csrPrefix> <changesDir> <outDir> [--source s]
 *       Ground truth: apply the batch and run Dijkstra per objective on the
 *       updated graph -> <outDir>/obj<k>/distancesUpdated.txt,
 *       SSSPTreeUpdated.txt.
 */

#include "changeGenerator.h"
#include "csrGraph.h"
#include "dijkstra.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;

namespace {

/// Parse a decimal integer argument in [lo, hi]; prints an error if not.
bool parseIntArg(const char *text, const char *name, long long lo,
                 long long hi, int &value) {
  char *end = nullptr;
  errno = 0;
  const long long parsed = strtoll(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < lo ||
      parsed > hi) {
    cerr << "Error: " << name << " must be an integer in [" << lo << ", "
         << hi << "], got '" << text << "'.\n";
    return false;
  }
  value = static_cast<int>(parsed);
  return true;
}

/// K in [1, 32] and weights 1 <= wmin <= wmax <= 2^31-1.
bool parseWeightArgs(char **argv, int &K, int &wmin, int &wmax) {
  return parseIntArg(argv[0], "K", 1, 32, K) &&
         parseIntArg(argv[1], "wmin", 1, INT_MAX, wmin) &&
         parseIntArg(argv[2], "wmax", wmin, INT_MAX, wmax);
}

double msSince(chrono::steady_clock::time_point start) {
  return chrono::duration<double, milli>(chrono::steady_clock::now() - start)
      .count();
}

int usage() {
  cerr << "usage: mospPrep mtx2csr <in.mtx> <outPrefix> <K> <wmin> <wmax> "
          "<seed>\n"
          "       mospPrep widen <inPrefix> <outPrefix> <K> <wmin> <wmax> "
          "<seed>\n"
          "       mospPrep cache <csrPrefix> <binaryPath>\n"
          "       mospPrep changes <csrPrefix> <outDir> [--changes N] "
          "[--ins PCT]\n"
          "                [--mode uniform|targeted|reweight|increase] "
          "[--local HOPS]\n"
          "                [--safe] [--seed S] [--source s] [--wmin a] "
          "[--wmax b]\n"
          "       mospPrep init <csrPrefix> <outDir> [--source s]\n"
          "       mospPrep expected <csrPrefix> <changesDir> <outDir> "
          "[--source s]\n";
  return 2;
}

int mtxToCsr(const string &in, const string &prefix, int K, int wmin,
             int wmax, unsigned int seed) {
  FILE *file = fopen(in.c_str(), "r");
  if (file == nullptr) {
    perror(in.c_str());
    return 1;
  }
  static char line[1 << 16];
  if (fgets(line, sizeof(line), file) == nullptr) {
    fclose(file);
    return 1;
  }
  const bool symmetric = strstr(line, "symmetric") != nullptr;
  while (fgets(line, sizeof(line), file) != nullptr && line[0] == '%') {
  }
  long long rows = 0, cols = 0, entries = 0;
  if (sscanf(line, "%lld %lld %lld", &rows, &cols, &entries) != 3 ||
      rows <= 0 || rows != cols || rows > INT32_MAX) {
    cerr << "invalid Matrix Market header\n";
    fclose(file);
    return 1;
  }
  vector<uint64_t> keys;
  keys.reserve(symmetric ? 2 * entries : entries);
  for (long long i = 0; i < entries; ++i) {
    if (fgets(line, sizeof(line), file) == nullptr) {
      break;
    }
    char *p = line;
    long long a = strtoll(p, &p, 10) - 1, b = strtoll(p, &p, 10) - 1;
    if (a == b || a < 0 || b < 0 || a >= rows || b >= rows) {
      continue;
    }
    keys.push_back((static_cast<uint64_t>(a) << 32) | static_cast<uint32_t>(b));
    if (symmetric) {
      keys.push_back((static_cast<uint64_t>(b) << 32) |
                     static_cast<uint32_t>(a));
    }
  }
  fclose(file);
  sort(keys.begin(), keys.end());
  keys.erase(unique(keys.begin(), keys.end()), keys.end());

  CsrGraph graph;
  graph.numberOfNodes = static_cast<int>(rows);
  graph.numberOfObjectives = K;
  graph.rowPtr.assign(graph.numberOfNodes + 1, 0);
  graph.colInd.resize(keys.size());
  for (size_t e = 0; e < keys.size(); ++e) {
    ++graph.rowPtr[(keys[e] >> 32) + 1];
    graph.colInd[e] = static_cast<int>(keys[e] & 0xffffffffu);
  }
  for (int u = 0; u < graph.numberOfNodes; ++u) {
    graph.rowPtr[u + 1] += graph.rowPtr[u];
  }
  mt19937 rng(seed);
  uniform_int_distribution<int> weight(wmin, wmax);
  graph.weights.resize(keys.size() * K);
  for (auto &w : graph.weights) {
    w = weight(rng);
  }
  if (!writeCsrGraph(prefix, graph)) {
    return 1;
  }
  cout << "mtx2csr: n=" << graph.numberOfNodes
       << " directed edges=" << graph.numberOfEdges()
       << " symmetric=" << symmetric << "\n";
  return 0;
}

int widen(const string &in, const string &out, int K, int wmin, int wmax,
          unsigned int seed) {
  CsrGraph graph;
  if (!readCsrGraph(in, graph)) {
    return 1;
  }
  const int oldK = graph.numberOfObjectives;
  if (K < oldK) {
    cerr << "graph already has " << oldK << " objectives\n";
    return 1;
  }
  mt19937 rng(seed);
  uniform_int_distribution<int> weight(wmin, wmax);
  vector<int> widened(static_cast<size_t>(graph.numberOfEdges()) * K);
  for (int e = 0; e < graph.numberOfEdges(); ++e) {
    for (int k = 0; k < K; ++k) {
      widened[static_cast<size_t>(e) * K + k] =
          k < oldK ? graph.weight(e, k) : weight(rng);
    }
  }
  graph.weights.swap(widened);
  graph.numberOfObjectives = K;
  return writeCsrGraph(out, graph) ? 0 : 1;
}

/// Parse "--name value" style options into generator options.
bool parseChangeOptions(int argc, char **argv, int first,
                        ChangeGeneratorOptions &opt) {
  for (int i = first; i < argc; ++i) {
    string a = argv[i];
    if (a == "--safe") {
      opt.safeDeletions = true;
      continue;
    }
    if (i + 1 >= argc) {
      return false;
    }
    string v = argv[++i];
    if (a == "--changes") {
      opt.numberOfChanges = atoi(v.c_str());
    } else if (a == "--ins") {
      opt.insertionPercentage = atof(v.c_str());
    } else if (a == "--mode") {
      if (!parseChangeMode(v, opt.mode)) {
        return false;
      }
    } else if (a == "--local") {
      opt.localHops = atoi(v.c_str());
    } else if (a == "--seed") {
      opt.seed = static_cast<unsigned int>(strtoul(v.c_str(), nullptr, 10));
    } else if (a == "--source") {
      opt.source = atoi(v.c_str());
    } else if (a == "--wmin") {
      if (!parseIntArg(v.c_str(), "--wmin", 1, INT_MAX, opt.weightMin)) {
        return false;
      }
    } else if (a == "--wmax") {
      if (!parseIntArg(v.c_str(), "--wmax", 1, INT_MAX, opt.weightMax)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

int sourceOption(int argc, char **argv, int first) {
  for (int i = first; i + 1 < argc; ++i) {
    if (strcmp(argv[i], "--source") == 0) {
      return atoi(argv[i + 1]);
    }
  }
  return 0;
}

int dijkstraAll(const CsrGraph &graph, int source, const string &outDir,
                const char *distName, const char *treeName) {
  for (int k = 0; k < graph.numberOfObjectives; ++k) {
    auto t = chrono::steady_clock::now();
    vector<long long> dist;
    vector<int> parent;
    dijkstraCsrGraph(graph, k, source, dist, parent);
    string dir = outDir + "/obj" + to_string(k);
    if (!writeDistances(dir + "/" + distName, dist) ||
        !writeParents(dir + "/" + treeName, parent)) {
      return 1;
    }
    cout << "dijkstra obj" << k << ": " << msSince(t) << " ms\n";
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    return usage();
  }
  const string command = argv[1];
  auto start = chrono::steady_clock::now();
  int rc = 0;

  int K = 0, wmin = 0, wmax = 0;
  if (command == "mtx2csr" && argc == 8) {
    if (!parseWeightArgs(argv + 4, K, wmin, wmax)) {
      return 2;
    }
    rc = mtxToCsr(argv[2], argv[3], K, wmin, wmax,
                  static_cast<unsigned>(atoll(argv[7])));
  } else if (command == "widen" && argc == 8) {
    if (!parseWeightArgs(argv + 4, K, wmin, wmax)) {
      return 2;
    }
    rc = widen(argv[2], argv[3], K, wmin, wmax,
               static_cast<unsigned>(atoll(argv[7])));
  } else if (command == "cache" && argc == 4) {
    CsrGraph graph;
    rc = readCsrGraph(argv[2], graph) && saveCsrGraphBinary(argv[3], graph)
             ? 0
             : 1;
  } else if (command == "changes" && argc >= 4) {
    ChangeGeneratorOptions opt;
    if (!parseChangeOptions(argc, argv, 4, opt)) {
      return usage();
    }
    CsrGraph graph;
    ChangeBatch batch;
    string report;
    const string dir = argv[3];
    rc = readCsrGraph(argv[2], graph) &&
                 generateChangeBatch(graph, opt, batch, &report) &&
                 writeChangeBatch(batch, dir + "/insert.txt",
                                  dir + "/delete.txt")
             ? 0
             : 1;
    cout << "changes: " << report << "\n";
  } else if (command == "init" && argc >= 4) {
    CsrGraph graph;
    rc = readCsrGraph(argv[2], graph)
             ? dijkstraAll(graph, sourceOption(argc, argv, 4), argv[3],
                           "distancesOriginal.txt", "SSSPTreeOriginal.txt")
             : 1;
  } else if (command == "expected" && argc >= 5) {
    CsrGraph original, updated;
    ChangeBatch batch;
    const string changes = argv[3];
    rc = readCsrGraph(argv[2], original) &&
                 readChangeBatch(changes + "/insert.txt",
                                 changes + "/delete.txt",
                                 original.numberOfObjectives,
                                 original.numberOfNodes, batch) &&
                 applyChangeBatch(original, batch, updated)
             ? dijkstraAll(updated, sourceOption(argc, argv, 5), argv[4],
                           "distancesUpdated.txt", "SSSPTreeUpdated.txt")
             : 1;
  } else {
    return usage();
  }
  cout << command << " done in " << msSince(start) << " ms (rc=" << rc
       << ")\n";
  return rc;
}
