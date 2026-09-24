// Driver that times the original code (baseline-2026-09) on prepared
// inputs; not part of the original sources (build.sh compiles it together
// with them). It runs the MOSP steps of the original main.cpp:
//   [optional] per objective runDijkstraCSR (initial SOSP tree)
//   per objective parallelSOSPUpdate
//   parallelCombinedGraph
// and prints the stage timers (prof.h), the total wall time and,
// optionally, a comparison of the updated distances with reference files.
//
// usage: mospBench <csrPrefix> <K> <source> <changesDir> <initDir> <outDir>
//                  [--with-dijkstra] [--expected <dir>]
//   <changesDir>  insert.txt, delete.txt
//   <initDir>     obj<k>/distancesOriginal.txt, obj<k>/SSSPTreeOriginal.txt
//   --expected    obj<k>/distancesUpdated.txt (e.g. from `mospPrep expected`)
#include "dijkstra.h"
#include "parallelCombinedGraph.h"
#include "parallelSOSPUpdate.h"
#include "prof.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
using namespace std;

static bool sameFile(const string &a, const string &b, long long &mism) {
  ifstream fa(a), fb(b);
  if (!fa || !fb) return false;
  string la, lb;
  mism = 0;
  while (getline(fa, la) && getline(fb, lb))
    if (la != lb) ++mism;
  return mism == 0;
}

int main(int argc, char **argv) {
  if (argc < 7) {
    fprintf(stderr,
            "usage: mospBench <csrPrefix> <K> <source> <changesDir> "
            "<initDir> <outDir> [--with-dijkstra] [--expected <dir>]\n");
    return 1;
  }
  string csr = argv[1];
  int K = atoi(argv[2]), source = atoi(argv[3]);
  string chg = argv[4], init = argv[5], out = argv[6];
  bool withDij = false;
  string expected;
  for (int i = 7; i < argc; ++i) {
    if (!strcmp(argv[i], "--with-dijkstra")) withDij = true;
    if (!strcmp(argv[i], "--expected") && i + 1 < argc) expected = argv[++i];
  }
  auto T0 = chrono::steady_clock::now();
  vector<string> trees;
  for (int k = 0; k < K; ++k) {
    string pre = "obj" + to_string(k) + "/";
    string d0 = init + "/obj" + to_string(k);
    if (withDij) {
      StageTimer st(pre);
      st.begin("dijkstra_initial_tree_host");
      runDijkstraCSR(csr, k, source, d0 + "/distancesOriginal.txt",
                     d0 + "/SSSPTreeOriginal.txt");
    }
    string od = out + "/obj" + to_string(k);
    profPrefix() = pre;
    {
      StageTimer st(pre);
      st.begin("TOTAL_parallelSOSPUpdate");
      if (!parallelSOSPUpdate(csr, d0 + "/distancesOriginal.txt",
                              d0 + "/SSSPTreeOriginal.txt", chg + "/insert.txt",
                              chg + "/delete.txt", k, source,
                              od + "/distancesParallelUpdate.txt",
                              od + "/SSSPTreeParallelUpdate.txt")) {
        fprintf(stderr, "parallelSOSPUpdate failed\n");
        return 2;
      }
    }
    profPrefix() = "";
    trees.push_back(od + "/SSSPTreeParallelUpdate.txt");
  }
  {
    StageTimer st("");
    st.begin("TOTAL_parallelCombinedGraph");
    if (!parallelCombinedGraph(csr, trees, K, source, out + "/combinedGraph",
                               out + "/combinedGraph/distancesCsr.txt",
                               out + "/combinedGraph/SSSPTreeCsr.txt")) {
      fprintf(stderr, "combined failed\n");
      return 3;
    }
  }
  double tot = chrono::duration<double, milli>(chrono::steady_clock::now() - T0).count();
  profDump();
  printf("TOTAL_WALL_MS %.3f\n", tot);
  {
    ifstream ps("/proc/self/status");
    string l;
    while (getline(ps, l))
      if (l.rfind("VmHWM", 0) == 0) printf("HOST_PEAK_RSS %s\n", l.c_str());
  }
  if (!expected.empty()) {
    for (int k = 0; k < K; ++k) {
      long long mm = 0;
      bool ok = sameFile(expected + "/obj" + to_string(k) + "/distancesUpdated.txt",
                         out + "/obj" + to_string(k) + "/distancesParallelUpdate.txt", mm);
      printf("VALIDATE obj%d distances vs Dijkstra(updated): %s (mismatch lines=%lld)\n",
             k, ok ? "PASS" : "FAIL", mm);
    }
  }
  return 0;
}
