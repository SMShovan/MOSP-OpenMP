/**
 * @file mospTest.cpp
 * @brief Oracle tests for the SOSP update and the combined-graph step.
 *
 * Every case builds a seeded graph, initial SOSP trees (Dijkstra) and a
 * change batch, runs the update through the public file-based API and
 * checks the result against Dijkstra on the updated graph:
 *   - distances equal (unreachable vertices = INF),
 *   - parent consistency: an existing edge (p,v) with d[p]+w(p,v) == d[v],
 *   - canonical parents: the lowest id among equal-distance parents, so
 *     every tree equals the (canonical) Dijkstra tree exactly,
 *   - determinism: running the parallel update twice gives identical files.
 *
 * Change sets: uniform (connectivity-safe and unsafe), deletions only,
 * disconnecting deletions, insertions only, tree-edge weight increases,
 * re-weighting, targeted (thesis) and local batches. Graphs: random
 * directed graphs (the repository's generator) and road-like grids.
 *
 * Usage: mospTest [--seed S] [--work DIR]   (exit code 0 = all passed)
 */

#include "changeGenerator.h"
#include "csrGraph.h"
#include "dijkstra.h"
#include "generateChangedEdges.h"
#include "generateGraphCSR.h"
#include "parallelCombinedGraph.h"
#include "parallelSOSPUpdate.h"
#include "sequentialSOSPUpdate.h"
#include "updateGraphCSR.h"
#include "validation.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

using namespace std;

namespace {

string g_work = "mospTest-work";
int g_failures = 0;

// ============================================================================
// Graph builders
// ============================================================================

/// Road-like graph: a width x height grid, both directions, K weights in
/// [1, maxWeight]; a fraction of the grid edges is dropped (both ways) so
/// the graph has dead ends and long detours.
CsrGraph gridGraph(int width, int height, int K, int maxWeight,
                   double dropFraction, unsigned int seed) {
  mt19937 rng(seed);
  uniform_int_distribution<int> weight(1, maxWeight);
  uniform_real_distribution<double> coin(0.0, 1.0);
  const int n = width * height;
  vector<vector<pair<int, vector<int>>>> rows(n);
  auto id = [&](int x, int y) { return y * width + x; };
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      for (auto [dx, dy] : {pair<int, int>{1, 0}, pair<int, int>{0, 1}}) {
        int nx = x + dx, ny = y + dy;
        if (nx >= width || ny >= height || coin(rng) < dropFraction) {
          continue;
        }
        int u = id(x, y), v = id(nx, ny);
        vector<int> w1(K), w2(K);
        for (int k = 0; k < K; ++k) {
          w1[k] = weight(rng);
          w2[k] = weight(rng);
        }
        rows[u].push_back({v, w1});
        rows[v].push_back({u, w2});
      }
    }
  }
  CsrGraph graph;
  graph.numberOfNodes = n;
  graph.numberOfObjectives = K;
  graph.rowPtr.assign(n + 1, 0);
  for (int u = 0; u < n; ++u) {
    graph.rowPtr[u + 1] = graph.rowPtr[u] + static_cast<int>(rows[u].size());
    for (auto &edge : rows[u]) {
      graph.colInd.push_back(edge.first);
      graph.weights.insert(graph.weights.end(), edge.second.begin(),
                           edge.second.end());
    }
  }
  return graph;
}

/// Random directed graph from the repository's generator (spanning chain
/// 0 -> 1 -> ... -> n-1 plus random edges).
CsrGraph randomGraph(const string &dir, int n, int m, int K, int maxWeight,
                     unsigned int seed) {
  CsrGraph graph;
  generateGraphCSR(n, m, true, dir + "/random/graphCsr", K, 1, maxWeight, seed);
  readCsrGraph(dir + "/random/graphCsr", graph);
  return graph;
}

// ============================================================================
// Checks
// ============================================================================

void report(const string &name, bool ok, const string &detail) {
  if (!ok) {
    ++g_failures;
    cout << "  FAIL " << name << ": " << detail << "\n";
  }
}

struct CaseFiles {
  string dir, graph, insert, remove, init;
};

/// Write graph, change batch and initial trees for one case.
CaseFiles writeCase(const string &dir, const CsrGraph &graph,
                    const ChangeBatch &batch, int source) {
  CaseFiles files{dir, dir + "/graph/graphCsr", dir + "/changes/insert.txt",
                  dir + "/changes/delete.txt", dir + "/init"};
  writeCsrGraph(files.graph, graph);
  writeChangeBatch(batch, files.insert, files.remove);
  for (int k = 0; k < graph.numberOfObjectives; ++k) {
    vector<long long> dist;
    vector<int> parent;
    dijkstraCsrGraph(graph, k, source, dist, parent);
    string obj = files.init + "/obj" + to_string(k);
    writeDistances(obj + "/distances.txt", dist);
    writeParents(obj + "/tree.txt", parent);
  }
  return files;
}

/// Output directory of one implementation ("label") for objective k.
string outputDir(const CaseFiles &files, const string &label, int k) {
  return files.dir + "/" + label + "/obj" + to_string(k);
}

using UpdateFunction = function<bool(const CaseFiles &, int objective,
                                     int source, const string &distOut,
                                     const string &treeOut)>;

/// Run one update implementation for every objective and validate.
/// Returns the updated trees (parents) for the combined-graph check.
bool runAndCheck(const string &label, const UpdateFunction &update,
                 const CaseFiles &files, const CsrGraph &updated,
                 const CsrGraph &reverse, int source,
                 vector<string> *treePaths = nullptr) {
  bool allOk = true;
  for (int k = 0; k < updated.numberOfObjectives; ++k) {
    string out = outputDir(files, label, k);
    string distOut = out + "/distances.txt", treeOut = out + "/tree.txt";
    if (!update(files, k, source, distOut, treeOut)) {
      report(label, false, files.dir + " obj" + to_string(k) + ": call failed");
      allOk = false;
      continue;
    }
    vector<long long> dist, refDist;
    vector<int> parent, refParent;
    readDistances(distOut, updated.numberOfNodes, dist);
    readParents(treeOut, updated.numberOfNodes, parent);
    dijkstraCsrGraph(updated, k, source, refDist, refParent);
    TreeCheck check =
        checkSospTree(reverse, k, source, dist, parent, refDist, &refParent);
    bool ok = check.ok(true);
    report(label, ok, files.dir + " obj" + to_string(k) + ": " + check.summary());
    allOk = allOk && ok;
    if (treePaths != nullptr) {
      treePaths->push_back(treeOut);
    }
  }
  return allOk;
}

/// Host reference for the combined graph: edge (p,v) is in the combined
/// graph iff p is the parent of v in some tree T_i; its weight is
/// L * (K + 1) - sum_i L / Pref_i over those trees (K + 1 - membership
/// for Pref = 1s).
CsrGraph combinedReference(const vector<vector<int>> &parents, int source,
                           const vector<int> &pref) {
  const int K = static_cast<int>(parents.size());
  const int n = static_cast<int>(parents[0].size());
  CsrGraph combined;
  combined.numberOfNodes = n;
  combined.numberOfObjectives = 1;
  const long long scale = preferenceScale(pref, K);
  vector<vector<pair<int, int>>> rows(n);
  for (int v = 0; v < n; ++v) {
    if (v == source) {
      continue;
    }
    vector<int> seen;
    for (int k = 0; k < K; ++k) {
      int p = parents[k][v];
      if (p < 0 || find(seen.begin(), seen.end(), p) != seen.end()) {
        continue;
      }
      seen.push_back(p);
      unsigned int mask = 0;
      for (int j = 0; j < K; ++j) {
        mask |= parents[j][v] == p ? 1u << j : 0u;
      }
      rows[p].push_back(
          {v, static_cast<int>(combinedEdgeWeight(mask, pref, K, scale))});
    }
  }
  combined.rowPtr.assign(n + 1, 0);
  for (int u = 0; u < n; ++u) {
    combined.rowPtr[u + 1] = combined.rowPtr[u] + static_cast<int>(rows[u].size());
    for (auto &edge : rows[u]) {
      combined.colInd.push_back(edge.first);
      combined.weights.push_back(edge.second);
    }
  }
  return combined;
}

void checkCombined(const CaseFiles &files, const vector<string> &trees,
                   int n, int source, const vector<int> &pref) {
  const int K = static_cast<int>(trees.size());
  string dir = files.dir + "/combined" + to_string(pref.empty() ? 0 : pref[0]);
  if (!parallelCombinedGraph(files.graph, trees, K, source, dir,
                             dir + "/distances.txt", dir + "/tree.txt",
                             pref)) {
    report("combined", false, files.dir + ": call failed");
    return;
  }
  vector<vector<int>> parents(K);
  for (int k = 0; k < K; ++k) {
    readParents(trees[k], n, parents[k]);
  }
  CsrGraph combined = combinedReference(parents, source, pref), reverse;
  transposeCsrGraph(combined, reverse);
  vector<long long> dist, refDist;
  vector<int> parent, refParent;
  readDistances(dir + "/distances.txt", n, dist);
  readParents(dir + "/tree.txt", n, parent);
  dijkstraCsrGraph(combined, 0, source, refDist, refParent);
  TreeCheck check =
      checkSospTree(reverse, 0, source, dist, parent, refDist, &refParent);
  report("combined", check.ok(true), files.dir + ": " + check.summary());
}

// ============================================================================
// Test sets
// ============================================================================

/// Run the parallel update a second time; outputs must be identical.
void checkDeterminism(const CaseFiles &files, const string &firstLabel, int n,
                      int K, int source) {
  for (int k = 0; k < K; ++k) {
    string obj = files.init + "/obj" + to_string(k);
    string first = outputDir(files, firstLabel, k);
    string again = outputDir(files, "rerun", k);
    parallelSOSPUpdate(files.graph, obj + "/distances.txt", obj + "/tree.txt",
                       files.insert, files.remove, k, source,
                       again + "/distances.txt", again + "/tree.txt");
    vector<int> firstTree, againTree;
    vector<long long> firstDist, againDist;
    readParents(first + "/tree.txt", n, firstTree);
    readParents(again + "/tree.txt", n, againTree);
    readDistances(first + "/distances.txt", n, firstDist);
    readDistances(again + "/distances.txt", n, againDist);
    report("determinism", firstTree == againTree && firstDist == againDist,
           files.dir + " obj" + to_string(k) + ": two runs differ");
  }
}

struct ChangeSet {
  string name;
  ChangeGeneratorOptions options;
  /// Optional custom batch (overrides the generator).
  function<void(const CsrGraph &, unsigned int, ChangeBatch &)> custom;
};

/// Delete every in-edge of a few random vertices: they (and whatever is
/// only reachable through them) become unreachable.
void disconnectingBatch(const CsrGraph &graph, unsigned int seed,
                        ChangeBatch &batch) {
  batch = ChangeBatch();
  batch.numberOfObjectives = graph.numberOfObjectives;
  CsrGraph reverse;
  transposeCsrGraph(graph, reverse);
  mt19937 rng(seed);
  uniform_int_distribution<int> pick(1, graph.numberOfNodes - 1);
  const int victims = max(1, graph.numberOfNodes / 50);
  for (int i = 0; i < victims; ++i) {
    int v = pick(rng);
    for (int e = reverse.rowPtr[v]; e < reverse.rowPtr[v + 1]; ++e) {
      batch.deleteFrom.push_back(reverse.colInd[e]);
      batch.deleteTo.push_back(v);
    }
  }
}

vector<ChangeSet> changeSets(int numberOfChanges) {
  auto make = [&](const string &name, ChangeMode mode, double ins, int local,
                  bool safe) {
    ChangeSet set;
    set.name = name;
    set.options.numberOfChanges = numberOfChanges;
    set.options.mode = mode;
    set.options.insertionPercentage = ins;
    set.options.localHops = local;
    set.options.safeDeletions = safe;
    set.options.weightMax = 50;
    return set;
  };
  vector<ChangeSet> sets = {
      make("uniform-safe", ChangeMode::Uniform, 50, 0, true),
      make("uniform-unsafe", ChangeMode::Uniform, 50, 0, false),
      make("deletions-only", ChangeMode::Uniform, 0, 0, false),
      make("insertions-only", ChangeMode::Uniform, 100, 0, false),
      make("tree-weight-increases", ChangeMode::Increase, 0, 0, false),
      make("reweight", ChangeMode::Reweight, 0, 0, false),
      make("targeted", ChangeMode::Targeted, 50, 0, false),
      make("local", ChangeMode::Uniform, 50, 4, false),
      make("local-reweight", ChangeMode::Reweight, 0, 4, false),
  };
  ChangeSet cut;
  cut.name = "disconnecting";
  cut.custom = disconnectingBatch;
  sets.push_back(cut);
  return sets;
}

void runSosp(unsigned int seed) {
  struct GraphSpec {
    string name;
    function<CsrGraph(const string &, unsigned int)> build;
    int changes;
  };
  vector<GraphSpec> graphs = {
      {"random-tiny",
       [](const string &d, unsigned int s) { return randomGraph(d, 12, 30, 3, 50, s); },
       6},
      {"random-small",
       [](const string &d, unsigned int s) { return randomGraph(d, 60, 240, 2, 50, s); },
       20},
      {"random-medium",
       [](const string &d, unsigned int s) { return randomGraph(d, 800, 3200, 3, 100, s); },
       200},
      {"grid-small",
       [](const string &, unsigned int s) { return gridGraph(8, 8, 2, 50, 0.1, s); },
       12},
      {"grid-medium",
       [](const string &, unsigned int s) { return gridGraph(40, 30, 3, 100, 0.15, s); },
       150},
  };
  const int source = 0;
  int cases = 0;
  for (const auto &spec : graphs) {
    for (int rep = 0; rep < 3; ++rep) {
      unsigned int caseSeed = seed * 1000u + static_cast<unsigned>(rep) * 97u +
                              static_cast<unsigned>(spec.name.size());
      string base = g_work + "/" + spec.name + "_" + to_string(rep);
      CsrGraph graph = spec.build(base, caseSeed);
      for (const auto &set : changeSets(spec.changes)) {
        ChangeBatch batch;
        if (set.custom) {
          set.custom(graph, caseSeed, batch);
        } else {
          ChangeGeneratorOptions options = set.options;
          options.seed = caseSeed;
          options.source = source;
          if (!generateChangeBatch(graph, options, batch)) {
            continue; // e.g. too few edges in a local region
          }
        }
        string dir = base + "/" + set.name;
        CaseFiles files = writeCase(dir, graph, batch, source);
        CsrGraph updated, reverse;
        ChangeBatch copy = batch;
        applyChangeBatch(graph, copy, updated);
        transposeCsrGraph(updated, reverse);

        auto parallel = [](const CaseFiles &f, int k, int s, const string &d,
                           const string &t) {
          string obj = f.init + "/obj" + to_string(k);
          return parallelSOSPUpdate(f.graph, obj + "/distances.txt",
                                    obj + "/tree.txt", f.insert, f.remove, k, s,
                                    d, t);
        };
        auto sequential = [](const CaseFiles &f, int k, int s, const string &d,
                             const string &t) {
          string obj = f.init + "/obj" + to_string(k);
          return sequentialSOSPUpdate(f.graph, obj + "/distances.txt",
                                      obj + "/tree.txt", f.insert, f.remove, k,
                                      s, d, t);
        };
        vector<string> trees;
        runAndCheck("parallel/" + set.name, parallel, files, updated, reverse,
                    source, &trees);
        runAndCheck("sequential/" + set.name, sequential, files, updated,
                    reverse, source);
        checkDeterminism(files, "parallel/" + set.name, graph.numberOfNodes,
                         graph.numberOfObjectives, source);
        if (static_cast<int>(trees.size()) == graph.numberOfObjectives) {
          // Default Pref (all 1s) and a skewed Pref = (K+1, 1, K+1, ...).
          checkCombined(files, trees, graph.numberOfNodes, source, {});
          vector<int> skewed(graph.numberOfObjectives, graph.numberOfObjectives + 1);
          skewed[graph.numberOfObjectives > 1 ? 1 : 0] = 1;
          checkCombined(files, trees, graph.numberOfNodes, source, skewed);
        }
        ++cases;
      }
    }
  }
  cout << "sosp: " << cases << " cases (parallel, sequential, combined)\n";
}

/// The worked example of thesis Ch. 4 (Fig. "Finding a single MOSP"):
/// three SOSP updates, then the combined graph with Pref = {4,1,4} and
/// {4,4,1}. Vertices u1..u7 are 0..6, source u1.
///
/// The edge u3->u6 (4,2,8) drawn in the preliminaries figure is omitted: with
/// it, the objective-1 tree of the example (u6 reached through u5 at 11)
/// would not be a shortest-path tree (u1->u3->u6 costs 8). Without it the
/// three updated trees are exactly those of sub-figures (a)-(c).
void runThesisExample(unsigned int) {
  const int n = 7, K = 3, source = 0;
  struct E { int u, v, w1, w2, w3; };
  vector<E> edges = {{0, 1, 2, 1, 5},   {0, 2, 4, 1, 1},  {2, 1, 10, 15, 2},
                     {1, 3, 2, 4, 2},   {2, 3, 5, 16, 3}, {3, 4, 1, 1, 1},
                     {4, 1, 4, 3, 2},   {4, 5, 1, 2, 2},  {4, 6, 5, 6, 2},
                     {5, 6, 1, 1, 1}};
  stable_sort(edges.begin(), edges.end(),
              [](const E &a, const E &b) { return a.u < b.u; });
  CsrGraph graph;
  graph.numberOfNodes = n;
  graph.numberOfObjectives = K;
  graph.rowPtr.assign(n + 1, 0);
  for (const auto &e : edges) {
    ++graph.rowPtr[e.u + 1];
  }
  for (int u = 0; u < n; ++u) {
    graph.rowPtr[u + 1] += graph.rowPtr[u];
  }
  for (const auto &e : edges) { // now in row order
    graph.colInd.push_back(e.v);
    graph.weights.insert(graph.weights.end(), {e.w1, e.w2, e.w3});
  }
  ChangeBatch batch;
  batch.numberOfObjectives = K;
  batch.deleteFrom = {1, 4}; // (u2,u4), (u5,u2)
  batch.deleteTo = {3, 1};
  batch.insertFrom = {3, 1}; // (u4,u6):(10,2,12), (u2,u6):(12,1,14)
  batch.insertTo = {5, 5};
  batch.insertWeights = {10, 2, 12, 12, 1, 14};

  string dir = g_work + "/thesis-example";
  CaseFiles files = writeCase(dir, graph, batch, source);
  CsrGraph updated;
  ChangeBatch copy = batch;
  applyChangeBatch(graph, copy, updated);

  // Updated SOSP trees of sub-figures (a), (b), (c).
  const vector<vector<int>> expectedTrees = {{-1, 0, 0, 2, 3, 4, 5},
                                             {-1, 0, 0, 2, 3, 1, 5},
                                             {-1, 2, 0, 2, 3, 4, 4}};
  vector<string> trees;
  for (int k = 0; k < K; ++k) {
    string obj = files.init + "/obj" + to_string(k);
    string out = outputDir(files, "parallel", k);
    parallelSOSPUpdate(files.graph, obj + "/distances.txt", obj + "/tree.txt",
                       files.insert, files.remove, k, source,
                       out + "/distances.txt", out + "/tree.txt");
    vector<int> parent;
    readParents(out + "/tree.txt", n, parent);
    report("thesis-example", parent == expectedTrees[k],
           "objective " + to_string(k + 1) + " tree differs from the thesis");
    trees.push_back(out + "/tree.txt");
  }

  struct Expectation {
    vector<int> pref;
    vector<int> pathToU7;           // u7, its parent, ... , u1
    vector<long long> costOfU7;     // original objective values
  };
  const vector<Expectation> expectations = {
      {{4, 1, 4}, {6, 5, 1, 0}, {15, 3, 20}},     // sub-figures (d), (e)
      {{4, 4, 1}, {6, 4, 3, 2, 0}, {15, 24, 7}},  // sub-figures (f), (g)
  };
  for (const auto &expect : expectations) {
    string out = dir + "/combined_" + to_string(expect.pref[0]) +
                 to_string(expect.pref[1]) + to_string(expect.pref[2]);
    parallelCombinedGraph(files.graph, trees, K, source, out,
                          out + "/distances.txt", out + "/tree.txt",
                          expect.pref);
    vector<int> parent;
    readParents(out + "/tree.txt", n, parent);
    vector<int> path{6};
    while (parent[path.back()] >= 0) {
      path.push_back(parent[path.back()]);
    }
    vector<long long> costs;
    mospPathCosts(updated, parent, source, costs);
    vector<long long> costOfU7(costs.begin() + 6 * K, costs.begin() + 7 * K);
    string label = "Pref={" + to_string(expect.pref[0]) + "," +
                   to_string(expect.pref[1]) + "," + to_string(expect.pref[2]) +
                   "}";
    report("thesis-example", path == expect.pathToU7,
           label + ": MOSP path to u7 differs from the thesis");
    report("thesis-example", costOfU7 == expect.costOfU7,
           label + ": MOSP cost of u7 differs from the thesis");
  }
  cout << "thesis-example: 1 case (3 trees, 2 preference vectors)\n";
}

/// Uniform generator mode reproduces generateChangedEdges() exactly.
void runGeneratorEquivalence(unsigned int seed) {
  int cases = 0;
  for (int rep = 0; rep < 5; ++rep) {
    unsigned int s = seed * 31u + rep;
    string dir = g_work + "/generator_" + to_string(rep);
    CsrGraph graph = randomGraph(dir, 30 + 10 * rep, 120 + 40 * rep, 2, 40, s);
    generateChangedEdges(1, 40, 2, graph.numberOfNodes, 25, 60, 40, true, true,
                         true, false, dir + "/random/graphCsr",
                         dir + "/old/insert.txt", dir + "/old/delete.txt", s);
    ChangeGeneratorOptions options;
    options.numberOfChanges = 25;
    options.insertionPercentage = 60;
    options.weightMax = 40;
    options.seed = s;
    ChangeBatch batch, old;
    generateChangeBatch(graph, options, batch);
    readChangeBatch(dir + "/old/insert.txt", dir + "/old/delete.txt", 2,
                    graph.numberOfNodes, old);
    bool same = batch.insertFrom == old.insertFrom &&
                batch.insertTo == old.insertTo &&
                batch.insertWeights == old.insertWeights &&
                batch.deleteFrom == old.deleteFrom &&
                batch.deleteTo == old.deleteTo;
    report("generator", same, dir + ": uniform mode differs from generateChangedEdges");
    ++cases;
  }
  cout << "generator: " << cases << " cases\n";
}

/// applyChangeBatch() produces the same graph as updateGraphCSR().
void runApplyEquivalence(unsigned int seed) {
  int cases = 0;
  for (int rep = 0; rep < 5; ++rep) {
    unsigned int s = seed * 17u + rep;
    string dir = g_work + "/apply_" + to_string(rep);
    CsrGraph graph = randomGraph(dir, 25, 90, 3, 30, s);
    generateChangedEdges(1, 30, 3, graph.numberOfNodes, 30, 50, 50, true, true,
                         true, false, dir + "/random/graphCsr",
                         dir + "/c/insert.txt", dir + "/c/delete.txt", s);
    updateGraphCSR(dir + "/random/graphCsr", dir + "/updated/graphCsr",
                   dir + "/c/insert.txt", dir + "/c/delete.txt", true);
    ChangeBatch batch;
    CsrGraph mine, theirs;
    readChangeBatch(dir + "/c/insert.txt", dir + "/c/delete.txt", 3,
                    graph.numberOfNodes, batch);
    applyChangeBatch(graph, batch, mine);
    readCsrGraph(dir + "/updated/graphCsr", theirs);
    auto edges = [](const CsrGraph &g) {
      vector<tuple<int, int, vector<int>>> list;
      for (int u = 0; u < g.numberOfNodes; ++u) {
        for (int e = g.rowPtr[u]; e < g.rowPtr[u + 1]; ++e) {
          vector<int> w(g.weights.begin() + static_cast<size_t>(e) * g.numberOfObjectives,
                        g.weights.begin() + static_cast<size_t>(e + 1) * g.numberOfObjectives);
          list.emplace_back(u, g.colInd[e], w);
        }
      }
      sort(list.begin(), list.end());
      return list;
    };
    report("apply", edges(mine) == edges(theirs),
           dir + ": applyChangeBatch differs from updateGraphCSR");
    ++cases;
  }
  cout << "apply: " << cases << " cases\n";
}

} // namespace

int main(int argc, char **argv) {
  unsigned int seed = 1;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
      seed = static_cast<unsigned int>(strtoul(argv[++i], nullptr, 10));
    } else if (!strcmp(argv[i], "--work") && i + 1 < argc) {
      g_work = argv[++i];
    } else {
      cerr << "usage: mospTest [--seed S] [--work DIR]\n";
      return 2;
    }
  }
  // Silence the chatty library calls; results are reported below.
  streambuf *saved = cout.rdbuf();
  filesystem::remove_all(g_work);
  cout << "mospTest seed " << seed << "\n";

  auto quiet = [&](const function<void(unsigned int)> &test) {
    ostringstream sink;
    cout.rdbuf(sink.rdbuf());
    test(seed);
    cout.rdbuf(saved);
    // Forward summary and failure lines only.
    istringstream lines(sink.str());
    string line;
    while (getline(lines, line)) {
      if (line.rfind("  FAIL", 0) == 0 || line.find(" cases") != string::npos) {
        cout << line << "\n";
      }
    }
  };
  quiet(runThesisExample);
  quiet(runGeneratorEquivalence);
  quiet(runApplyEquivalence);
  quiet(runSosp);

  cout << (g_failures == 0 ? "=== mospTest: all checks passed ===\n"
                           : "=== mospTest: " + to_string(g_failures) +
                                 " checks FAILED ===\n");
  return g_failures == 0 ? 0 : 1;
}
