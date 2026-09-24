/**
 * @file changeGenerator.cpp
 * @brief Seeded change-batch generator with uniform, thesis-style targeted,
 *        re-weighting and tree-edge weight-increase modes, optional
 *        locality and connectivity-safe deletions.
 */

#include "changeGenerator.h"

#include "csrGraph.h"
#include "dijkstra.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace std;

namespace {

uint64_t edgeKey(int u, int v) {
  return (static_cast<uint64_t>(u) << 32) ^ static_cast<uint32_t>(v);
}

/// Vertices within @p hops of @p centre, ignoring edge direction.
vector<int> localBall(const CsrGraph &graph, const CsrGraph &reverse,
                      int centre, int hops) {
  vector<int> level(graph.numberOfNodes, -1);
  vector<int> ball{centre};
  level[centre] = 0;
  for (size_t i = 0; i < ball.size(); ++i) {
    int u = ball[i];
    if (level[u] == hops) {
      continue;
    }
    for (const CsrGraph *g : {&graph, &reverse}) {
      for (int e = g->rowPtr[u]; e < g->rowPtr[u + 1]; ++e) {
        int w = g->colInd[e];
        if (level[w] < 0) {
          level[w] = level[u] + 1;
          ball.push_back(w);
        }
      }
    }
  }
  return ball;
}

/// Vertices reachable from @p source.
vector<char> reachableFrom(const CsrGraph &graph, int source) {
  vector<char> seen(graph.numberOfNodes, 0);
  vector<int> queue{source};
  seen[source] = 1;
  for (size_t i = 0; i < queue.size(); ++i) {
    int u = queue[i];
    for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
      int w = graph.colInd[e];
      if (!seen[w]) {
        seen[w] = 1;
        queue.push_back(w);
      }
    }
  }
  return seen;
}

/// Remove deletions until no vertex reachable from the source in the
/// original graph becomes unreachable (at most a few rounds).
int filterUnsafeDeletions(const CsrGraph &graph, int source,
                          ChangeBatch &batch) {
  const vector<char> before = reachableFrom(graph, source);
  int rounds = 0;
  while (batch.numberOfDeletes() > 0) {
    ++rounds;
    ChangeBatch probe = batch;
    CsrGraph updated;
    applyChangeBatch(graph, probe, updated);
    const vector<char> after = reachableFrom(updated, source);
    vector<char> lost(graph.numberOfNodes, 0);
    bool anyLost = false;
    for (int v = 0; v < graph.numberOfNodes; ++v) {
      if (before[v] && !after[v]) {
        lost[v] = 1;
        anyLost = true;
      }
    }
    if (!anyLost) {
      break;
    }
    vector<int> keepFrom, keepTo;
    for (int i = 0; i < batch.numberOfDeletes(); ++i) {
      int u = batch.deleteFrom[i], v = batch.deleteTo[i];
      if (!lost[u] && !lost[v]) {
        keepFrom.push_back(u);
        keepTo.push_back(v);
      }
    }
    if (static_cast<int>(keepFrom.size()) == batch.numberOfDeletes()) {
      break; // cannot happen: a lost vertex is an endpoint of a cut edge
    }
    batch.deleteFrom.swap(keepFrom);
    batch.deleteTo.swap(keepTo);
  }
  return rounds;
}

} // namespace

bool parseChangeMode(const string &name, ChangeMode &mode) {
  if (name == "uniform") {
    mode = ChangeMode::Uniform;
  } else if (name == "targeted") {
    mode = ChangeMode::Targeted;
  } else if (name == "reweight") {
    mode = ChangeMode::Reweight;
  } else if (name == "increase") {
    mode = ChangeMode::Increase;
  } else {
    return false;
  }
  return true;
}

bool generateChangeBatch(const CsrGraph &graph,
                         const ChangeGeneratorOptions &options,
                         ChangeBatch &batch, string *report) {
  const int n = graph.numberOfNodes;
  const int K = graph.numberOfObjectives;
  batch = ChangeBatch();
  batch.numberOfObjectives = K;
  if (n < 2 || K <= 0 || options.numberOfChanges < 0 ||
      options.weightMin > options.weightMax || options.weightMin < 1 ||
      options.source < 0 || options.source >= n) {
    cout << "Error: invalid change generator options.\n";
    return false;
  }

  const ChangeMode mode = options.mode;
  const bool splitsInsertDelete =
      mode == ChangeMode::Uniform || mode == ChangeMode::Targeted;
  int insertCount = 0, deleteCount = 0, reweightCount = 0;
  if (splitsInsertDelete) {
    insertCount = static_cast<int>(llround(options.numberOfChanges *
                                           options.insertionPercentage / 100.0));
    insertCount = max(0, min(insertCount, options.numberOfChanges));
    deleteCount = options.numberOfChanges - insertCount;
  } else {
    reweightCount = options.numberOfChanges;
  }

  mt19937 rng(options.seed);
  uniform_int_distribution<int> weightDist(options.weightMin,
                                           options.weightMax);

  // --- Candidate vertices (all, or a local ball) -------------------------
  CsrGraph reverse;
  vector<int> region;
  vector<char> inRegion;
  int centre = -1;
  if (options.localHops > 0) {
    transposeCsrGraph(graph, reverse);
    centre = uniform_int_distribution<int>(0, n - 1)(rng);
    region = localBall(graph, reverse, centre, options.localHops);
    inRegion.assign(n, 0);
    for (int v : region) {
      inRegion[v] = 1;
    }
    if (region.size() < 2) {
      cout << "Error: local region around vertex " << centre
           << " has fewer than 2 vertices.\n";
      return false;
    }
  }
  auto randomVertex = [&]() {
    if (region.empty()) {
      return uniform_int_distribution<int>(0, n - 1)(rng);
    }
    return region[uniform_int_distribution<size_t>(0, region.size() - 1)(rng)];
  };

  // --- Existing edges eligible for deletion / re-weighting ---------------
  // In the order of the original generator: row order, first occurrence
  // of each (u,v), no self-loops.
  auto collectEdges = [&](bool treeOnly, const vector<vector<int>> &trees) {
    vector<pair<int, int>> edges;
    if (treeOnly) {
      unordered_set<uint64_t> seen;
      for (int v = 0; v < n; ++v) {
        if (!inRegion.empty() && !inRegion[v]) {
          continue;
        }
        for (const auto &tree : trees) {
          int p = tree[v];
          if (p >= 0 && seen.insert(edgeKey(p, v)).second) {
            edges.push_back({p, v});
          }
        }
      }
      return edges;
    }
    unordered_set<uint64_t> seen;
    for (int u = 0; u < n; ++u) {
      if (!inRegion.empty() && !inRegion[u]) {
        continue;
      }
      for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
        int v = graph.colInd[e];
        if (u == v || (!inRegion.empty() && !inRegion[v])) {
          continue;
        }
        if (seen.insert(edgeKey(u, v)).second) {
          edges.push_back({u, v});
        }
      }
    }
    return edges;
  };

  vector<vector<int>> trees;
  if (mode == ChangeMode::Targeted || mode == ChangeMode::Increase) {
    for (int k = 0; k < K; ++k) {
      vector<long long> dist;
      vector<int> parent;
      dijkstraCsrGraph(graph, k, options.source, dist, parent);
      trees.push_back(move(parent));
    }
  }

  // --- Insertions ----------------------------------------------------------
  vector<pair<int, int>> insertEdges;
  insertEdges.reserve(insertCount);
  while (static_cast<int>(insertEdges.size()) < insertCount) {
    int u = randomVertex();
    int v = randomVertex();
    if (u != v) {
      insertEdges.push_back({u, v});
    }
  }

  // --- Deletions / re-weighted edges --------------------------------------
  vector<pair<int, int>> deleteEdges, reweightEdges;
  if (deleteCount > 0 || reweightCount > 0) {
    const bool treeOnly =
        mode == ChangeMode::Targeted || mode == ChangeMode::Increase;
    vector<pair<int, int>> candidates = collectEdges(treeOnly, trees);
    const int wanted = max(deleteCount, reweightCount);
    if (candidates.empty()) {
      cout << "Error: No existing edges available for the batch.\n";
      return false;
    }
    vector<pair<int, int>> &chosen =
        deleteCount > 0 ? deleteEdges : reweightEdges;
    if (mode == ChangeMode::Uniform || mode == ChangeMode::Reweight) {
      // Sampling with replacement, as generateChangedEdges(duplicate=true).
      uniform_int_distribution<int> pick(
          0, static_cast<int>(candidates.size()) - 1);
      for (int i = 0; i < wanted; ++i) {
        chosen.push_back(candidates[pick(rng)]);
      }
    } else {
      // Distinct tree edges.
      shuffle(candidates.begin(), candidates.end(), rng);
      chosen.assign(candidates.begin(),
                    candidates.begin() +
                        min<size_t>(candidates.size(), wanted));
    }
  }

  // --- Weights ------------------------------------------------------------
  vector<int> belowAverage(K, options.weightMax);
  if (mode == ChangeMode::Targeted && graph.numberOfEdges() > 0) {
    for (int k = 0; k < K; ++k) {
      double sum = 0;
      for (int e = 0; e < graph.numberOfEdges(); ++e) {
        sum += graph.weight(e, k);
      }
      int average = static_cast<int>(sum / graph.numberOfEdges());
      belowAverage[k] = max(options.weightMin, average - 1);
    }
  }
  for (const auto &edge : insertEdges) {
    batch.insertFrom.push_back(edge.first);
    batch.insertTo.push_back(edge.second);
    for (int k = 0; k < K; ++k) {
      int w = mode == ChangeMode::Targeted
                  ? uniform_int_distribution<int>(options.weightMin,
                                                  belowAverage[k])(rng)
                  : weightDist(rng);
      batch.insertWeights.push_back(w);
    }
  }
  for (const auto &edge : reweightEdges) {
    int u = edge.first, v = edge.second;
    int e = graph.rowPtr[u];
    while (graph.colInd[e] != v) {
      ++e;
    }
    batch.insertFrom.push_back(u);
    batch.insertTo.push_back(v);
    for (int k = 0; k < K; ++k) {
      // Increases saturate at the largest valid weight (2^31-1).
      int w = mode == ChangeMode::Increase
                  ? static_cast<int>(min<long long>(
                        INT_MAX,
                        static_cast<long long>(graph.weight(e, k)) +
                            uniform_int_distribution<int>(
                                1, options.weightMax)(rng)))
                  : weightDist(rng);
      batch.insertWeights.push_back(w);
    }
  }
  for (const auto &edge : deleteEdges) {
    batch.deleteFrom.push_back(edge.first);
    batch.deleteTo.push_back(edge.second);
  }

  int safeRounds = 0;
  const int requestedDeletes = batch.numberOfDeletes();
  if (options.safeDeletions) {
    safeRounds = filterUnsafeDeletions(graph, options.source, batch);
  }

  if (report != nullptr) {
    ostringstream out;
    out << "inserts=" << batch.numberOfInserts() - (int)reweightEdges.size()
        << " reweights=" << reweightEdges.size()
        << " deletes=" << batch.numberOfDeletes();
    if (options.safeDeletions) {
      out << " (safe: kept " << batch.numberOfDeletes() << " of "
          << requestedDeletes << " after " << safeRounds << " rounds)";
    }
    if (centre >= 0) {
      out << " local centre=" << centre << " hops=" << options.localHops
          << " region=" << region.size() << " vertices";
    }
    *report = out.str();
  }
  return true;
}

bool writeChangeBatch(const ChangeBatch &batch, const string &insertPath,
                      const string &deletePath) {
  for (const string &path : {insertPath, deletePath}) {
    filesystem::path parent = filesystem::path(path).parent_path();
    if (!parent.empty()) {
      filesystem::create_directories(parent);
    }
  }
  ofstream insertFile(insertPath), deleteFile(deletePath);
  if (!insertFile.is_open() || !deleteFile.is_open()) {
    cout << "Error: Could not open change-edge output files for writing.\n";
    return false;
  }
  const int K = batch.numberOfObjectives;
  for (int i = 0; i < batch.numberOfInserts(); ++i) {
    insertFile << batch.insertFrom[i] << " " << batch.insertTo[i];
    for (int k = 0; k < K; ++k) {
      insertFile << " " << batch.insertWeights[static_cast<size_t>(i) * K + k];
    }
    insertFile << "\n";
  }
  for (int i = 0; i < batch.numberOfDeletes(); ++i) {
    deleteFile << batch.deleteFrom[i] << " " << batch.deleteTo[i] << "\n";
  }
  return insertFile.good() && deleteFile.good();
}
