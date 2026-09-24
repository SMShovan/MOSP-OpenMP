/**
 * @file mospUpdate.cpp
 * @brief In-memory MOSP update with OpenMP: shared topology, K SOSP updates
 *        and the combined graph with buffers allocated once per run.
 */

#include "mospUpdate.h"

#include "csrGraph.h"
#include "stageTimer.h"

#include <omp.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace std;

namespace {

double msSince(chrono::steady_clock::time_point start) {
  return chrono::duration<double, milli>(chrono::steady_clock::now() - start)
      .count();
}

/// Out- and in-edge CSR of the updated graph with one weight column per
/// objective (objective-major), shared by all objectives.
struct HostGraph {
  int n = 0, m = 0;
  vector<int> outWeights;               ///< K * m
  vector<int> inRowPtr, inColInd, inWeights; ///< reverse graph, K * m

  HostCsr out(const CsrGraph &graph, int k) const {
    HostCsr view;
    view.numberOfNodes = n;
    view.numberOfEdges = m;
    view.rowPtr = graph.rowPtr.data();
    view.colInd = graph.colInd.data();
    view.weights = outWeights.data() + static_cast<size_t>(k) * m;
    return view;
  }
  HostCsr in(int k) const {
    HostCsr view;
    view.numberOfNodes = n;
    view.numberOfEdges = m;
    view.rowPtr = inRowPtr.data();
    view.colInd = inColInd.data();
    view.weights = inWeights.data() + static_cast<size_t>(k) * m;
    return view;
  }
};

/// Weight columns and the reverse graph (count, prefix sum, parallel fill).
void buildHostGraph(const CsrGraph &graph, int K, HostGraph &host) {
  const int n = graph.numberOfNodes;
  const int m = graph.numberOfEdges();
  const int KG = graph.numberOfObjectives;
  host.n = n;
  host.m = m;
  host.outWeights.resize(static_cast<size_t>(m) * K);
#pragma omp parallel for schedule(static)
  for (int e = 0; e < m; ++e) {
    for (int k = 0; k < K; ++k) {
      host.outWeights[static_cast<size_t>(k) * m + e] =
          graph.weights[static_cast<size_t>(e) * KG + k];
    }
  }
  vector<int> cursor(static_cast<size_t>(n) + 1, 0);
#pragma omp parallel for schedule(static)
  for (int e = 0; e < m; ++e) {
    __atomic_fetch_add(&cursor[graph.colInd[e]], 1, __ATOMIC_RELAXED);
  }
  host.inRowPtr.resize(static_cast<size_t>(n) + 1);
  host.inRowPtr[0] = 0;
  partial_sum(cursor.begin(), cursor.end() - 1, host.inRowPtr.begin() + 1);
  copy(host.inRowPtr.begin(), host.inRowPtr.end() - 1, cursor.begin());
  host.inColInd.resize(m);
  host.inWeights.resize(static_cast<size_t>(m) * K);
#pragma omp parallel for schedule(dynamic, 1024)
  for (int u = 0; u < n; ++u) {
    for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
      int position =
          __atomic_fetch_add(&cursor[graph.colInd[e]], 1, __ATOMIC_RELAXED);
      host.inColInd[position] = u;
      for (int k = 0; k < K; ++k) {
        host.inWeights[static_cast<size_t>(k) * m + position] =
            host.outWeights[static_cast<size_t>(k) * m + e];
      }
    }
  }
}

} // namespace

double MospTimings::compute() const {
  double total = combined;
  for (double t : objectives) {
    total += t;
  }
  return total;
}

void canonicalizeTree(const CsrGraph &graph, int objective, int source,
                      const vector<long long> &distances, int *parent) {
  for (int u = 0; u < graph.numberOfNodes; ++u) {
    const long long du = distances[u];
    if (du >= DISTANCE_INF / 2) {
      continue;
    }
    for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
      const int v = graph.colInd[e];
      if (v != source && u < parent[v] &&
          du + graph.weight(e, objective) == distances[v]) {
        parent[v] = u;
      }
    }
  }
}

bool mospUpdate(const CsrGraph &original, ChangeBatch &batch,
                const vector<long long> &initialDistances,
                const vector<int> &initialParents, const MospOptions &options,
                CsrGraph &updated, MospResult &result) {
  const int n = original.numberOfNodes;
  const int K = options.numberOfObjectives > 0
                    ? min(options.numberOfObjectives,
                          original.numberOfObjectives)
                    : original.numberOfObjectives;
  const size_t treeSize = static_cast<size_t>(K) * n;
  result = MospResult();
  result.numberOfObjectives = K;
  MospTimings &timings = result.timings;
  if (n <= 0 || K <= 0 || options.source < 0 || options.source >= n ||
      initialDistances.size() < treeSize || initialParents.size() < treeSize) {
    cerr << "Error: invalid MOSP update input.\n";
    return false;
  }

  // --- Apply the batch once ---------------------------------------------------
  auto start = chrono::steady_clock::now();
  {
    ScopedStage stage("apply_batch");
    if (!applyChangeBatch(original, batch, updated)) {
      return false;
    }
  }
  timings.applyBatch = msSince(start);

  // Per objective: largest weight before and after the batch and the
  // average weight (default near-far bucket width).
  const int KG = original.numberOfObjectives;
  const long long m = original.numberOfEdges();
  vector<long long> maxWeight(K, 1), weightSum(K, 0);
  for (int k = 0; k < K; ++k) {
    long long largest = 1, sum = 0;
#pragma omp parallel for schedule(static) reduction(max : largest) \
    reduction(+ : sum)
    for (long long e = 0; e < m; ++e) {
      const int w = original.weights[static_cast<size_t>(e) * KG + k];
      largest = max(largest, static_cast<long long>(w));
      sum += w;
    }
    for (int i = 0; i < batch.numberOfInserts(); ++i) {
      largest = max(largest, static_cast<long long>(
                                 batch.insertWeights[static_cast<size_t>(i) * KG + k]));
    }
    maxWeight[k] = largest;
    weightSum[k] = sum;
  }

  // Changes that may invalidate a subtree, per objective: every deletion
  // and every insertion that raised the objective's weight of an edge.
  vector<vector<int>> changedFrom(K), changedTo(K);
  for (int k = 0; k < K; ++k) {
    changedFrom[k] = batch.deleteFrom;
    changedTo[k] = batch.deleteTo;
    for (int i = 0; i < batch.numberOfInserts(); ++i) {
      if (batch.weightIncreaseMask[i] & (1u << k)) {
        changedFrom[k].push_back(batch.insertFrom[i]);
        changedTo[k].push_back(batch.insertTo[i]);
      }
    }
  }

  // --- Shared topology, trees and buffers (once) -------------------------------
  start = chrono::steady_clock::now();
  HostGraph graph;
  SospWorkspace sospWorkspace;
  CombineWorkspace combineWorkspace;
  {
    ScopedStage stage("prepare");
    buildHostGraph(updated, K, graph);
    sospWorkspace.reserve(n);
    result.distances.assign(initialDistances.begin(),
                            initialDistances.begin() + treeSize);
    result.parents.assign(initialParents.begin(),
                          initialParents.begin() + treeSize);
    result.combinedDistances.resize(n);
    result.combinedParent.resize(n);
  }
  timings.prepare = msSince(start);

  // --- Step 1 of MOSP: K SOSP updates -------------------------------------------
  result.objectiveStats.resize(K);
  for (int k = 0; k < K; ++k) {
    HostChanges changes;
    changes.changedFrom = changedFrom[k].data();
    changes.changedTo = changedTo[k].data();
    changes.numberOfChanged = static_cast<int>(changedFrom[k].size());
    changes.insertHeads = batch.insertTo.data();
    changes.numberOfInsertHeads = batch.numberOfInserts();
    const long long delta =
        options.delta > 0 ? options.delta
                          : defaultDelta(max(m, 1LL), n, weightSum[k]);
    start = chrono::steady_clock::now();
    ScopedStage stage("obj" + to_string(k) + "/sosp_update_compute");
    if (!sospUpdateCpu(graph.out(updated, k), graph.in(k), changes,
                       options.source, delta, maxWeight[k], sospWorkspace,
                       result.distances.data() + static_cast<size_t>(k) * n,
                       result.parents.data() + static_cast<size_t>(k) * n,
                       &result.objectiveStats[k])) {
      cerr << "Error: SOSP update of objective " << k << " failed.\n";
      return false;
    }
    stage.stop();
    timings.objectives.push_back(msSince(start));
  }

  // --- Steps 2-3 of MOSP: combined graph and its SOSP tree ----------------------
  start = chrono::steady_clock::now();
  {
    ScopedStage stage("combined_graph_compute");
    if (!combinedGraphSospCpu(result.parents.data(), n, K,
                              options.preferences, options.source, 0,
                              combineWorkspace, sospWorkspace,
                              result.combinedDistances.data(),
                              result.combinedParent.data(),
                              &result.combineStats)) {
      cerr << "Error: combined graph step failed.\n";
      return false;
    }
  }
  timings.combined = msSince(start);
  return true;
}
