/**
 * @file combinedGraphCpu.cpp
 * @brief Steps 2 and 3 of the MOSP update with OpenMP.
 *
 * Step 2 (combined graph): the in-edges of v in the combined graph are the
 * distinct values among Parent_1[v] .. Parent_K[v], so each vertex compares
 * its K parents (one loop iteration per vertex, "a single thread per vertex
 * compares its parents"). A count / prefix-sum / fill pass builds a CSR of
 * the out-edges; the order inside a row does not matter because Step 3
 * breaks distance ties by the lowest parent id.
 *
 * Step 3: a near-far SSSP from the source with the same engine as the SOSP
 * update (sospFromScratchCpu).
 */

#include "combinedGraphCpu.h"

#include "csrGraph.h"
#include "parallelCombinedGraph.h"

#include <omp.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

using namespace std;

namespace {

/**
 * If parent k of v is the first occurrence of its value among the K
 * parents, return true and its combined-graph weight
 *   base - sum_{j : Parent_j[v] == p} prefTerm[j].
 */
inline bool combinedEdge(const int *parents, int n, int K, int v, int k,
                         const vector<int> &prefTerms, int base, int &p,
                         int &weight) {
  p = parents[static_cast<size_t>(k) * n + v];
  if (p < 0) {
    return false;
  }
  for (int j = 0; j < k; ++j) {
    if (parents[static_cast<size_t>(j) * n + v] == p) {
      return false; // counted with its first occurrence
    }
  }
  weight = base - prefTerms[k];
  for (int j = k + 1; j < K; ++j) {
    if (parents[static_cast<size_t>(j) * n + v] == p) {
      weight -= prefTerms[j];
    }
  }
  return true;
}

} // namespace

// ============================================================================
// Host helpers (declared in parallelCombinedGraph.h)
// ============================================================================

long long preferenceScale(const vector<int> &preferences, int K) {
  if (preferences.empty()) {
    return 1;
  }
  if (static_cast<int>(preferences.size()) != K) {
    return 0;
  }
  long long scale = 1;
  for (int pref : preferences) {
    if (pref < 1) {
      return 0;
    }
    scale = scale / gcd(scale, static_cast<long long>(pref)) * pref;
    if (scale > (1LL << 20)) {
      return 0;
    }
  }
  return scale;
}

long long combinedEdgeWeight(unsigned int treeMask,
                             const vector<int> &preferences, int K,
                             long long scale) {
  long long weight = scale * (K + 1);
  for (int i = 0; i < K; ++i) {
    if (treeMask & (1u << i)) {
      weight -= preferences.empty() ? scale : scale / preferences[i];
    }
  }
  return weight;
}

bool mospPathCosts(const CsrGraph &graph, const vector<int> &parent,
                   int source, vector<long long> &costs) {
  const int n = graph.numberOfNodes;
  const int K = graph.numberOfObjectives;
  costs.assign(static_cast<size_t>(n) * K, DISTANCE_INF);
  // Children lists of the tree, then a traversal from the source.
  vector<int> childStart(n + 1, 0), children(n);
  for (int v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      ++childStart[parent[v] + 1];
    }
  }
  for (int v = 0; v < n; ++v) {
    childStart[v + 1] += childStart[v];
  }
  vector<int> cursor(childStart.begin(), childStart.end() - 1);
  for (int v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      children[cursor[parent[v]]++] = v;
    }
  }
  for (int k = 0; k < K; ++k) {
    costs[static_cast<size_t>(source) * K + k] = 0;
  }
  vector<int> queue{source};
  for (size_t i = 0; i < queue.size(); ++i) {
    int p = queue[i];
    for (int c = childStart[p]; c < childStart[p + 1]; ++c) {
      int v = children[c];
      int edge = -1;
      for (int e = graph.rowPtr[p]; e < graph.rowPtr[p + 1]; ++e) {
        if (graph.colInd[e] == v) {
          edge = e;
          break;
        }
      }
      if (edge < 0) {
        return false;
      }
      for (int k = 0; k < K; ++k) {
        costs[static_cast<size_t>(v) * K + k] =
            costs[static_cast<size_t>(p) * K + k] + graph.weight(edge, k);
      }
      queue.push_back(v);
    }
  }
  return true;
}

bool combinedGraphSospCpu(const int *parents, int n, int K,
                          const vector<int> &preferences, int source,
                          long long delta, CombineWorkspace &ws,
                          SospWorkspace &sospWorkspace, long long *distances,
                          int *parent, CombineStats *stats) {
  CombineStats local;
  CombineStats &s = stats != nullptr ? *stats : local;
  s = CombineStats();
  if (K <= 0 || K > 32 || n <= 0 || source < 0 || source >= n) {
    cerr << "Error: invalid combined-graph parameters.\n";
    return false;
  }
  const long long scale = preferenceScale(preferences, K);
  if (scale == 0) {
    cerr << "Error: invalid preference vector (need K values >= 1).\n";
    return false;
  }
  vector<int> terms(K);
  for (int k = 0; k < K; ++k) {
    terms[k] = static_cast<int>(preferences.empty() ? scale
                                                    : scale / preferences[k]);
  }
  const int base = static_cast<int>(scale * (K + 1));

  // Step 2: count out-degrees, prefix sum, fill.
  ws.cursor.assign(static_cast<size_t>(n) + 1, 0);
  long long edges = 0, weightSum = 0;
  int *degree = ws.cursor.data();
#pragma omp parallel for schedule(static) reduction(+ : edges, weightSum)
  for (int v = 0; v < n; ++v) {
    if (v == source) {
      continue;
    }
    for (int k = 0; k < K; ++k) {
      int p, weight;
      if (combinedEdge(parents, n, K, v, k, terms, base, p, weight)) {
        __atomic_fetch_add(&degree[p], 1, __ATOMIC_RELAXED);
        ++edges;
        weightSum += weight;
      }
    }
  }
  ws.rowPtr.resize(static_cast<size_t>(n) + 1);
  ws.rowPtr[0] = 0;
  partial_sum(ws.cursor.begin(), ws.cursor.end() - 1, ws.rowPtr.begin() + 1);
  copy(ws.rowPtr.begin(), ws.rowPtr.end() - 1, ws.cursor.begin());
  ws.colInd.resize(static_cast<size_t>(edges));
  ws.weights.resize(static_cast<size_t>(edges));
  int *cursor = ws.cursor.data();
#pragma omp parallel for schedule(static)
  for (int v = 0; v < n; ++v) {
    if (v == source) {
      continue;
    }
    for (int k = 0; k < K; ++k) {
      int p, weight;
      if (combinedEdge(parents, n, K, v, k, terms, base, p, weight)) {
        int position = __atomic_fetch_add(&cursor[p], 1, __ATOMIC_RELAXED);
        ws.colInd[position] = v;
        ws.weights[position] = weight;
      }
    }
  }

  HostCsr combined;
  combined.numberOfNodes = n;
  combined.numberOfEdges = static_cast<int>(edges);
  combined.rowPtr = ws.rowPtr.data();
  combined.colInd = ws.colInd.data();
  combined.weights = ws.weights.data();
  s.numberOfEdges = combined.numberOfEdges;
  s.scale = scale;
  s.delta = delta > 0 ? delta : defaultDelta(edges, n, weightSum);

  // Step 3: SSSP on the combined graph.
  return sospFromScratchCpu(combined, source, s.delta, base, sospWorkspace,
                            distances, parent, &s.search);
}
