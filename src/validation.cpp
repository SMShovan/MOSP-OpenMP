/**
 * @file validation.cpp
 * @brief Oracles for SOSP trees: distances, parent consistency and the
 *        canonical (lowest-id) parent rule.
 */

#include "validation.h"

#include "csrGraph.h"
#include "parallelCombinedGraph.h"

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

string TreeCheck::summary() const {
  ostringstream out;
  out << "distance mismatches=" << distanceMismatches
      << " inconsistent parents=" << inconsistentParents
      << " non-canonical parents=" << nonCanonicalParents
      << " parent mismatches=" << parentMismatches;
  return out.str();
}

TreeCheck checkSospTree(const CsrGraph &reverse, int objective, int source,
                        const vector<long long> &distances,
                        const vector<int> &parents,
                        const vector<long long> &referenceDistances,
                        const vector<int> *referenceParents) {
  TreeCheck check;
  const int n = reverse.numberOfNodes;
  auto finite = [](long long d) { return d < DISTANCE_INF / 2; };

  for (int v = 0; v < n; ++v) {
    const long long dv = distances[v];
    const bool reachable = finite(dv);
    if (reachable != finite(referenceDistances[v]) ||
        (reachable && dv != referenceDistances[v])) {
      ++check.distanceMismatches;
    }
    if (referenceParents != nullptr && parents[v] != (*referenceParents)[v]) {
      ++check.parentMismatches;
    }

    if (v == source) {
      if (parents[v] != -1 || dv != 0) {
        ++check.inconsistentParents;
      }
      continue;
    }
    if (!reachable) {
      if (parents[v] != -1) {
        ++check.inconsistentParents;
      }
      continue;
    }

    const int p = parents[v];
    bool consistent = false;
    int lowest = -1;
    for (int e = reverse.rowPtr[v]; e < reverse.rowPtr[v + 1]; ++e) {
      const int u = reverse.colInd[e];
      if (!finite(distances[u])) {
        continue;
      }
      // A tight edge must have a positive weight: parents then have
      // strictly smaller distances, so consistent parents form no cycle.
      const int w = reverse.weight(e, objective);
      if (w <= 0 || distances[u] + w != dv) {
        continue;
      }
      if (u == p) {
        consistent = true;
      }
      if (lowest < 0 || u < lowest) {
        lowest = u;
      }
    }
    if (!consistent) {
      ++check.inconsistentParents;
    } else if (p != lowest) {
      ++check.nonCanonicalParents;
    }
  }
  return check;
}

CsrGraph combinedGraphReference(const vector<int> &parents, int n, int K,
                                int source, const vector<int> &preferences) {
  const long long scale = preferenceScale(preferences, K);
  CsrGraph combined;
  combined.numberOfNodes = n;
  combined.numberOfObjectives = 1;
  vector<vector<pair<int, int>>> rows(n);
  for (int v = 0; v < n; ++v) {
    if (v == source) {
      continue;
    }
    for (int k = 0; k < K; ++k) {
      const int p = parents[static_cast<size_t>(k) * n + v];
      bool seen = p < 0;
      for (int j = 0; j < k && !seen; ++j) {
        seen = parents[static_cast<size_t>(j) * n + v] == p;
      }
      if (seen) {
        continue;
      }
      unsigned int mask = 0;
      for (int j = 0; j < K; ++j) {
        mask |= parents[static_cast<size_t>(j) * n + v] == p ? 1u << j : 0u;
      }
      rows[p].push_back(
          {v, static_cast<int>(combinedEdgeWeight(mask, preferences, K, scale))});
    }
  }
  combined.rowPtr.assign(n + 1, 0);
  for (int u = 0; u < n; ++u) {
    combined.rowPtr[u + 1] = combined.rowPtr[u] + static_cast<int>(rows[u].size());
    for (const auto &edge : rows[u]) {
      combined.colInd.push_back(edge.first);
      combined.weights.push_back(edge.second);
    }
  }
  return combined;
}
