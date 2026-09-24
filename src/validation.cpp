/**
 * @file validation.cpp
 * @brief Oracles for SOSP trees: distances, parent consistency and the
 *        canonical (lowest-id) parent rule.
 */

#include "validation.h"

#include "csrGraph.h"

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
      if (distances[u] + reverse.weight(e, objective) != dv) {
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
