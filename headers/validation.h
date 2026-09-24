#ifndef VALIDATION_H
#define VALIDATION_H

#include <string>
#include <vector>

struct CsrGraph;

/**
 * @brief Result of checking an SOSP tree against a reference.
 */
struct TreeCheck {
  long long distanceMismatches = 0;  ///< dist[v] != reference[v]
  long long inconsistentParents = 0; ///< no edge (p,v) with d[p]+w == d[v]
  long long nonCanonicalParents = 0; ///< consistent, but not the lowest id
  long long parentMismatches = 0;    ///< parent[v] != referenceParent[v]

  bool ok(bool requireCanonical) const {
    return distanceMismatches == 0 && inconsistentParents == 0 &&
           parentMismatches == 0 && (!requireCanonical || nonCanonicalParents == 0);
  }
  std::string summary() const;
};

/**
 * @brief Validate distances and parents of an SOSP tree.
 *
 * @details
 * - distances must equal @p referenceDistances (typically Dijkstra on the
 *   updated graph); unreachable vertices must be DISTANCE_INF;
 * - parent consistency: the source and unreachable vertices have parent -1;
 *   every other vertex v has a parent p such that the graph contains the
 *   edge (p,v) with w(p,v) > 0 and dist[p] + w(p,v) == dist[v] (so the
 *   parents of consistent vertices cannot form a cycle);
 * - canonical parents: p is the lowest vertex id among all in-neighbours u
 *   with dist[u] + w(u,v) == dist[v] (the tie-break rule of this code);
 * - if @p referenceParents is non-null, parents must match it exactly.
 *
 * @param reverse   Reverse (in-edge) CSR of the graph the tree belongs to.
 * @param objective Objective index used as the weight.
 */
TreeCheck checkSospTree(const CsrGraph &reverse, int objective, int source,
                        const std::vector<long long> &distances,
                        const std::vector<int> &parents,
                        const std::vector<long long> &referenceDistances,
                        const std::vector<int> *referenceParents = nullptr);

/**
 * @brief Host reference of the combined graph (thesis Ch. 4, Step 2).
 *
 * @details
 * Edge (p,v) is in the graph iff p is the parent of v in some tree T_i;
 * its weight is L * (K + 1) - sum_i L / Pref_i over those trees (see
 * combinedEdgeWeight()). One objective per edge.
 *
 * @param parents K * n parent arrays, objective-major.
 */
CsrGraph combinedGraphReference(const std::vector<int> &parents, int n, int K,
                                int source,
                                const std::vector<int> &preferences);

#endif // VALIDATION_H
