#ifndef MOSP_UPDATE_H
#define MOSP_UPDATE_H

/**
 * @file mospUpdate.h
 * @brief In-memory MOSP update (thesis Ch. 4, Algorithm MOSP_Update): K SOSP
 *        updates on a graph shared by all objectives, the combined graph
 *        and its SOSP tree, with OpenMP.
 */

#include "combinedGraphCpu.h"

#include <vector>

struct CsrGraph;
struct ChangeBatch;

/// Options of mospUpdate().
struct MospOptions {
  int source = 0;
  int numberOfObjectives = 0;   ///< 0: every objective of the graph
  std::vector<int> preferences; ///< Pref vector; empty = all 1s
  long long delta = 0;          ///< near-far bucket width; <= 0: default
};

/// Wall-clock times (milliseconds) of the stages of mospUpdate().
struct MospTimings {
  double applyBatch = 0;          ///< apply the batch to the CSR
  double prepare = 0;             ///< reverse graph and weight columns
  std::vector<double> objectives; ///< SOSP update of each objective
  double combined = 0;            ///< combined graph + its SOSP tree

  /// The region the papers time: the K SOSP updates and Steps 2-3.
  double compute() const;
};

/// Output of mospUpdate().
struct MospResult {
  int numberOfObjectives = 0;
  std::vector<long long> distances;  ///< K * n, objective-major
  std::vector<int> parents;          ///< K * n, objective-major
  std::vector<long long> combinedDistances; ///< units of 1/L
  std::vector<int> combinedParent;   ///< the MOSP tree
  std::vector<SospStats> objectiveStats;
  CombineStats combineStats;
  MospTimings timings;
};

/**
 * @brief Run the MOSP update on an in-memory graph.
 *
 * @details
 * The batch is applied once (applyChangeBatch); the reverse graph and the
 * K weight columns are built once, in parallel, and shared by all
 * objectives. The K SOSP updates (sospUpdateCpu) and Steps 2-3
 * (combinedGraphSospCpu) reuse buffers allocated once for the whole run.
 *
 * @param original           Graph before the batch.
 * @param batch              The batch (weightIncreaseMask is filled).
 * @param initialDistances   K * n initial distances (objective-major).
 * @param initialParents     K * n initial parents (objective-major); ties
 *                           broken by the lowest parent id.
 * @param updated            Output: the updated graph (host copy).
 */
bool mospUpdate(const CsrGraph &original, ChangeBatch &batch,
                const std::vector<long long> &initialDistances,
                const std::vector<int> &initialParents,
                const MospOptions &options, CsrGraph &updated,
                MospResult &result);

/**
 * @brief Force the lowest-id tie rule on a tree of @p graph.
 *
 * @details
 * For every edge (u,v) with dist[u] + w(u,v) == dist[v] and u < parent[v],
 * parent[v] becomes u. Used to normalize initial trees produced by other
 * tools (O(m), one pass over the out-edges).
 */
void canonicalizeTree(const CsrGraph &graph, int objective, int source,
                      const std::vector<long long> &distances,
                      int *parent);

#endif // MOSP_UPDATE_H
