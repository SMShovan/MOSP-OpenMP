#ifndef COMBINED_GRAPH_CPU_H
#define COMBINED_GRAPH_CPU_H

/**
 * @file combinedGraphCpu.h
 * @brief Steps 2 and 3 of the MOSP update with OpenMP: build the combined
 *        graph from the K SOSP trees and find its SOSP tree.
 */

#include "sospUpdateCpu.h"

#include <vector>

/// Counters reported by combinedGraphSospCpu().
struct CombineStats {
  int numberOfEdges = 0; ///< edges of the combined graph
  long long scale = 1;   ///< L: combined distances are in units of 1/L
  long long delta = 0;   ///< near-far bucket width used for Step 3
  SospStats search;      ///< Step 3 statistics
};

/// Buffers of the combined graph (children CSR); reuse across runs.
struct CombineWorkspace {
  std::vector<int> rowPtr, cursor, colInd, weights;
};

/**
 * @brief Build the combined graph of the K trees and run SSSP on it.
 *
 * @details
 * Edge (p, v) belongs to the combined graph iff p is the parent of v in
 * some tree T_i; its weight is L * (K + 1) - sum_{i : Parent_i[v] == p}
 * L / Pref_i (L = lcm(Pref); Pref = all 1s gives K + 1 - m). The edges into
 * v come from comparing the K parents of v (one iteration per vertex, no
 * map or edge-list sort); a count / prefix-sum / fill pass builds the
 * out-edge CSR and Step 3 is a near-far SSSP from @p source.
 *
 * @param parents     K parent arrays, objective-major: parents[k * n + v].
 * @param preferences Pref vector (K values >= 1); empty = all 1s.
 * @param delta       Near-far bucket width; <= 0 selects the default.
 */
bool combinedGraphSospCpu(const int *parents, int numberOfNodes, int K,
                          const std::vector<int> &preferences, int source,
                          long long delta, CombineWorkspace &workspace,
                          SospWorkspace &sospWorkspace, long long *distances,
                          int *parent, CombineStats *stats = nullptr);

#endif // COMBINED_GRAPH_CPU_H
