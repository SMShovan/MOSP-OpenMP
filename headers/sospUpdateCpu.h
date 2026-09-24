#ifndef SOSP_UPDATE_CPU_H
#define SOSP_UPDATE_CPU_H

/**
 * @file sospUpdateCpu.h
 * @brief Shared-memory (OpenMP) SOSP update engine: Steps 1 and 2 of the
 *        SOSP update, and SSSP from scratch for Step 3 of the MOSP update.
 *
 * The CPU counterpart of MOSP-CUDA's sospUpdateGpu (same algorithm, same
 * results); see sospUpdateCpu.cpp.
 */

#include <vector>

/// A CSR graph with one weight per edge.
struct HostCsr {
  int numberOfNodes = 0;
  int numberOfEdges = 0;
  const int *rowPtr = nullptr;
  const int *colInd = nullptr;
  const int *weights = nullptr;
};

/// The change batch as seen by one objective.
struct HostChanges {
  /// Edges (from[i], to[i]) that were deleted or whose weight increased;
  /// the head of such an edge is a root if the edge is its tree edge.
  const int *changedFrom = nullptr;
  const int *changedTo = nullptr;
  int numberOfChanged = 0;
  /// Heads of inserted edges (their distance may decrease).
  const int *insertHeads = nullptr;
  int numberOfInsertHeads = 0;
};

/// Counters reported by one update.
struct SospStats {
  int invalidated = 0;  ///< vertices in invalidated subtrees
  int iterations = 0;   ///< near-far push iterations
  int epochs = 0;       ///< far-pile threshold increases
  long long pushes = 0; ///< vertex expansions
  /// false: distances alone did not fit next to the parent ids, parents
  /// were recovered after the search
  bool packedParents = true;
};

/**
 * @brief Scratch space for updates on graphs with up to capacity
 *        vertices; reserve once and reuse for every objective.
 */
struct SospWorkspace {
  /// Allocate for @p capacity vertices (no-op if already large enough).
  void reserve(int capacity);
  /// Fresh stamp generation (stamps deduplicate list insertions).
  int nextGeneration();

  int capacity = 0;
  int generation = 0;
  std::vector<unsigned long long> packed; ///< (distance << b | parent)
  std::vector<int> stamp;                 ///< last generation listed
  std::vector<char> inFar;                ///< vertex is in the far pile
  std::vector<char> state;                ///< invalidation walk states
  std::vector<int> nearA, nearB, far, far2, candidates, frontier;
};

/**
 * @brief Default near-far bucket width: 32 * average weight / average
 *        out-degree (at least 1), the same heuristic as the GPU version.
 */
long long defaultDelta(long long numberOfEdges, int numberOfNodes,
                       long long weightSum);

/**
 * @brief Incremental SOSP update.
 *
 * On entry @p distances / @p parent hold the canonical tree of the old
 * graph; on return the canonical tree of the new graph, whose out- and
 * in-edges are @p out and @p in. Unreachable vertices get DISTANCE_INF and
 * parent -1.
 *
 * @param maxWeight Largest edge weight before or after the batch.
 * @param delta     Near-far bucket width (> 0).
 * @return false (with a message) on invalid input, e.g. an input distance
 *         above (n - 1) * maxWeight or a parent cycle in the input tree.
 */
bool sospUpdateCpu(const HostCsr &out, const HostCsr &in,
                   const HostChanges &changes, int source, long long delta,
                   long long maxWeight, SospWorkspace &workspace,
                   long long *distances, int *parent,
                   SospStats *stats = nullptr);

/** @brief Single-source shortest paths from scratch (canonical tree). */
bool sospFromScratchCpu(const HostCsr &out, int source, long long delta,
                        long long maxWeight, SospWorkspace &workspace,
                        long long *distances, int *parent,
                        SospStats *stats = nullptr);

#endif // SOSP_UPDATE_CPU_H
