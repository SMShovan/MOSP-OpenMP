#ifndef CHANGE_GENERATOR_H
#define CHANGE_GENERATOR_H

#include <string>

struct CsrGraph;
struct ChangeBatch;

/**
 * @brief How the changed edges of a batch are drawn.
 *
 * - Uniform:  insertions join two random vertices with random weights in
 *             [weightMin, weightMax]; deletions are sampled (with
 *             replacement) from the existing edges. For the same seed this
 *             reproduces generateChangedEdges(..., directed=true,
 *             exist=true, duplicate=true, selfLoop=false, seed) exactly.
 * - Targeted: the thesis workload (Ch. 4, Sec. "Performance"): insertions
 *             join two random vertices with weights below the graph's
 *             average weight of each objective, and every deletion removes
 *             a distinct edge of an SOSP tree (tree of a random objective).
 * - Reweight: every change overwrites the weights of an existing edge with
 *             new random weights in [weightMin, weightMax] (written as an
 *             insertion of an existing edge; increases and decreases).
 * - Increase: every change raises all weights of a distinct SOSP-tree edge
 *             by a random amount in [1, weightMax] (tree-edge weight
 *             increases, which invalidate subtrees like deletions do).
 */
enum class ChangeMode { Uniform, Targeted, Reweight, Increase };

/** @brief Parse "uniform", "targeted", "reweight" or "increase". */
bool parseChangeMode(const std::string &name, ChangeMode &mode);

/** @brief Options of generateChangeBatch(). */
struct ChangeGeneratorOptions {
  int numberOfChanges = 0;
  double insertionPercentage = 50; ///< Uniform/Targeted: share of insertions
  ChangeMode mode = ChangeMode::Uniform;
  int weightMin = 1;
  int weightMax = 100;
  unsigned int seed = 1;
  /// > 0: every endpoint lies within this many hops (ignoring edge
  /// direction) of a random centre vertex, giving a local batch.
  int localHops = 0;
  /// Drop deletions that would disconnect a vertex that is reachable from
  /// @ref source ("connectivity-safe" batches, the thesis assumption).
  bool safeDeletions = false;
  int source = 0;
};

/**
 * @brief Generate a seeded batch of changes for @p graph.
 *
 * @param report  Optional human-readable summary (counts, centre, ...).
 * @return false if the options are invalid or the graph has too few edges.
 */
bool generateChangeBatch(const CsrGraph &graph,
                         const ChangeGeneratorOptions &options,
                         ChangeBatch &batch, std::string *report = nullptr);

/** @brief Write a batch as insert.txt / delete.txt. */
bool writeChangeBatch(const ChangeBatch &batch, const std::string &insertPath,
                      const std::string &deletePath);

#endif // CHANGE_GENERATOR_H
