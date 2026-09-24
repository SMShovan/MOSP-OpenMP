#ifndef PARALLEL_COMBINED_GRAPH_H
#define PARALLEL_COMBINED_GRAPH_H

#include <string>
#include <vector>

/**
 * @brief Build a combined graph from K SSSP trees and find its SSSP.
 *
 * @details
 * Given K pre-computed SSSP tree parent arrays (one per objective),
 * this function:
 *
 * 1. Extracts all directed tree edges from each parent array.
 * 2. Counts how many trees each edge belongs to (membership count m ∈ {1..K}).
 * 3. Assigns edge weights by preference (thesis Ch. 4, Algorithm
 *    MOSP_Update, Step 2):
 *       W(e) = K + 1 - sum_{i : e in T_i} 1 / Pref_i
 *    Pref_i >= 1 (a lower value means a higher priority). The weights are
 *    kept integral by scaling with L = lcm(Pref_1..Pref_K):
 *       W'(e) = L * (K + 1) - sum_{i : e in T_i} L / Pref_i
 *    so the combined-graph distances are in units of 1/L. With the default
 *    Pref = (1,...,1), L = 1 and W(e) = K + 1 - m, where m is the number of
 *    trees containing e (m == K -> 1, ..., m == 1 -> K).
 * 4. Writes the combined-graph edges as a fresh insertion batch.
 * 5. Calls parallelSOSPUpdate with a blank base graph (all distances = INF,
 *    source distance = 0, empty SSSP tree) so the algorithm runs like
 *    Bellman-Ford seeded from the source on the combined graph.
 *
 * All intermediate files are written under @p workDir.
 *
 * @param originalCsrPrefix   Prefix of the original CSR files (used only to
 *                             determine the number of vertices; no edge data
 *                             from this graph is used in the combined graph).
 * @param treeInputPaths       Paths to the K SSSP tree parent files produced
 *                             by parallelSOSPUpdate (one per objective).
 * @param K                    Number of objectives / trees (must equal
 *                             treeInputPaths.size()).
 * @param source               Source vertex (0-indexed, default 0).
 * @param workDir              Directory for intermediate temp files
 *                             (default "output/combinedGraph").
 * @param distancesOutputPath  Output path for SSSP distances on combined graph.
 * @param treeOutputPath       Output path for SSSP parent array on combined
 * graph.
 * @param preferences          Pref vector (K entries >= 1); empty = all 1s.
 * @return True on success; false on any I/O or algorithm error.
 */
bool parallelCombinedGraph(
    const std::string &originalCsrPrefix,
    const std::vector<std::string> &treeInputPaths, int K, int source = 0,
    const std::string &workDir = "output/combinedGraph",
    const std::string &distancesOutputPath =
        "output/combinedGraph/distancesCsr.txt",
    const std::string &treeOutputPath = "output/combinedGraph/SSSPTreeCsr.txt",
    const std::vector<int> &preferences = std::vector<int>());

/**
 * @brief Weight scale L = lcm(Pref_1..Pref_K) of the combined graph.
 *
 * @param preferences K preference values (empty = all 1s).
 * @return L, or 0 if a preference is < 1, the size is not K, or L would
 *         exceed 2^20.
 */
long long preferenceScale(const std::vector<int> &preferences, int K);

/**
 * @brief Integer (scaled) weight of a combined-graph edge.
 *
 * @param treeMask Bit i is set when the edge belongs to tree T_i.
 * @param scale    L from preferenceScale().
 * @return L * (K + 1) - sum_{i in treeMask} L / Pref_i  (always >= L).
 */
long long combinedEdgeWeight(unsigned int treeMask,
                             const std::vector<int> &preferences, int K,
                             long long scale);

struct CsrGraph;

/**
 * @brief Thesis Step 3, last line: the MOSP cost vectors.
 *
 * @details
 * Re-assigns the original weights of @p graph (the updated graph) to the
 * edges of the SOSP tree found on the combined graph and sums them along
 * the tree: costs[v * K + k] is the objective-k length of the MOSP path
 * from @p source to v (DISTANCE_INF if v is unreachable).
 *
 * @return false if a tree edge (parent[v], v) does not exist in @p graph.
 */
bool mospPathCosts(const CsrGraph &graph, const std::vector<int> &parent,
                   int source, std::vector<long long> &costs);

#endif // PARALLEL_COMBINED_GRAPH_H
