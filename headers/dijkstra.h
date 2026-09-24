#ifndef DIJKSTRA_H
#define DIJKSTRA_H

#include <string>
#include <vector>

struct CsrGraph;

/**
 * @brief Run single-objective Dijkstra on an MTX file.
 *
 * @param inputFile Path to input Matrix Market file.
 * @param objectiveNumber Objective index (0-based).
 * @param source Source vertex (0-based).
 * @param distanceOutputPath Output file for distances.
 * @param treeOutputPath Output file for parent tree.
 * @return True on success; false otherwise.
 */
bool runDijkstra(
    const std::string &inputFile,
    int objectiveNumber,
    int source,
    const std::string &distanceOutputPath = "output/distances.txt",
    const std::string &treeOutputPath = "output/SSSPTree.txt"
);

/**
 * @brief Run single-objective Dijkstra on CSR files.
 *
 * @param inputPrefix CSR input prefix.
 * @param objectiveNumber Objective index (0-based).
 * @param source Source vertex (0-based).
 * @param distanceOutputPath Output file for distances.
 * @param treeOutputPath Output file for parent tree.
 * @return True on success; false otherwise.
 */
bool runDijkstraCSR(
    const std::string &inputPrefix,
    int objectiveNumber,
    int source,
    const std::string &distanceOutputPath = "output/distancesCsr.txt",
    const std::string &treeOutputPath = "output/SSSPTreeCsr.txt"
);

/**
 * @brief Single-objective Dijkstra on an in-memory CSR graph.
 *
 * @param graph       Out-edge CSR graph.
 * @param objective   Objective index (0-based) used as the edge weight.
 * @param source      Source vertex.
 * @param distances   Output distances (DISTANCE_INF when unreachable).
 * @param parent      Output parent array (-1 for the source/unreachable).
 */
void dijkstraCsrGraph(const CsrGraph &graph, int objective, int source,
                      std::vector<long long> &distances,
                      std::vector<int> &parent);

#endif
