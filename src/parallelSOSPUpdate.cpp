/**
 * @file parallelSOSPUpdate.cpp
 * @brief Parallel (OpenMP) Single-Objective Shortest Path (SOSP) Update.
 *
 * This is the OpenMP-parallel version of sequentialSOSPUpdate.cpp.
 * It produces identical final results (distances + parent arrays) to the
 * sequential version — and therefore matches Dijkstra recalculation on the
 * updated graph.
 *
 * ============================================================================
 * ALGORITHM AND PARALLELIZATION
 * ============================================================================
 *
 * Phase 0 (reading the inputs, applying the batch to the adjacency lists)
 * is sequential. Steps 1 and 2 run in sospUpdateCpu() (OpenMP, see
 * sospUpdateCpu.cpp):
 *   - Step 1, straight from the change list: the head v of every deleted
 *     or weight-increased edge (u,v) with parent[v] == u is a root; the
 *     SOSP subtrees of the roots are invalidated; the invalidated vertices
 *     and the heads of inserted edges pull the best (distance, id) pair
 *     over their in-neighbours.
 *   - Step 2: a push-based near-far worklist with an atomic minimum on the
 *     packed word (distance << b | parent). Distances only decrease, so
 *     there is no iteration cap and no reachability post-pass, and only
 *     improved vertices are expanded.
 *
 * ============================================================================
 */

#include "parallelSOSPUpdate.h"

#include "read.h"
#include "sospUpdateCpu.h"
#include "stageTimer.h"

#include <omp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

namespace {

/// A lightweight edge structure for the internal adjacency lists.
struct WeightedNeighbor {
  int vertex;
  long long weight;
};

/**
 * @brief Parse a line of space-separated integers from a string.
 */
vector<int> parseIntTokens(const string &line) {
  vector<int> tokens;
  istringstream stream(line);
  int value;
  while (stream >> value) {
    tokens.push_back(value);
  }
  return tokens;
}

/**
 * @brief Build forward and reverse adjacency lists from a Graph.
 */
void buildAdjacencyLists(const Graph &graph, int objectiveIndex,
                         vector<vector<WeightedNeighbor>> &outAdjacency,
                         vector<vector<WeightedNeighbor>> &inAdjacency) {
  int numberOfNodes = static_cast<int>(graph.size());
  outAdjacency.assign(numberOfNodes, {});
  inAdjacency.assign(numberOfNodes, {});

  for (int u = 0; u < numberOfNodes; ++u) {
    for (const auto &edge : graph[u]) {
      int v = edge.to;
      long long w = edge.weights[objectiveIndex];
      outAdjacency[u].push_back({v, w});
      inAdjacency[v].push_back({u, w});
    }
  }
}

/**
 * @brief Remove a specific directed edge from an adjacency list entry.
 */
void removeEdgeFromList(vector<WeightedNeighbor> &neighbors, int targetVertex) {
  for (auto it = neighbors.begin(); it != neighbors.end(); ++it) {
    if (it->vertex == targetVertex) {
      neighbors.erase(it);
      return;
    }
  }
}

/**
 * @brief Read distances from a Dijkstra output file.
 */
bool readDistancesFromFile(const string &path, vector<long long> &distances,
                           int numberOfNodes, long long INF_VALUE) {
  ifstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open distances file: " << path << "\n";
    return false;
  }

  distances.assign(numberOfNodes, INF_VALUE);

  string line;
  while (getline(file, line)) {
    if (line.empty())
      continue;
    istringstream stream(line);
    int vertexId;
    string distanceStr;
    stream >> vertexId >> distanceStr;

    if (vertexId < 0 || vertexId >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in distances file.\n";
      return false;
    }

    if (distanceStr == "INF") {
      distances[vertexId] = INF_VALUE;
    } else {
      distances[vertexId] = stoll(distanceStr);
    }
  }

  return true;
}

/**
 * @brief Read SSSP tree (parent array) from a Dijkstra output file.
 */
bool readParentFromFile(const string &path, vector<int> &parent,
                        int numberOfNodes) {
  ifstream file(path);
  if (!file.is_open()) {
    cout << "Error: Could not open SSSP tree file: " << path << "\n";
    return false;
  }

  parent.assign(numberOfNodes, -1);

  string line;
  while (getline(file, line)) {
    if (line.empty())
      continue;
    istringstream stream(line);
    int vertexId, parentId;
    stream >> vertexId >> parentId;

    if (vertexId < 0 || vertexId >= numberOfNodes) {
      cout << "Error: Vertex ID out of range in SSSP tree file.\n";
      return false;
    }

    parent[vertexId] = parentId;
  }

  return true;
}

/**
 * @brief Flatten adjacency lists to CSR (int weights) for the engine.
 */
void flattenToCSR(const vector<vector<WeightedNeighbor>> &adjacency,
                  vector<int> &rowPtr, vector<int> &colInd,
                  vector<int> &weights) {
  const int n = static_cast<int>(adjacency.size());
  rowPtr.assign(n + 1, 0);
  for (int i = 0; i < n; ++i) {
    rowPtr[i + 1] = rowPtr[i] + static_cast<int>(adjacency[i].size());
  }
  colInd.resize(rowPtr[n]);
  weights.resize(rowPtr[n]);
  for (int i = 0; i < n; ++i) {
    int offset = rowPtr[i];
    for (const auto &neighbor : adjacency[i]) {
      colInd[offset] = neighbor.vertex;
      weights[offset] = static_cast<int>(neighbor.weight);
      ++offset;
    }
  }
}

} // namespace

/**
 * @brief Run the parallel (OpenMP) SOSP Update algorithm.
 *
 * @see parallelSOSPUpdate.h for full parameter documentation.
 */
bool parallelSOSPUpdate(const string &originalCsrPrefix,
                        const string &distancesInputPath,
                        const string &treeInputPath, const string &insertPath,
                        const string &deletePath, int objectiveIndex,
                        int source, const string &distancesOutputPath,
                        const string &treeOutputPath) {
  const long long INF_VALUE = numeric_limits<long long>::max() / 4;

  // ========================================================================
  // PHASE 0: PREPARATION (Sequential — I/O dominated)
  // ========================================================================

  // --- 0a. Read original graph from CSR and determine dimensions ---
  ScopedStage readStage("sosp/0a_read_csr_text");
  Graph originalGraph;
  int numberOfObjectives = 0;
  if (!readCSR(originalCsrPrefix, originalGraph, numberOfObjectives)) {
    cout << "Error: Could not read original CSR graph.\n";
    return false;
  }

  int numberOfNodes = static_cast<int>(originalGraph.size());
  if (numberOfNodes == 0) {
    cout << "Error: Graph has no vertices.\n";
    return false;
  }

  if (objectiveIndex < 0 || objectiveIndex >= numberOfObjectives) {
    cout << "Error: objectiveIndex out of range.\n";
    return false;
  }

  if (source < 0 || source >= numberOfNodes) {
    cout << "Error: source vertex out of range.\n";
    return false;
  }

  readStage.stop();

  // --- 0b. Build forward and reverse adjacency lists ---
  ScopedStage adjacencyStage("sosp/0b_build_adjacency");
  vector<vector<WeightedNeighbor>> outAdjacency;
  vector<vector<WeightedNeighbor>> inAdjacency;
  buildAdjacencyLists(originalGraph, objectiveIndex, outAdjacency, inAdjacency);

  originalGraph.clear();
  long long maxWeight = 1;
  for (const auto &row : outAdjacency) {
    for (const auto &neighbor : row) {
      maxWeight = max(maxWeight, neighbor.weight);
    }
  }
  adjacencyStage.stop();

  // --- 0c. Read original distances and parent arrays ---
  ScopedStage treeStage("sosp/0c_read_tree_text");
  vector<long long> distances;
  if (!readDistancesFromFile(distancesInputPath, distances, numberOfNodes,
                             INF_VALUE)) {
    return false;
  }

  vector<int> parent;
  if (!readParentFromFile(treeInputPath, parent, numberOfNodes)) {
    return false;
  }

  treeStage.stop();

  // --- 0d. Read inserted and deleted edges ---
  ScopedStage changesStage("sosp/0d_read_changes_text");
  struct InsertedEdge {
    int from;
    int to;
    long long weight;
  };

  struct DeletedEdge {
    int from;
    int to;
  };

  vector<InsertedEdge> insertedEdges;
  {
    ifstream insertFile(insertPath);
    if (!insertFile.is_open()) {
      cout << "Error: Could not open insert file: " << insertPath << "\n";
      return false;
    }
    string line;
    while (getline(insertFile, line)) {
      if (line.empty())
        continue;
      vector<int> tokens = parseIntTokens(line);
      if (static_cast<int>(tokens.size()) < 2 + numberOfObjectives) {
        cout << "Error: Invalid insert line.\n";
        return false;
      }
      int u = tokens[0];
      int v = tokens[1];
      long long w = tokens[2 + objectiveIndex];
      insertedEdges.push_back({u, v, w});
    }
  }

  vector<DeletedEdge> deletedEdges;
  {
    ifstream deleteFile(deletePath);
    if (!deleteFile.is_open()) {
      cout << "Error: Could not open delete file: " << deletePath << "\n";
      return false;
    }
    string line;
    while (getline(deleteFile, line)) {
      if (line.empty())
        continue;
      vector<int> tokens = parseIntTokens(line);
      if (tokens.size() < 2)
        continue;
      int u = tokens[0];
      int v = tokens[1];
      deletedEdges.push_back({u, v});
    }
  }

  changesStage.stop();

  // --- 0e. Apply topological changes to adjacency lists (Sequential) ---
  ScopedStage applyStage("sosp/0e_apply_changes");
  struct WeightIncrease {
    int from;
    int to;
  };
  vector<WeightIncrease> weightIncreases;

  // Deletions first
  for (const auto &edge : deletedEdges) {
    removeEdgeFromList(outAdjacency[edge.from], edge.to);
    removeEdgeFromList(inAdjacency[edge.to], edge.from);
  }

  // Then insertions: REPLACE if edge already exists, otherwise add.
  for (const auto &edge : insertedEdges) {
    bool replacedOut = false;
    for (auto &neighbor : outAdjacency[edge.from]) {
      if (neighbor.vertex == edge.to) {
        if (edge.weight > neighbor.weight) {
          weightIncreases.push_back({edge.from, edge.to});
        }
        neighbor.weight = edge.weight;
        replacedOut = true;
        break;
      }
    }
    if (!replacedOut) {
      outAdjacency[edge.from].push_back({edge.to, edge.weight});
    }

    bool replacedIn = false;
    for (auto &neighbor : inAdjacency[edge.to]) {
      if (neighbor.vertex == edge.from) {
        neighbor.weight = edge.weight;
        replacedIn = true;
        break;
      }
    }
    if (!replacedIn) {
      inAdjacency[edge.to].push_back({edge.from, edge.weight});
    }
  }

  applyStage.stop();

  // ========================================================================
  // STEPS 1 AND 2 (OpenMP, see sospUpdateCpu.cpp)
  // ========================================================================
  // Heads of inserted edges, and edges that may invalidate a subtree:
  // deletions and weight increases (as (from, to) pairs).
  vector<int> insertHeads, changedFrom, changedTo;
  for (const auto &edge : insertedEdges) {
    insertHeads.push_back(edge.to);
  }
  for (const auto &edge : deletedEdges) {
    changedFrom.push_back(edge.from);
    changedTo.push_back(edge.to);
  }
  for (const auto &wi : weightIncreases) {
    changedFrom.push_back(wi.from);
    changedTo.push_back(wi.to);
  }

  ScopedStage flattenStage("sosp/2a_flatten_csr");
  vector<int> outRowPtr, outColInd, outWeights;
  vector<int> inRowPtr, inColInd, inWeights;
  flattenToCSR(outAdjacency, outRowPtr, outColInd, outWeights);
  flattenToCSR(inAdjacency, inRowPtr, inColInd, inWeights);
  outAdjacency.clear();
  inAdjacency.clear();
  long long weightSum = 0;
  for (int w : outWeights) {
    weightSum += w;
    maxWeight = max(maxWeight, static_cast<long long>(w));
  }
  const long long delta = defaultDelta(
      static_cast<long long>(outColInd.size()), numberOfNodes, weightSum);
  HostCsr outCsr;
  outCsr.numberOfNodes = numberOfNodes;
  outCsr.numberOfEdges = static_cast<int>(outColInd.size());
  outCsr.rowPtr = outRowPtr.data();
  outCsr.colInd = outColInd.data();
  outCsr.weights = outWeights.data();
  HostCsr inCsr = outCsr;
  inCsr.rowPtr = inRowPtr.data();
  inCsr.colInd = inColInd.data();
  inCsr.weights = inWeights.data();
  HostChanges changes;
  changes.changedFrom = changedFrom.data();
  changes.changedTo = changedTo.data();
  changes.numberOfChanged = static_cast<int>(changedFrom.size());
  changes.insertHeads = insertHeads.data();
  changes.numberOfInsertHeads = static_cast<int>(insertHeads.size());
  SospWorkspace workspace;
  workspace.reserve(numberOfNodes);
  flattenStage.stop();

  SospStats stats;
  {
    ScopedStage updateStage("sosp/update_compute");
    if (!sospUpdateCpu(outCsr, inCsr, changes, source, delta, maxWeight,
                       workspace, distances.data(), parent.data(), &stats)) {
      cout << "Error: SOSP update failed.\n";
      return false;
    }
  }
  recordCounter("sosp/invalidated", stats.invalidated);
  recordCounter("sosp/iterations", stats.iterations);
  recordCounter("sosp/epochs", stats.epochs);
  recordCounter("sosp/pushes", stats.pushes);

  // ========================================================================
  // WRITE OUTPUT (Sequential — I/O)
  // ========================================================================
  ScopedStage writeStage("sosp/5_write_text");

  filesystem::path distOutPath(distancesOutputPath);
  if (!distOutPath.parent_path().empty()) {
    filesystem::create_directories(distOutPath.parent_path());
  }

  filesystem::path treeOutPath(treeOutputPath);
  if (!treeOutPath.parent_path().empty()) {
    filesystem::create_directories(treeOutPath.parent_path());
  }

  ofstream distancesOut(distancesOutputPath);
  if (!distancesOut.is_open()) {
    cout << "Error: Could not write updated distances file.\n";
    return false;
  }

  for (int i = 0; i < numberOfNodes; ++i) {
    distancesOut << i << " ";
    if (distances[i] >= INF_VALUE / 2) {
      distancesOut << "INF";
    } else {
      distancesOut << distances[i];
    }
    distancesOut << "\n";
  }

  ofstream treeOut(treeOutputPath);
  if (!treeOut.is_open()) {
    cout << "Error: Could not write updated SSSP tree file.\n";
    return false;
  }

  for (int i = 0; i < numberOfNodes; ++i) {
    treeOut << i << " " << parent[i] << "\n";
  }

  return true;
}
