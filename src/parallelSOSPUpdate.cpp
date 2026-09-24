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
 * PARALLELIZATION STRATEGY
 * ============================================================================
 *
 * Phases 0 and 1 (preparation + initial edge processing) remain sequential
 * because they are I/O-dominated and operate on tiny batches.
 *
 * Phase 2 (iterative propagation) is parallelized:
 *   - Step 2a (collect candidates): parallel for with thread-local vectors
 *     and atomic compare-exchange for deduplication.
 *   - Step 2b (update distances):  parallel for with per-candidate
 *     findBestParent; thread-local affected lists and atomic flags.
 *
 * Post-processing BFS (reachability check) uses level-synchronous parallel
 * BFS with atomic visited flags.
 *
 * Race Condition Analysis:
 *   - isCandidate[]/isAffected[]: atomic compare-exchange on char arrays
 *   - candidateVertices/affectedVertices: thread-local + critical merge
 *   - distances[v]/parent[v] writes in Step 2b: no race because each
 *     candidate v appears exactly once (deduplicated)
 *   - distances[u] reads in findBestParent: benign race under Chaotic
 *     Bellman-Ford semantics — convergence guaranteed, same final result
 *   - reachable[] in BFS: atomic compare-exchange
 *
 * ============================================================================
 */

#include "parallelSOSPUpdate.h"

#include "read.h"
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
 * @brief Find the in-neighbor that gives the minimum distance to a vertex.
 *
 * Thread-safety: reads distances[] which may be concurrently written by
 * other threads (benign race — Chaotic Bellman-Ford convergence guarantees
 * the final result is correct after sufficient iterations).
 */
void findBestParent(int vertex,
                    const vector<vector<WeightedNeighbor>> &inAdjacency,
                    const vector<long long> &distances, long long INF_VALUE,
                    int &bestParent, long long &bestDistance) {
  bestParent = -1;
  bestDistance = INF_VALUE;

  for (const auto &inNeighbor : inAdjacency[vertex]) {
    int candidateParent = inNeighbor.vertex;
    long long candidateWeight = inNeighbor.weight;

    // Skip unreachable in-neighbors to avoid overflow
    if (distances[candidateParent] >= INF_VALUE / 2) {
      continue;
    }

    long long candidateDistance = distances[candidateParent] + candidateWeight;
    // Ties go to the lowest parent id (canonical SOSP tree).
    if (candidateDistance < bestDistance ||
        (candidateDistance == bestDistance && candidateParent < bestParent)) {
      bestDistance = candidateDistance;
      bestParent = candidateParent;
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
  // PHASE 1: PROCESS CHANGED EDGES (Sequential — tiny batch, order-dependent)
  // ========================================================================
  ScopedStage step1Stage("sosp/1_process_changes_compute");

  // Use char arrays instead of bool vectors for atomic compare-exchange
  vector<char> isAffected(numberOfNodes, 0);
  vector<int> affectedVertices;

  // --- 1a. Process insertions ---
  for (const auto &edge : insertedEdges) {
    int u = edge.from;
    int v = edge.to;

    if (distances[u] >= INF_VALUE / 2) {
      continue;
    }

    long long actualWeight = -1;
    for (const auto &neighbor : outAdjacency[u]) {
      if (neighbor.vertex == v) {
        actualWeight = neighbor.weight;
        break;
      }
    }
    if (actualWeight < 0) {
      continue;
    }

    long long newDistance = distances[u] + actualWeight;
    if (newDistance < distances[v]) {
      distances[v] = newDistance;
      parent[v] = u;
      if (!isAffected[v]) {
        isAffected[v] = 1;
        affectedVertices.push_back(v);
      }
    } else if (newDistance == distances[v] && v != source && u < parent[v]) {
      parent[v] = u; // equal distance: the lowest parent id wins
    }
  }

  // --- 1b. Process deletions ---
  for (const auto &edge : deletedEdges) {
    int u = edge.from;
    int v = edge.to;

    if (parent[v] != u) {
      continue;
    }

    int bestAlternativeParent = -1;
    long long bestAlternativeDistance = INF_VALUE;
    findBestParent(v, inAdjacency, distances, INF_VALUE, bestAlternativeParent,
                   bestAlternativeDistance);

    parent[v] = bestAlternativeParent;
    distances[v] = bestAlternativeDistance;

    if (!isAffected[v]) {
      isAffected[v] = 1;
      affectedVertices.push_back(v);
    }
  }

  // --- 1c. Process weight increases on existing edges ---
  for (const auto &wi : weightIncreases) {
    int u = wi.from;
    int v = wi.to;

    if (parent[v] != u) {
      continue;
    }

    int bestNewParent = -1;
    long long bestNewDistance = INF_VALUE;
    findBestParent(v, inAdjacency, distances, INF_VALUE, bestNewParent,
                   bestNewDistance);

    parent[v] = bestNewParent;
    distances[v] = bestNewDistance;

    if (!isAffected[v]) {
      isAffected[v] = 1;
      affectedVertices.push_back(v);
    }
  }

  step1Stage.stop();
  recordCounter("sosp/initial_affected", affectedVertices.size());

  // ========================================================================
  // PHASE 2: PROPAGATE THE UPDATE (Parallel — OpenMP)
  // ========================================================================
  ScopedStage propagateStage("sosp/2_propagate_compute");
  // Iteratively propagate changes through the graph until convergence.
  // Uses Chaotic Bellman-Ford semantics: concurrent reads of distances[]
  // during findBestParent are benign races that do not affect final
  // correctness — convergence to the global shortest-path optimum is
  // guaranteed.

  int iterationCount = 0;
  const int maxIterations = numberOfNodes;

  while (!affectedVertices.empty() && iterationCount < maxIterations) {
    ++iterationCount;

    // --- 2a. Identify candidate vertices (out-neighbors of affected) ---
    // Use char array for atomic compare-exchange deduplication.
    vector<char> isCandidate(numberOfNodes, 0);
    vector<int> candidateVertices;

    int numAffected = static_cast<int>(affectedVertices.size());

#pragma omp parallel
    {
      vector<int> localCandidates;

#pragma omp for schedule(dynamic, 64)
      for (int i = 0; i < numAffected; ++i) {
        int affectedVertex = affectedVertices[i];
        isAffected[affectedVertex] = 0; // Clear affected flag

        for (const auto &outNeighbor : outAdjacency[affectedVertex]) {
          int neighborVertex = outNeighbor.vertex;

          // CRITICAL: Never update the source vertex
          if (neighborVertex == source) {
            continue;
          }

          // Atomic test-and-set for deduplication.
          // Only the thread that successfully flips 0→1 adds the vertex.
          char expected = 0;
          if (__atomic_compare_exchange_n(
                  &isCandidate[neighborVertex], &expected, static_cast<char>(1),
                  false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            localCandidates.push_back(neighborVertex);
          }
        }
      }

// Merge thread-local candidates into the shared vector
#pragma omp critical
      {
        candidateVertices.insert(candidateVertices.end(),
                                 localCandidates.begin(),
                                 localCandidates.end());
      }
    }

    affectedVertices.clear();

    // --- 2b. Update distances of candidate vertices ---
    // Each candidate v is unique (deduplicated above), so writes to
    // distances[v] and parent[v] are race-free. Reads of distances[u]
    // in findBestParent may see stale values — this is the Chaotic
    // Bellman-Ford benign race.
    int numCandidates = static_cast<int>(candidateVertices.size());

#pragma omp parallel
    {
      vector<int> localAffected;

#pragma omp for schedule(dynamic, 64)
      for (int i = 0; i < numCandidates; ++i) {
        int candidateVertex = candidateVertices[i];

        int bestNewParent = -1;
        long long bestNewDistance = INF_VALUE;
        findBestParent(candidateVertex, inAdjacency, distances, INF_VALUE,
                       bestNewParent, bestNewDistance);

        // ALWAYS update parent to current best (parent consistency fix).
        // Only propagate further if the DISTANCE actually changed.
        bool distanceChanged = (bestNewDistance != distances[candidateVertex]);

        // No race: each candidateVertex is unique in the list
        parent[candidateVertex] = bestNewParent;
        distances[candidateVertex] = bestNewDistance;

        if (distanceChanged) {
          // Atomic test-and-set for deduplication in affected list
          char expected = 0;
          if (__atomic_compare_exchange_n(
                  &isAffected[candidateVertex], &expected, static_cast<char>(1),
                  false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            localAffected.push_back(candidateVertex);
          }
        }
      }

// Merge thread-local affected lists into the shared vector
#pragma omp critical
      {
        affectedVertices.insert(affectedVertices.end(), localAffected.begin(),
                                localAffected.end());
      }
    }
  }

  propagateStage.stop();
  recordCounter("sosp/iterations", iterationCount);

  if (iterationCount >= maxIterations && !affectedVertices.empty()) {
    cout << "Warning: SOSP update reached maximum iteration limit ("
         << maxIterations << "). Running reachability check.\n";
  }

  // ========================================================================
  // POST-PROCESSING: REACHABILITY CHECK (Parallel BFS)
  // ========================================================================
  ScopedStage bfsStage("sosp/3_bfs_reachability_compute");
  // Level-synchronous parallel BFS from the source vertex.
  // Uses atomic compare-exchange on a char array for visited flags.

  {
    vector<char> reachable(numberOfNodes, 0);
    reachable[source] = 1;

    vector<int> frontier;
    frontier.push_back(source);

    while (!frontier.empty()) {
      int frontierSize = static_cast<int>(frontier.size());
      vector<int> nextFrontier;

#pragma omp parallel
      {
        vector<int> localNext;

#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < frontierSize; ++i) {
          int current = frontier[i];
          for (const auto &neighbor : outAdjacency[current]) {
            // Atomic test-and-set: only the thread that flips
            // 0→1 adds the vertex to the next frontier.
            char expected = 0;
            if (__atomic_compare_exchange_n(&reachable[neighbor.vertex],
                                            &expected, static_cast<char>(1),
                                            false, __ATOMIC_RELAXED,
                                            __ATOMIC_RELAXED)) {
              localNext.push_back(neighbor.vertex);
            }
          }
        }

#pragma omp critical
        {
          nextFrontier.insert(nextFrontier.end(), localNext.begin(),
                              localNext.end());
        }
      }

      frontier = move(nextFrontier);
    }

// Mark unreachable vertices (trivially parallel)
#pragma omp parallel for schedule(static)
    for (int v = 0; v < numberOfNodes; ++v) {
      if (!reachable[v]) {
        distances[v] = INF_VALUE;
        parent[v] = -1;
      }
    }
  }

  bfsStage.stop();

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
