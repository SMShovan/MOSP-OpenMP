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
 * is sequential.
 *
 * Phase 1 (OpenMP): the head v of every deleted or weight-increased edge
 * (u,v) with parent[v] == u is a root; the SOSP subtree of every root is
 * invalidated (distance INF, parent -1) — each vertex walks up its parent
 * chain to the first vertex with a known state and writes that state along
 * the path. The invalidated vertices and the heads of inserted edges then
 * pull the best (distance, parent id) pair over their in-neighbours.
 *
 * Phase 2 (OpenMP): the thesis' propagation loop — collect the
 * out-neighbours of the affected vertices, re-evaluate each candidate —
 * with a monotone update (a vertex only takes a strictly better pair).
 * Distances only decrease, so the loop needs no iteration cap, cannot
 * "count to infinity" through a stale cycle, and vertices cut off from the
 * source keep INF without a reachability post-pass.
 *
 * Race Condition Analysis:
 *   - isCandidate[]/isAffected[]: atomic compare-exchange on char arrays
 *   - candidateVertices/affectedVertices: thread-local + critical merge
 *   - distances[v]/parent[v] writes in Step 2b: no race because each
 *     candidate v appears exactly once (deduplicated)
 *   - distances[u] reads in findBestParent: benign race under Chaotic
 *     Bellman-Ford semantics — any change marks u affected, so v is
 *     re-evaluated in the next iteration
 *   - invalidation states: relaxed atomic char loads/stores; concurrent
 *     walks over a shared path write the same state
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
  // PHASE 1: PROCESS CHANGED EDGES (Parallel — OpenMP)
  // ========================================================================
  // 1a. Roots: the head v of every deleted or weight-increased edge (u,v)
  //     with parent[v] == u.
  // 1b. Invalidate the SOSP subtree of every root (distance INF, parent -1).
  //     Every vertex walks up its parent chain until it meets a vertex whose
  //     state is known (a root: invalid; the tree root: valid) and writes
  //     that state along the path, so each vertex is resolved about once.
  //     Concurrent walks over a shared path write the same state.
  // 1c. The invalidated vertices and the heads of all inserted edges are
  //     re-evaluated over their in-neighbours (monotone: a vertex only takes
  //     a strictly better (distance, parent id) pair).
  ScopedStage step1Stage("sosp/1_process_changes_compute");

  // State per vertex: 0 unknown, 1 valid, 2 invalid (in a root's subtree).
  vector<char> state(numberOfNodes, 0);
  auto loadState = [&](int v) {
    return __atomic_load_n(&state[v], __ATOMIC_RELAXED);
  };
  auto storeState = [&](int v, char s) {
    __atomic_store_n(&state[v], s, __ATOMIC_RELAXED);
  };
  auto markRoot = [&](int u, int v) {
    if (parent[v] == u) {
      storeState(v, 2);
    }
  };
  const int numDeleted = static_cast<int>(deletedEdges.size());
  const int numIncreased = static_cast<int>(weightIncreases.size());
#pragma omp parallel for schedule(static)
  for (int i = 0; i < numDeleted; ++i) {
    markRoot(deletedEdges[i].from, deletedEdges[i].to);
  }
#pragma omp parallel for schedule(static)
  for (int i = 0; i < numIncreased; ++i) {
    markRoot(weightIncreases[i].from, weightIncreases[i].to);
  }

  vector<char> isInitialCandidate(numberOfNodes, 0);
  vector<int> candidateVertices;
  if (numDeleted + numIncreased > 0) {
#pragma omp parallel
    {
      vector<int> localInvalid;
#pragma omp for schedule(dynamic, 1024)
      for (int v = 0; v < numberOfNodes; ++v) {
        if (loadState(v) != 0) {
          continue;
        }
        int u = v;
        while (loadState(u) == 0 && parent[u] >= 0) {
          u = parent[u];
        }
        char s = loadState(u) == 0 ? 1 : loadState(u); // tree root: valid
        for (u = v; u >= 0 && loadState(u) == 0; u = parent[u]) {
          storeState(u, s);
        }
      }
      // Every vertex is resolved now (the implicit barrier above).
#pragma omp for schedule(static)
      for (int v = 0; v < numberOfNodes; ++v) {
        if (state[v] == 2) {
          distances[v] = INF_VALUE;
          parent[v] = -1;
          isInitialCandidate[v] = 1;
          localInvalid.push_back(v);
        }
      }
#pragma omp critical
      candidateVertices.insert(candidateVertices.end(), localInvalid.begin(),
                               localInvalid.end());
    }
  }
  recordCounter("sosp/invalidated", candidateVertices.size());
  for (const auto &edge : insertedEdges) {
    int v = edge.to;
    if (v != source && !isInitialCandidate[v]) {
      isInitialCandidate[v] = 1;
      candidateVertices.push_back(v);
    }
  }

  // Re-evaluate candidate v; returns true if its distance decreased.
  auto relax = [&](int v) {
    int bestNewParent = -1;
    long long bestNewDistance = INF_VALUE;
    findBestParent(v, inAdjacency, distances, INF_VALUE, bestNewParent,
                   bestNewDistance);
    long long current = distances[v];
    bool better = bestNewDistance < current ||
                  (bestNewDistance == current && bestNewParent >= 0 &&
                   bestNewParent < parent[v]);
    if (!better) {
      return false;
    }
    distances[v] = bestNewDistance;
    parent[v] = bestNewParent;
    return bestNewDistance < current;
  };

  // Use char arrays instead of bool vectors for atomic compare-exchange
  vector<char> isAffected(numberOfNodes, 0);
  vector<int> affectedVertices;
  const int numCandidates0 = static_cast<int>(candidateVertices.size());
#pragma omp parallel
  {
    vector<int> localAffected;
#pragma omp for schedule(dynamic, 64)
    for (int i = 0; i < numCandidates0; ++i) {
      int v = candidateVertices[i];
      if (relax(v)) {
        char expected = 0;
        if (__atomic_compare_exchange_n(&isAffected[v], &expected,
                                        static_cast<char>(1), false,
                                        __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
          localAffected.push_back(v);
        }
      }
    }
#pragma omp critical
    affectedVertices.insert(affectedVertices.end(), localAffected.begin(),
                            localAffected.end());
  }

  // ========================================================================
  // PHASE 2: PROPAGATE THE UPDATE (Parallel — OpenMP)
  // ========================================================================
  // Iteratively propagate changes through the graph until convergence.
  // Uses Chaotic Bellman-Ford semantics: concurrent reads of distances[]
  // during findBestParent are benign races that do not affect final
  // correctness. The update is monotone (distances only decrease), so the
  // loop terminates without an iteration cap and vertices cut off from the
  // source keep the INF they got in Phase 1 (no reachability post-pass).
  step1Stage.stop();
  recordCounter("sosp/initial_affected", affectedVertices.size());
  ScopedStage propagateStage("sosp/2_propagate_compute");

  int iterationCount = 0;

  while (!affectedVertices.empty()) {
    ++iterationCount;
    if (iterationCount > numberOfNodes) {
      // Every sweep settles at least one more hop of every shortest path,
      // so this cannot happen.
      cout << "Error: SOSP update did not converge.\n";
      return false;
    }

    // --- 2a. Identify candidate vertices (out-neighbors of affected) ---
    // Use char array for atomic compare-exchange deduplication.
    vector<char> isCandidate(numberOfNodes, 0);
    candidateVertices.clear();

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

    // --- 2b. Update distances of candidate vertices (monotone) ---
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
        if (relax(candidateVertex)) {
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
