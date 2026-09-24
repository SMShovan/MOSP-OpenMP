/**
 * @file parallelCombinedGraph.cpp
 * @brief Build a combined graph from K SSSP trees and find its SSSP.
 *
 * ============================================================================
 * ALGORITHM OVERVIEW
 * ============================================================================
 *
 * Given K SSSP trees (one per objective), produced by K independent runs of
 * parallelSOSPUpdate, this module:
 *
 *   Phase 0 — Read & extract tree edges
 *     For each of the K parent arrays, derive the set of directed tree edges.
 *     An edge (u → v) is in tree k if parent_k[v] == u.
 *
 *   Phase 1 — Count membership & assign weights
 *     For every unique edge across all K trees, record which trees contain
 *     it and assign the preference weight (scaled by L = lcm(Pref)):
 *       W'(e) = L * (K + 1) - sum_{i : e in T_i} L / Pref_i
 *     With the default Pref = (1,...,1) this is w = K + 1 - m:
 *       m == K  →  w = 1  (appears in ALL trees — highest preference)
 *       ...
 *       m == 1  →  w = K  (appears in ONE tree — lowest preference)
 *
 *   Phase 2 — Write temporary input files for parallelSOSPUpdate
 *     Write an EMPTY base CSR (same vertex count, no edges).
 *     Write blank initial distances (dist[source]=0, rest=INF) and
 *       a blank SSSP tree (all parents = -1).
 *     Write all combined-graph edges as an insertion file.
 *     Write an empty deletion file.
 *
 *   Phase 3 — Run parallelSOSPUpdate on the combined graph
 *     With no base edges and all combined edges treated as insertions,
 *     Phase 1 of parallelSOSPUpdate seeds affectedVertices from the source,
 *     and Phase 2 propagates exactly like Bellman-Ford — correct & parallel.
 *
 * ============================================================================
 * PARALLELISM
 * ============================================================================
 *
 * Edge membership counting (Phase 1) is parallelized with OpenMP:
 *   - Each thread fills a thread-local map<pair<int,int>, int>.
 *   - Maps are merged under #pragma omp critical.
 * File I/O (Phases 2 output) is sequential (I/O dominated, tiny data).
 * Phase 3 is fully parallel (delegated to parallelSOSPUpdate).
 *
 * ============================================================================
 */

#include "parallelCombinedGraph.h"

#include "csrGraph.h"
#include "parallelSOSPUpdate.h"
#include "read.h"
#include "stageTimer.h"

#include <omp.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace {

/**
 * @brief Read a parent array from a Dijkstra / parallelSOSPUpdate tree file.
 *
 * Expected format per line: "vertexId parentId"  (parentId == -1 for source).
 *
 * @param path          Path to the SSP tree file.
 * @param parent        Output parent array.
 * @param numberOfNodes Expected vertex count.
 * @return True on success; false otherwise.
 */
bool readParent(const string &path, vector<int> &parent, int numberOfNodes) {
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
    istringstream ss(line);
    int v, p;
    if (!(ss >> v >> p))
      continue;
    if (v < 0 || v >= numberOfNodes) {
      cout << "Error: Vertex ID " << v << " out of range in: " << path << "\n";
      return false;
    }
    parent[v] = p;
  }
  return true;
}

} // namespace

long long preferenceScale(const vector<int> &preferences, int K) {
  if (preferences.empty()) {
    return 1;
  }
  if (static_cast<int>(preferences.size()) != K) {
    return 0;
  }
  long long scale = 1;
  for (int pref : preferences) {
    if (pref < 1) {
      return 0;
    }
    scale = scale / gcd(scale, static_cast<long long>(pref)) * pref;
    if (scale > (1LL << 20)) {
      return 0;
    }
  }
  return scale;
}

long long combinedEdgeWeight(unsigned int treeMask,
                             const vector<int> &preferences, int K,
                             long long scale) {
  long long weight = scale * (K + 1);
  for (int i = 0; i < K; ++i) {
    if (treeMask & (1u << i)) {
      weight -= preferences.empty() ? scale : scale / preferences[i];
    }
  }
  return weight;
}

bool mospPathCosts(const CsrGraph &graph, const vector<int> &parent,
                   int source, vector<long long> &costs) {
  const int n = graph.numberOfNodes;
  const int K = graph.numberOfObjectives;
  costs.assign(static_cast<size_t>(n) * K, DISTANCE_INF);
  // Children lists of the tree, then a traversal from the source.
  vector<int> childStart(n + 1, 0), children(n);
  for (int v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      ++childStart[parent[v] + 1];
    }
  }
  for (int v = 0; v < n; ++v) {
    childStart[v + 1] += childStart[v];
  }
  vector<int> cursor(childStart.begin(), childStart.end() - 1);
  for (int v = 0; v < n; ++v) {
    if (v != source && parent[v] >= 0) {
      children[cursor[parent[v]]++] = v;
    }
  }
  for (int k = 0; k < K; ++k) {
    costs[static_cast<size_t>(source) * K + k] = 0;
  }
  vector<int> queue{source};
  for (size_t i = 0; i < queue.size(); ++i) {
    int p = queue[i];
    for (int c = childStart[p]; c < childStart[p + 1]; ++c) {
      int v = children[c];
      int edge = -1;
      for (int e = graph.rowPtr[p]; e < graph.rowPtr[p + 1]; ++e) {
        if (graph.colInd[e] == v) {
          edge = e;
          break;
        }
      }
      if (edge < 0) {
        return false;
      }
      for (int k = 0; k < K; ++k) {
        costs[static_cast<size_t>(v) * K + k] =
            costs[static_cast<size_t>(p) * K + k] + graph.weight(edge, k);
      }
      queue.push_back(v);
    }
  }
  return true;
}

/**
 * @brief Build a combined graph from K SSSP trees and find its SSSP.
 *
 * @see parallelCombinedGraph.h for full parameter documentation.
 */
bool parallelCombinedGraph(const string &originalCsrPrefix,
                           const vector<string> &treeInputPaths, int K,
                           int source, const string &workDir,
                           const string &distancesOutputPath,
                           const string &treeOutputPath,
                           const vector<int> &preferences) {

  if (K <= 0 || static_cast<int>(treeInputPaths.size()) < K) {
    cout << "Error: K=" << K << " but only " << treeInputPaths.size()
         << " tree paths supplied.\n";
    return false;
  }

  if (K > 32) {
    cout << "Error: at most 32 objectives are supported.\n";
    return false;
  }
  const long long scale = preferenceScale(preferences, K);
  if (scale == 0) {
    cout << "Error: invalid preference vector (need K values >= 1).\n";
    return false;
  }

  // ========================================================================
  // PHASE 0: READ BASE GRAPH (for vertex count) AND ALL K PARENT ARRAYS
  // ========================================================================

  // --- 0a. Determine numberOfNodes from the CSR ---
  ScopedStage readStage("comb/0a_read_csr_for_n");
  Graph baseGraph;
  int numberOfObjectives = 0;
  if (!readCSR(originalCsrPrefix, baseGraph, numberOfObjectives)) {
    cout << "Error: Could not read original CSR graph.\n";
    return false;
  }
  int numberOfNodes = static_cast<int>(baseGraph.size());
  baseGraph.clear(); // free immediately — we only needed the vertex count

  if (numberOfNodes == 0) {
    cout << "Error: Graph has no vertices.\n";
    return false;
  }
  if (source < 0 || source >= numberOfNodes) {
    cout << "Error: source vertex " << source << " out of range.\n";
    return false;
  }

  readStage.stop();

  // --- 0b. Read K parent arrays ---
  ScopedStage treeStage("comb/0b_read_trees_text");
  vector<vector<int>> parents(K);
  for (int k = 0; k < K; ++k) {
    if (!readParent(treeInputPaths[k], parents[k], numberOfNodes)) {
      return false;
    }
  }

  // ========================================================================
  // PHASE 1: COUNT EDGE MEMBERSHIP & ASSIGN WEIGHTS (OpenMP parallel)
  // ========================================================================
  //
  // For each tree k and each vertex v (v != source, parent_k[v] != -1),
  // the tree edge is (parent_k[v] → v).
  // We count how many trees each directed edge belongs to, then assign weight.

  treeStage.stop();
  ScopedStage mapStage("comb/1_membership_map_compute");

  // Outer loop over K trees is only size 3 (or small K) — parallelize the
  // inner loop over vertices instead.
  map<pair<int, int>, unsigned int>
      edgeMembership; // edge → bit mask of the trees it appears in

#pragma omp parallel
  {
    map<pair<int, int>, unsigned int> localMap;

    for (int k = 0; k < K; ++k) {
      const vector<int> &par = parents[k];

#pragma omp for schedule(static) nowait
      for (int v = 0; v < numberOfNodes; ++v) {
        if (v == source)
          continue;
        int u = par[v];
        if (u < 0 || u >= numberOfNodes)
          continue; // unreachable vertex or invalid parent

        localMap[{u, v}] |= 1u << k;
      }
    }

#pragma omp critical
    {
      for (const auto &kv : localMap) {
        edgeMembership[kv.first] |= kv.second;
      }
    }
  }

  // ========================================================================
  // PHASE 2: WRITE TEMPORARY FILES FOR parallelSOSPUpdate
  // ========================================================================

  mapStage.stop();
  recordCounter("comb/edges", edgeMembership.size());
  ScopedStage writeStage("comb/2_write_temp_files");

  // Create the work directory
  filesystem::create_directories(workDir);

  // --- 2a. Write EMPTY base CSR (same vertex count, zero edges) ---
  //
  // CSR format expected by readCSR:
  //   prefixRowPtr.txt   — n+1 zeros (row offsets, all 0 → no edges)
  //   prefixColInd.txt   — empty (no column indices)
  //   prefixValues.txt   — empty (no values)
  //
  // parallelSOSPUpdate passes objectiveIndex=0 and reads weights[0]; the
  // combined graph has exactly 1 objective weight per edge, so
  // numberOfObjectives=1.

  const string tempCsrPrefix = workDir + "/tempCsr";

  // Write a minimal temp CSR with a single dummy self-loop on the source
  // vertex (source → source, weight 0, 1 objective).  This edge is inert:
  // parallelSOSPUpdate never updates the source vertex (it is guarded by
  // "if (neighborVertex == source) continue"), so the self-loop can never
  // propagate or affect any distance. Its only purpose is to make readCSR
  // infer numberOfObjectives=1 (otherwise an empty Values file gives 0,
  // which causes parallelSOSPUpdate to reject objectiveIndex=0).
  {
    // RowPtr: source row has 1 outgoing edge; all others have 0.
    // row_ptr[i+1] - row_ptr[i] = number of edges from vertex i.
    // source row: ptr goes 0 → 1; all subsequent rows stay at 1.
    ofstream rowFile(tempCsrPrefix + "RowPtr.txt");
    if (!rowFile.is_open()) {
      cout << "Error: Could not write temp CSR RowPtr file.\n";
      return false;
    }
    for (int i = 0; i <= numberOfNodes; ++i) {
      // ptr[0..source] = 0; ptr[source+1..n] = 1
      rowFile << (i <= source ? 0 : 1) << "\n";
    }
  }
  {
    // ColInd: the single edge goes to 'source' itself.
    ofstream colFile(tempCsrPrefix + "ColInd.txt");
    if (!colFile.is_open()) {
      cout << "Error: Could not write temp CSR ColInd file.\n";
      return false;
    }
    colFile << source << "\n";
  }
  {
    // Values: weight 0 for the dummy self-loop (1 objective).
    ofstream valFile(tempCsrPrefix + "Values.txt");
    if (!valFile.is_open()) {
      cout << "Error: Could not write temp CSR Values file.\n";
      return false;
    }
    valFile << "0\n"; // one edge, one objective, weight 0
  }

  // --- 2b. Write blank initial distances (source=0, rest=INF) ---
  const string blankDistPath = workDir + "/blankDistances.txt";
  {
    ofstream distFile(blankDistPath);
    if (!distFile.is_open()) {
      cout << "Error: Could not write blank distances file.\n";
      return false;
    }
    for (int v = 0; v < numberOfNodes; ++v) {
      distFile << v << " ";
      if (v == source) {
        distFile << "0";
      } else {
        distFile << "INF";
      }
      distFile << "\n";
    }
  }

  // --- 2c. Write blank initial SSSP tree (all parents = -1) ---
  const string blankTreePath = workDir + "/blankTree.txt";
  {
    ofstream treeFile(blankTreePath);
    if (!treeFile.is_open()) {
      cout << "Error: Could not write blank tree file.\n";
      return false;
    }
    for (int v = 0; v < numberOfNodes; ++v) {
      treeFile << v << " -1\n";
    }
  }

  // --- 2d. Write combined-graph edges as insertion file ---
  //
  // Format expected by parallelSOSPUpdate: "u v w1" (1 objective weight).
  // Weight = L * (K + 1) - sum over the trees containing the edge of
  // L / Pref_i; with Pref = 1s: K + 1 - membershipCount (all-3 → 1, ...).

  const string insertPath = workDir + "/insert.txt";
  {
    ofstream insertFile(insertPath);
    if (!insertFile.is_open()) {
      cout << "Error: Could not write combined graph insert file.\n";
      return false;
    }
    for (const auto &entry : edgeMembership) {
      int u = entry.first.first;
      int v = entry.first.second;
      long long w = combinedEdgeWeight(entry.second, preferences, K, scale);
      insertFile << u << " " << v << " " << w << "\n";
    }
  }

  // --- 2e. Write empty deletion file ---
  const string deletePath = workDir + "/delete.txt";
  {
    ofstream deleteFile(deletePath);
    if (!deleteFile.is_open()) {
      cout << "Error: Could not write combined graph delete file.\n";
      return false;
    }
    // intentionally empty — no deletions
  }

  // ========================================================================
  // PHASE 3: RUN parallelSOSPUpdate ON THE COMBINED GRAPH
  // ========================================================================
  //
  // objectiveIndex = 0  (the combined graph has exactly 1 weight per edge)
  // The empty base CSR + "all edges as insertions" pattern causes Phase 1 of
  // parallelSOSPUpdate to relax every combined edge from the source outward,
  // and Phase 2 propagates like Chaotic Bellman-Ford until convergence.

  writeStage.stop();

  cout << "parallelCombinedGraph: running parallelSOSPUpdate on combined graph"
       << " (" << edgeMembership.size() << " edges, " << numberOfNodes
       << " vertices)...\n";

  const string outerPrefix = stagePrefix();
  setStagePrefix(outerPrefix + "comb/");
  bool updated = parallelSOSPUpdate(
      tempCsrPrefix, blankDistPath, blankTreePath, insertPath, deletePath,
      /*objectiveIndex=*/0, source, distancesOutputPath, treeOutputPath);
  setStagePrefix(outerPrefix);
  if (!updated) {
    cout << "Error: parallelSOSPUpdate failed on combined graph.\n";
    return false;
  }

  cout << "parallelCombinedGraph: done. Output written to:\n"
       << "  Distances: " << distancesOutputPath << "\n"
       << "  SSSP Tree: " << treeOutputPath << "\n";

  return true;
}
