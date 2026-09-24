/**
 * @file Dijkstra.cpp
 * @brief Single-objective Dijkstra shortest path tree generator.
 */

#include "dijkstra.h"

#include "csrGraph.h"
#include "read.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <string>
#include <utility>
#include <vector>

using namespace std;

/**
 * @brief Compute single-objective shortest paths and write output files.
 *
 * @param inputFile Path to the Matrix Market input file.
 * @param objectiveNumber 0-indexed objective to use as edge weight.
 * @param source Source vertex (0-indexed).
 * @param distanceOutputPath Output path for distance values.
 * @param treeOutputPath Output path for shortest-path tree.
 * @return True on success; false otherwise.
 */
bool runDijkstra(
    const string &inputFile,
    int objectiveNumber,
    int source,
    const string &distanceOutputPath,
    const string &treeOutputPath
) {
    Graph graph;
    int numberOfObjectives = 0;
    if (!readMtx(inputFile, graph, numberOfObjectives)) {
        return false;
    }

    // A graph without edges has no weights (the objective count cannot be
    // inferred from an empty values file), so any objective index is fine.
    if (objectiveNumber < 0 ||
        (numberOfObjectives > 0 && objectiveNumber >= numberOfObjectives)) {
        cout << "Error: objectiveNumber out of range.\n";
        return false;
    }

    int n = static_cast<int>(graph.size());
    if (source < 0 || source >= n) {
        cout << "Error: source vertex out of range.\n";
        return false;
    }

    const long long INF = numeric_limits<long long>::max() / 4;
    vector<long long> dist(n, INF);
    vector<int> parent(n, -1);

    using Node = pair<long long, int>;
    priority_queue<Node, vector<Node>, greater<Node>> pq;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) {
            continue;
        }

        for (const auto &edge : graph[u]) {
            int v = edge.to;
            int w = edge.weights[objectiveNumber];
            // Ties go to the lowest parent id (canonical SOSP tree).
            long long candidate = dist[u] + w;
            if (candidate < dist[v]) {
                dist[v] = candidate;
                parent[v] = u;
                pq.push({dist[v], v});
            } else if (candidate == dist[v] && v != source && u < parent[v]) {
                parent[v] = u;
            }
        }
    }

    filesystem::path distancePath(distanceOutputPath);
    if (!distancePath.parent_path().empty()) {
        filesystem::create_directories(distancePath.parent_path());
    }
    ofstream distOut(distanceOutputPath);
    if (!distOut.is_open()) {
        cout << "Error: Could not write distance output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        distOut << i << " ";
        if (dist[i] >= INF / 2) {
            distOut << "INF";
        } else {
            distOut << dist[i];
        }
        distOut << "\n";
    }

    filesystem::path treePath(treeOutputPath);
    if (!treePath.parent_path().empty()) {
        filesystem::create_directories(treePath.parent_path());
    }
    ofstream treeOut(treeOutputPath);
    if (!treeOut.is_open()) {
        cout << "Error: Could not write tree output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        treeOut << i << " " << parent[i] << "\n";
    }

    return true;
}

/**
 * @brief Compute single-objective shortest paths from CSR format and write output files.
 *
 * @param inputPrefix Base path to CSR files (e.g. "data/originalGraph/graphCsr").
 * @param objectiveNumber 0-indexed objective to use as edge weight.
 * @param source Source vertex (0-indexed).
 * @param distanceOutputPath Output path for distance values.
 * @param treeOutputPath Output path for shortest-path tree.
 * @return True on success; false otherwise.
 */
bool runDijkstraCSR(
    const string &inputPrefix,
    int objectiveNumber,
    int source,
    const string &distanceOutputPath,
    const string &treeOutputPath
) {
    Graph graph;
    int numberOfObjectives = 0;
    if (!readCSR(inputPrefix, graph, numberOfObjectives)) {
        return false;
    }

    // A graph without edges has no weights (the objective count cannot be
    // inferred from an empty values file), so any objective index is fine.
    if (objectiveNumber < 0 ||
        (numberOfObjectives > 0 && objectiveNumber >= numberOfObjectives)) {
        cout << "Error: objectiveNumber out of range.\n";
        return false;
    }

    int n = static_cast<int>(graph.size());
    if (source < 0 || source >= n) {
        cout << "Error: source vertex out of range.\n";
        return false;
    }

    const long long INF = numeric_limits<long long>::max() / 4;
    vector<long long> dist(n, INF);
    vector<int> parent(n, -1);

    using Node = pair<long long, int>;
    priority_queue<Node, vector<Node>, greater<Node>> pq;

    dist[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != dist[u]) {
            continue;
        }

        for (const auto &edge : graph[u]) {
            int v = edge.to;
            int w = edge.weights[objectiveNumber];
            // Ties go to the lowest parent id (canonical SOSP tree).
            long long candidate = dist[u] + w;
            if (candidate < dist[v]) {
                dist[v] = candidate;
                parent[v] = u;
                pq.push({dist[v], v});
            } else if (candidate == dist[v] && v != source && u < parent[v]) {
                parent[v] = u;
            }
        }
    }

    filesystem::path distancePath(distanceOutputPath);
    if (!distancePath.parent_path().empty()) {
        filesystem::create_directories(distancePath.parent_path());
    }
    ofstream distOut(distanceOutputPath);
    if (!distOut.is_open()) {
        cout << "Error: Could not write CSR distance output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        distOut << i << " ";
        if (dist[i] >= INF / 2) {
            distOut << "INF";
        } else {
            distOut << dist[i];
        }
        distOut << "\n";
    }

    filesystem::path treePath(treeOutputPath);
    if (!treePath.parent_path().empty()) {
        filesystem::create_directories(treePath.parent_path());
    }
    ofstream treeOut(treeOutputPath);
    if (!treeOut.is_open()) {
        cout << "Error: Could not write CSR tree output file.\n";
        return false;
    }

    for (int i = 0; i < n; ++i) {
        treeOut << i << " " << parent[i] << "\n";
    }

    return true;
}

/**
 * @brief Single-objective Dijkstra on an in-memory CSR graph.
 *
 * @see dijkstra.h
 */
void dijkstraCsrGraph(const CsrGraph &graph, int objective, int source,
                      vector<long long> &distances, vector<int> &parent) {
    const int n = graph.numberOfNodes;
    distances.assign(n, DISTANCE_INF);
    parent.assign(n, -1);
    if (source < 0 || source >= n) {
        return;
    }

    using Node = pair<long long, int>;
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    distances[source] = 0;
    pq.push({0, source});

    while (!pq.empty()) {
        auto [d, u] = pq.top();
        pq.pop();
        if (d != distances[u]) {
            continue;
        }
        for (int e = graph.rowPtr[u]; e < graph.rowPtr[u + 1]; ++e) {
            int v = graph.colInd[e];
            long long candidate = d + graph.weight(e, objective);
            if (candidate < distances[v]) {
                distances[v] = candidate;
                parent[v] = u;
                pq.push({candidate, v});
            } else if (candidate == distances[v] && v != source &&
                       u < parent[v]) {
                parent[v] = u; // ties go to the lowest parent id
            }
        }
    }
}
