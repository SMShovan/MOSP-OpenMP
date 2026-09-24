#include <string>
#include <vector>

#include "dijkstra.h"
#include "generateChangedEdges.h"
#include "generateGraph.h"
#include "generateGraphCSR.h"
#include "generateTestCases.h"
#include "parallelCombinedGraph.h"
#include "parallelSOSPUpdate.h"
#include "sequentialSOSPUpdate.h"
#include "updateGraphCSR.h"

using namespace std;

int main() {
  int numberOfNodes = 5;
  int numberOfEdges = 6;
  bool directed = true;
  string outputFile = "data/graph.mtx";
  int numberOfObjectives = 3;
  int objectiveStartRange = 1;
  int objectiveEndRange = 9;
  int objectiveNumber = 0;
  int source = 0;
  int numberOfChangedEdges = 4;
  double insertionPercentage = 50.0;
  double deletionPercentage = 50.0;

  if (!generateGraph(numberOfNodes, numberOfEdges, directed, outputFile,
                     numberOfObjectives, objectiveStartRange,
                     objectiveEndRange)) {
    return 1;
  }

  const string originalGraphPrefix = "data/originalGraph/graphCsr";
  const string updatedGraphPrefix = "data/updatedGraph/updatedGraphCsr";
  const string distancesTreesDir = "output/distancesTrees";
  const string updatedDistancesTreesDir = "output/updatedDistancesTrees";

  if (!generateGraphCSR(numberOfNodes, numberOfEdges, directed,
                        originalGraphPrefix, numberOfObjectives,
                        objectiveStartRange, objectiveEndRange)) {
    return 1;
  }

  if (!generateChangedEdges(objectiveStartRange, objectiveEndRange,
                            numberOfObjectives, numberOfNodes,
                            numberOfChangedEdges, insertionPercentage,
                            deletionPercentage, directed, true, true, false,
                            originalGraphPrefix)) {
    return 1;
  }

  if (!updateGraphCSR(originalGraphPrefix, updatedGraphPrefix,
                      "output/changedEdges/insert.txt",
                      "output/changedEdges/delete.txt", directed)) {
    return 1;
  }

  if (!runDijkstra(outputFile, objectiveNumber, source,
                   distancesTreesDir + "/distances.txt",
                   distancesTreesDir + "/SSSPTree.txt")) {
    return 1;
  }

  if (!runDijkstraCSR(originalGraphPrefix, objectiveNumber, source,
                      distancesTreesDir + "/distancesCsr.txt",
                      distancesTreesDir + "/SSSPTreeCsr.txt")) {
    return 1;
  }

  if (!runDijkstraCSR(updatedGraphPrefix, objectiveNumber, source,
                      updatedDistancesTreesDir + "/updatedDistancesCsr.txt",
                      updatedDistancesTreesDir + "/updatedSSSPTreeCsr.txt")) {
    return 1;
  }

  const string sospUpdateDir = "output/sospUpdateDistancesTrees";

  if (!sequentialSOSPUpdate(
          originalGraphPrefix, distancesTreesDir + "/distancesCsr.txt",
          distancesTreesDir + "/SSSPTreeCsr.txt",
          "output/changedEdges/insert.txt", "output/changedEdges/delete.txt",
          objectiveNumber, source, sospUpdateDir + "/distancesCsr.txt",
          sospUpdateDir + "/SSSPTreeCsr.txt")) {
    return 1;
  }

  // =========================================================================
  // Run parallelSOSPUpdate for each of the 3 objectives independently.
  // Each produces an updated SSSP tree (parent array) after the edge changes.
  // =========================================================================

  vector<string> objTreePaths(numberOfObjectives);

  // Initial SOSP trees of every objective first: they are inputs of the
  // update, so they are not computed inside the update loop.
  for (int obj = 0; obj < numberOfObjectives; ++obj) {
    const string objDir = "output/parallelSospObj" + to_string(obj);
    if (!runDijkstraCSR(originalGraphPrefix, obj, source,
                        objDir + "/distancesOriginal.txt",
                        objDir + "/SSSPTreeOriginal.txt")) {
      return 1;
    }
  }

  for (int obj = 0; obj < numberOfObjectives; ++obj) {
    const string objDir = "output/parallelSospObj" + to_string(obj);

    const string objDistOriginal = objDir + "/distancesOriginal.txt";
    const string objTreeOriginal = objDir + "/SSSPTreeOriginal.txt";
    const string objDistOut = objDir + "/distancesUpdated.txt";
    const string objTreeOut = objDir + "/SSSPTreeUpdated.txt";

    if (!parallelSOSPUpdate(originalGraphPrefix, objDistOriginal,
                            objTreeOriginal, "output/changedEdges/insert.txt",
                            "output/changedEdges/delete.txt", obj, source,
                            objDistOut, objTreeOut)) {
      return 1;
    }

    objTreePaths[obj] = objTreeOut;
  }

  // =========================================================================
  // Build the combined graph from the 3 SSSP trees and find its SSSP.
  // Edge weight = K + 1 - (number of trees the edge belongs to):
  //   belongs to all 3 trees → weight 1  (highest preference)
  //   belongs to exactly 2   → weight 2
  //   belongs to exactly 1   → weight 3  (lowest preference)
  // =========================================================================

  const string combinedGraphDir = "output/combinedGraph";

  if (!parallelCombinedGraph(originalGraphPrefix, objTreePaths,
                             numberOfObjectives, // K = 3
                             source, combinedGraphDir,
                             combinedGraphDir + "/distancesCsr.txt",
                             combinedGraphDir + "/SSSPTreeCsr.txt")) {
    return 1;
  }

  if (!generateTestCases("tests")) {
    return 1;
  }

  return 0;
}
