/**
 * @file sospUpdateCpu.cpp
 * @brief Work-efficient, disconnection-safe SOSP update with OpenMP.
 *
 * ============================================================================
 * ALGORITHM (the same as MOSP-CUDA's sospUpdateGpu)
 * ============================================================================
 *
 * Step 1 (from the change list, grouped by destination):
 *   - Roots: the head v of every deleted or weight-increased edge (u,v)
 *     with Parent[v] == u.
 *   - Subtree invalidation: every vertex walks up its parent chain to the
 *     first vertex whose state is known (a root: invalid; the tree root:
 *     valid) and writes that state along the path, so each vertex is
 *     resolved about once. The invalid vertices lose their distance (INF)
 *     and parent.
 *   - Pull pass: every invalidated vertex and every head of an inserted
 *     edge takes the best (distance, parent id) pair over its in-neighbours.
 *
 * Step 2 (propagation): a push-based near-far worklist (Delta-stepping
 * variant). Improved vertices relax their out-edges with an atomic minimum
 * (compare-and-swap) on the packed 64-bit word (distance << b | parent);
 * a head below the threshold joins the next near frontier (deduplicated by
 * generation stamps), otherwise the far pile; when the near frontier is
 * empty the threshold moves to the smallest far distance plus Delta.
 * Distances only decrease, so there is no iteration cap, no counting to
 * infinity and no reachability post-pass; ties go to the lowest parent id
 * (canonical trees); only improved vertices are expanded.
 *
 * Per-thread output lists are merged at prefix-sum offsets (ListGather).
 * If (n - 1) * maxWeight does not fit next to the parent ids, the words
 * hold the distance alone and parents are recovered in one pass over the
 * out-edges after the search.
 * ============================================================================
 */

#include "sospUpdateCpu.h"

#include "csrGraph.h"
#include "listGather.h"

#include <omp.h>

#include <algorithm>
#include <climits>
#include <iostream>
#include <vector>

using namespace std;

namespace {

using u64 = unsigned long long;
constexpr u64 PACKED_INF = ~0ULL;
constexpr u64 OUTPUT_MAX_DISTANCE = static_cast<u64>(DISTANCE_INF / 2 - 1);

/// Packed (distance, parent) words; parentBits == 0 means distance only.
struct Packing {
  int parentBits;
  u64 noParent;

  bool hasParents() const { return parentBits > 0; }
  u64 pack(u64 distance, int parent) const {
    if (parentBits == 0) {
      return distance;
    }
    return (distance << parentBits) |
           (parent < 0 ? noParent : static_cast<u64>(parent));
  }
  u64 distance(u64 word) const { return word >> parentBits; }
  int parent(u64 word) const {
    u64 p = word & noParent;
    return p == noParent ? -1 : static_cast<int>(p);
  }
  u64 maxDistance() const { return (PACKED_INF >> parentBits) - 1; }
};

Packing makePacking(int numberOfNodes, u64 bound) {
  int bits = 1;
  while ((1ULL << bits) - 1 < static_cast<u64>(numberOfNodes)) {
    ++bits;
  }
  Packing packed{bits, (1ULL << bits) - 1};
  return bound <= packed.maxDistance() ? packed : Packing{0, 0};
}

inline u64 load(const u64 *p) { return __atomic_load_n(p, __ATOMIC_RELAXED); }

/// Atomic minimum; returns the value before the operation.
inline u64 atomicMin(u64 *p, u64 value) {
  u64 old = load(p);
  while (value < old &&
         !__atomic_compare_exchange_n(p, &old, value, true, __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
  }
  return old;
}

inline void atomicMinInt(int *p, int value) {
  int old = __atomic_load_n(p, __ATOMIC_RELAXED);
  while (value < old &&
         !__atomic_compare_exchange_n(p, &old, value, true, __ATOMIC_RELAXED,
                                      __ATOMIC_RELAXED)) {
  }
}

inline bool claim(int *stamp, int v, int generation) {
  return __atomic_exchange_n(&stamp[v], generation, __ATOMIC_RELAXED) !=
         generation;
}

/// Smallest distance over a vertex list.
u64 smallestDistance(const vector<int> &list, const vector<u64> &packed,
                     const Packing &packing) {
  const int count = static_cast<int>(list.size());
  u64 smallest = PACKED_INF;
#pragma omp parallel for reduction(min : smallest) schedule(static)
  for (int i = 0; i < count; ++i) {
    u64 word = load(&packed[list[i]]);
    if (word != PACKED_INF) {
      smallest = min(smallest, packing.distance(word));
    }
  }
  return smallest;
}

/// Near-far propagation from ws.frontier.
void nearFar(const HostCsr &out, int source, u64 delta, const Packing &packing,
             SospWorkspace &ws, SospStats &stats) {
  if (ws.frontier.empty()) {
    return;
  }
  u64 *packed = ws.packed.data();
  int *stamp = ws.stamp.data();
  char *inFar = ws.inFar.data();
  const u64 smallest = smallestDistance(ws.frontier, ws.packed, packing);
  u64 threshold = (smallest == PACKED_INF ? 0 : smallest) + delta;

  vector<int> *current = &ws.nearA, *next = &ws.nearB;
  current->clear();
  ws.far.clear();
  {
    const int generation = ws.nextGeneration();
    const int count = static_cast<int>(ws.frontier.size());
    ListGather nearGather(*current), farGather(ws.far);
#pragma omp parallel
    {
      vector<int> localNear, localFar;
#pragma omp for schedule(static)
      for (int i = 0; i < count; ++i) {
        int v = ws.frontier[i];
        u64 word = load(&packed[v]);
        u64 d = word == PACKED_INF ? PACKED_INF : packing.distance(word);
        if (d < threshold) {
          if (claim(stamp, v, generation)) {
            localNear.push_back(v);
          }
        } else if (__atomic_exchange_n(&inFar[v], 1, __ATOMIC_RELAXED) == 0) {
          localFar.push_back(v);
        }
      }
      nearGather.gather(localNear);
      farGather.gather(localFar);
    }
  }

  while (true) {
    while (!current->empty()) {
      ++stats.iterations;
      stats.pushes += static_cast<long long>(current->size());
      const int generation = ws.nextGeneration();
      const int count = static_cast<int>(current->size());
      next->clear();
      ListGather nearGather(*next), farGather(ws.far);
#pragma omp parallel
      {
        vector<int> localNear, localFar;
#pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < count; ++i) {
          int u = (*current)[i];
          u64 word = load(&packed[u]);
          if (word == PACKED_INF) {
            continue;
          }
          u64 du = packing.distance(word);
          for (int e = out.rowPtr[u]; e < out.rowPtr[u + 1]; ++e) {
            int w = out.colInd[e];
            if (w == source) {
              continue;
            }
            u64 nd = du + static_cast<u64>(out.weights[e]);
            u64 candidate = packing.pack(nd, u);
            if (candidate >= load(&packed[w])) {
              continue;
            }
            u64 old = atomicMin(&packed[w], candidate);
            if (nd < packing.distance(old)) {
              if (nd < threshold) {
                if (claim(stamp, w, generation)) {
                  localNear.push_back(w);
                }
              } else if (__atomic_exchange_n(&inFar[w], 1, __ATOMIC_RELAXED) ==
                         0) {
                localFar.push_back(w);
              }
            }
          }
        }
        nearGather.gather(localNear);
        farGather.gather(localFar);
      }
      swap(current, next);
    }
    if (ws.far.empty()) {
      break;
    }
    // Raise the threshold past the far pile and re-split it.
    ++stats.epochs;
    threshold =
        max(threshold, smallestDistance(ws.far, ws.packed, packing)) + delta;
    const int generation = ws.nextGeneration();
    const int count = static_cast<int>(ws.far.size());
    current->clear();
    ws.far2.clear();
    ListGather nearGather(*current), keepGather(ws.far2);
#pragma omp parallel
    {
      vector<int> localNear, localKeep;
#pragma omp for schedule(static)
      for (int i = 0; i < count; ++i) {
        int v = ws.far[i];
        u64 word = load(&packed[v]);
        u64 d = word == PACKED_INF ? PACKED_INF : packing.distance(word);
        if (d < threshold) {
          inFar[v] = 0;
          if (claim(stamp, v, generation)) {
            localNear.push_back(v);
          }
        } else {
          localKeep.push_back(v);
        }
      }
      nearGather.gather(localNear);
      keepGather.gather(localKeep);
    }
    ws.far.swap(ws.far2);
  }
}

/// Write distances and parents from the packed words.
void unpack(const HostCsr &out, int source, const Packing &packing,
            const vector<u64> &packed, long long *distances, int *parent) {
  const int n = out.numberOfNodes;
  if (packing.hasParents()) {
#pragma omp parallel for schedule(static)
    for (int v = 0; v < n; ++v) {
      u64 word = packed[v];
      distances[v] = word == PACKED_INF
                         ? DISTANCE_INF
                         : static_cast<long long>(packing.distance(word));
      parent[v] = word == PACKED_INF ? -1 : packing.parent(word);
    }
    return;
  }
  // Distance-only words: recover the lowest-id parent over tight edges.
#pragma omp parallel for schedule(static)
  for (int v = 0; v < n; ++v) {
    u64 word = packed[v];
    distances[v] =
        word == PACKED_INF ? DISTANCE_INF : static_cast<long long>(word);
    parent[v] = word == PACKED_INF || v == source ? -1 : INT_MAX;
  }
#pragma omp parallel for schedule(dynamic, 256)
  for (int u = 0; u < n; ++u) {
    u64 du = packed[u];
    if (du == PACKED_INF) {
      continue;
    }
    for (int e = out.rowPtr[u]; e < out.rowPtr[u + 1]; ++e) {
      int w = out.colInd[e];
      if (w != source && du + static_cast<u64>(out.weights[e]) == packed[w]) {
        atomicMinInt(&parent[w], u);
      }
    }
  }
}

/// Choose the packing; fails only if distances could overflow 62 bits.
bool choosePacking(int n, long long maxWeight, Packing &packing, u64 &bound) {
  const u64 weight = static_cast<u64>(max(maxWeight, 1LL));
  const u64 hops = static_cast<u64>(max(n - 1, 1));
  if (weight > OUTPUT_MAX_DISTANCE / hops) {
    cerr << "Error: distances up to " << weight << " * " << hops
         << " do not fit in 62 bits.\n";
    return false;
  }
  bound = weight * hops;
  packing = makePacking(n, bound);
  return true;
}

} // namespace

void SospWorkspace::reserve(int requested) {
  if (requested <= capacity) {
    return;
  }
  const size_t n = static_cast<size_t>(requested);
  packed.assign(n, PACKED_INF);
  stamp.assign(n, 0);
  inFar.assign(n, 0);
  state.assign(n, 0);
  for (vector<int> *list : {&nearA, &nearB, &far, &far2, &candidates,
                            &frontier}) {
    list->clear();
    list->reserve(n);
  }
  capacity = requested;
  generation = 0;
}

int SospWorkspace::nextGeneration() {
  if (generation == INT_MAX) {
    fill(stamp.begin(), stamp.end(), 0);
    generation = 0;
  }
  return ++generation;
}

long long defaultDelta(long long numberOfEdges, int numberOfNodes,
                       long long weightSum) {
  if (numberOfEdges <= 0 || numberOfNodes <= 0) {
    return 1;
  }
  double averageWeight = static_cast<double>(weightSum) / numberOfEdges;
  double averageDegree = static_cast<double>(numberOfEdges) / numberOfNodes;
  return max(1LL, static_cast<long long>(32.0 * averageWeight / averageDegree));
}

bool sospUpdateCpu(const HostCsr &out, const HostCsr &in,
                   const HostChanges &changes, int source, long long delta,
                   long long maxWeight, SospWorkspace &ws,
                   long long *distances, int *parent, SospStats *stats) {
  const int n = out.numberOfNodes;
  SospStats local;
  SospStats &s = stats != nullptr ? *stats : local;
  s = SospStats();
  if (n == 0) {
    return true;
  }
  if (delta <= 0) {
    return false;
  }
  ws.reserve(n);
  Packing packing{};
  u64 bound = 0;
  if (!choosePacking(n, maxWeight, packing, bound)) {
    return false;
  }
  s.packedParents = packing.hasParents();
  u64 *packed = ws.packed.data();
  char *state = ws.state.data();
  int *stamp = ws.stamp.data();

  // ---- Pack the old tree. --------------------------------------------------
  bool overflow = false;
#pragma omp parallel for schedule(static) reduction(|| : overflow)
  for (int v = 0; v < n; ++v) {
    long long d = distances[v];
    state[v] = 0;
    if (d >= DISTANCE_INF / 2) {
      packed[v] = PACKED_INF;
    } else if (d < 0 || static_cast<u64>(d) > bound) {
      overflow = true;
      packed[v] = PACKED_INF;
    } else {
      packed[v] = packing.pack(static_cast<u64>(d), parent[v]);
    }
  }
  if (overflow) {
    cerr << "Error: an input distance exceeds (n - 1) * maxWeight.\n";
    return false;
  }

  // ---- Step 1: roots and subtree invalidation. ------------------------------
  ws.candidates.clear();
  const int generation = ws.nextGeneration();
  if (changes.numberOfChanged > 0) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < changes.numberOfChanged; ++i) {
      int v = changes.changedTo[i];
      if (parent[v] == changes.changedFrom[i]) {
        __atomic_store_n(&state[v], 2, __ATOMIC_RELAXED);
      }
    }
    auto loadState = [&](int v) {
      return __atomic_load_n(&state[v], __ATOMIC_RELAXED);
    };
    ListGather invalidGather(ws.candidates);
    // A walk longer than n steps means the input tree has a parent cycle
    // (e.g. a corrupt --init file): report it instead of looping forever.
    int cyclic = 0;
#pragma omp parallel
    {
      // 0 unknown, 1 valid, 2 invalid (a root among the vertex and its
      // ancestors). Concurrent walks over a shared path write the same
      // state, so relaxed atomics suffice.
#pragma omp for schedule(dynamic, 1024)
      for (int v = 0; v < n; ++v) {
        if (loadState(v) != 0 || __atomic_load_n(&cyclic, __ATOMIC_RELAXED)) {
          continue;
        }
        int u = v;
        int steps = 0;
        while (loadState(u) == 0 && parent[u] >= 0 && steps <= n) {
          u = parent[u];
          ++steps;
        }
        if (steps > n) {
          __atomic_store_n(&cyclic, 1, __ATOMIC_RELAXED);
          continue;
        }
        char result = loadState(u) == 0 ? 1 : loadState(u);
        for (u = v; u >= 0 && loadState(u) == 0; u = parent[u]) {
          __atomic_store_n(&state[u], result, __ATOMIC_RELAXED);
        }
      }
      vector<int> localInvalid;
#pragma omp for schedule(static)
      for (int v = 0; v < n; ++v) {
        if (state[v] == 2) {
          packed[v] = PACKED_INF;
          stamp[v] = generation;
          localInvalid.push_back(v);
        }
      }
      invalidGather.gather(localInvalid);
    }
    if (cyclic) {
      cerr << "Error: the input SOSP tree has a parent cycle.\n";
      return false;
    }
  }
  s.invalidated = static_cast<int>(ws.candidates.size());
  if (changes.numberOfInsertHeads > 0) {
    ListGather headGather(ws.candidates);
#pragma omp parallel
    {
      vector<int> localHeads;
#pragma omp for schedule(static)
      for (int i = 0; i < changes.numberOfInsertHeads; ++i) {
        int v = changes.insertHeads[i];
        if (v != source && claim(stamp, v, generation)) {
          localHeads.push_back(v);
        }
      }
      headGather.gather(localHeads);
    }
  }

  // ---- Step 1: pull pass. ---------------------------------------------------
  ws.frontier.clear();
  {
    const int pullGeneration = ws.nextGeneration();
    const int count = static_cast<int>(ws.candidates.size());
    ListGather frontierGather(ws.frontier);
#pragma omp parallel
    {
      vector<int> localFrontier;
#pragma omp for schedule(dynamic, 64)
      for (int i = 0; i < count; ++i) {
        int v = ws.candidates[i];
        u64 current = load(&packed[v]);
        u64 best = current;
        for (int e = in.rowPtr[v]; e < in.rowPtr[v + 1]; ++e) {
          int u = in.colInd[e];
          u64 word = load(&packed[u]);
          if (word != PACKED_INF) {
            best = min(best, packing.pack(packing.distance(word) +
                                              static_cast<u64>(in.weights[e]),
                                          u));
          }
        }
        if (best < current) {
          u64 old = atomicMin(&packed[v], best);
          if (packing.distance(best) < packing.distance(old) &&
              claim(stamp, v, pullGeneration)) {
            localFrontier.push_back(v);
          }
        }
      }
      frontierGather.gather(localFrontier);
    }
  }

  // ---- Step 2: near-far propagation. ----------------------------------------
  nearFar(out, source, static_cast<u64>(delta), packing, ws, s);
  unpack(out, source, packing, ws.packed, distances, parent);
  return true;
}

bool sospFromScratchCpu(const HostCsr &out, int source, long long delta,
                        long long maxWeight, SospWorkspace &ws,
                        long long *distances, int *parent, SospStats *stats) {
  const int n = out.numberOfNodes;
  SospStats local;
  SospStats &s = stats != nullptr ? *stats : local;
  s = SospStats();
  if (n == 0) {
    return true;
  }
  if (delta <= 0 || source < 0 || source >= n) {
    return false;
  }
  ws.reserve(n);
  Packing packing{};
  u64 bound = 0;
  if (!choosePacking(n, maxWeight, packing, bound)) {
    return false;
  }
  s.packedParents = packing.hasParents();
  u64 *packed = ws.packed.data();
#pragma omp parallel for schedule(static)
  for (int v = 0; v < n; ++v) {
    packed[v] = v == source ? packing.pack(0, -1) : PACKED_INF;
  }
  ws.frontier.assign(1, source);
  nearFar(out, source, static_cast<u64>(delta), packing, ws, s);
  unpack(out, source, packing, ws.packed, distances, parent);
  return true;
}
