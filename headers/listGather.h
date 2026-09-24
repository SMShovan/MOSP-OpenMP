#ifndef LIST_GATHER_H
#define LIST_GATHER_H

#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

/**
 * @brief Concatenate per-thread lists into a shared vector.
 *
 * @details
 * Every thread of a parallel region calls gather() exactly once with its
 * local list; the lists are appended to the output (after its current
 * contents) in thread order, at offsets from a prefix sum over their
 * sizes, so the copies run in parallel instead of being serialized in a
 * critical section. gather() contains barriers.
 */
class ListGather {
public:
  explicit ListGather(std::vector<int> &out)
      : out_(out), base_(out.size()),
        offsets_(static_cast<size_t>(omp_get_max_threads()) + 1, 0) {}

  void gather(const std::vector<int> &local) {
    const int thread = omp_get_thread_num();
    offsets_[thread + 1] = local.size();
#pragma omp barrier
#pragma omp single
    {
      const int threads = omp_get_num_threads();
      for (int i = 0; i < threads; ++i) {
        offsets_[i + 1] += offsets_[i];
      }
      out_.resize(base_ + offsets_[threads]);
    }
    std::copy(local.begin(), local.end(),
              out_.begin() + base_ + offsets_[thread]);
  }

private:
  std::vector<int> &out_;
  size_t base_;
  std::vector<size_t> offsets_;
};

#endif // LIST_GATHER_H
