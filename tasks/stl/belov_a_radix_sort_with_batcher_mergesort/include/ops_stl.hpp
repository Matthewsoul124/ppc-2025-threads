#ifndef OPS_STL_HPP
#define OPS_STL_HPP

#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

using Bigint = long long;

namespace belov_a_radix_batcher_mergesort_stl {

class RadixBatcherMergesortParallel : public ppc::core::Task {
 public:
  explicit RadixBatcherMergesortParallel(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void Sort(std::span<Bigint> arr);

 private:
  std::vector<Bigint> array_;  // input unsorted numbers array
  size_t n_ = 0;               // array size
  size_t num_threads_ = 1;     // number of threads for parallel execution

  static void RadixSort(std::vector<Bigint>& arr, bool invert);
  static void CountingSort(std::vector<Bigint>& arr, Bigint digit_place);
  static int GetNumberDigitCapacity(Bigint num);

  void SortParallel(std::vector<Bigint>& arr) const;
  void BatcherMergeParallel(std::vector<Bigint>& arr) const;
};

}  // namespace belov_a_radix_batcher_mergesort_stl

#endif  // OPS_STL_HPP