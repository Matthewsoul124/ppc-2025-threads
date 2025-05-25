#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace gusev_n_sorting_int_simple_merging_all {

class SortingIntSimpleMergingALL : public ppc::core::Task {
 public:
  explicit SortingIntSimpleMergingALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void CountingSort(std::vector<int>& arr, int exp);
  static void RadixSortForNonNegative(std::vector<int>& arr);
  static void RadixSort(std::vector<int>& arr);

  static void SplitBySign(const std::vector<int>& arr, std::vector<int>& negatives, std::vector<int>& positives);
  static void MergeResults(std::vector<int>& arr, const std::vector<int>& negatives, const std::vector<int>& positives);
  static std::vector<std::vector<int>> DistributeArray(const std::vector<int>& arr, int num_procs);
  static std::vector<int> MergeSortedArrays(const std::vector<std::vector<int>>& arrays);

  // Updated from RadixSort
  static void ProcessNegativeNumbers(std::vector<int>& negatives, std::vector<int>& sorted_negatives, int rank,
                                     int size);
  static void ProcessPositiveNumbers(std::vector<int>& positives, std::vector<int>& sorted_positives, int rank,
                                     int size);
  static void DistributeAndSortChunk(std::vector<int>& chunk, std::vector<std::vector<int>>& chunks, int rank, int size,
                                     int tag);
  static std::vector<int> GatherSortedChunks(const std::vector<int>& my_chunk, int rank, int size, int tag);

  std::vector<int> input_, output_;
};

}  // namespace gusev_n_sorting_int_simple_merging_all
