#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace opolin_d_radix_batcher_sort_stl {
void ParallelProcessRange(size_t total_size, unsigned int num_threads, const std::function<void(size_t, size_t)>& func);
uint32_t IntToUnsigned(int value);
int UnsignedToInt(uint32_t value);
void ParallelRunTasks(const std::vector<std::function<void()>>& tasks);
void RadixSortLSD(std::vector<uint32_t>::iterator begin, std::vector<uint32_t>::iterator end);
void IterativeOddEvenBlockMerge(std::vector<uint32_t>::iterator data_begin, std::vector<uint32_t>::iterator data_end,
                                size_t num_initial_blocks, size_t initial_block_size, unsigned int num_threads);
class RadixBatcherSortTaskStl : public ppc::core::Task {
 public:
  explicit RadixBatcherSortTaskStl(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<uint32_t> unsigned_data_;
  int size_;
};
}  // namespace opolin_d_radix_batcher_sort_stl