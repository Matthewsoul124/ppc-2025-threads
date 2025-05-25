#pragma once

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_radix_sort_simple_merging_omp {

class TestTaskOMP : public ppc::core::Task {
 public:
  explicit TestTaskOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  static void ConvertDoubleToBits(std::vector<double>& input, std::vector<uint64_t>& bits);
  static void CountDigits(const std::vector<uint64_t>& bits, int shift, std::vector<size_t>& count);
  static void ComputeOffsets(const std::vector<std::vector<size_t>>& thread_counts,
                             std::vector<std::vector<size_t>>& thread_offsets, std::vector<size_t>& count,
                             int num_threads, int radix);
  static void DistributeElements(std::vector<uint64_t>& bits, std::vector<uint64_t>& temp,
                                 std::vector<std::vector<size_t>>& thread_offsets,
                                 const std::vector<std::vector<uint64_t>>& thread_elements,
                                 const std::vector<std::vector<int>>& thread_digits);
  static void ConvertBitsToDouble(std::vector<uint64_t>& bits, std::vector<double>& output);
};

}  // namespace bessonov_e_radix_sort_simple_merging_omp