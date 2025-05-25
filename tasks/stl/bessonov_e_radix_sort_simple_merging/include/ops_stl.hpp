#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace bessonov_e_radix_sort_simple_merging_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> output_;

  static void ConvertDoubleToBits(const std::vector<double>& input, std::vector<uint64_t>& bits, size_t start,
                                  size_t end);
  static void ConvertBitsToDouble(const std::vector<uint64_t>& bits, std::vector<double>& output, size_t start,
                                  size_t end);
  static void RadixSortPass(std::vector<uint64_t>& bits, std::vector<uint64_t>& temp, int shift);
};

}  // namespace bessonov_e_radix_sort_simple_merging_stl