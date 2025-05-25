#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace deryabin_m_hoare_sort_simple_merge_stl {

void HoareSort(std::vector<double>& a, size_t first, size_t last);

class HoareSortTaskSequential : public ppc::core::Task {
 public:
  explicit HoareSortTaskSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_array_A_;
  size_t dimension_;
  size_t min_chunk_size_;
  size_t chunk_count_;
};
class HoareSortTaskSTL : public ppc::core::Task {
 public:
  explicit HoareSortTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_array_A_;
  size_t dimension_;
  size_t min_chunk_size_;
  size_t chunk_count_;
};
}  // namespace deryabin_m_hoare_sort_simple_merge_stl
