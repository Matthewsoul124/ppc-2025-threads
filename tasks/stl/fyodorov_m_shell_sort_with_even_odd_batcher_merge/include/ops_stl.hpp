#pragma once

#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl {

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_, output_;

  void ShellSort();
  static void BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result);
  void LaunchTasksForGroups(int gap, int n, size_t num_groups, size_t threads_to_use, const std::vector<int>& groups);
  static void ProcessGroup(int gap, int n, std::vector<int>& input_ref, int group);
};
}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl