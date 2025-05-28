#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi {
class TestTaskMPI : public ppc::core::Task {
 public:
  explicit TestTaskMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  const std::vector<int>& get_internal_output() const { return output_; }

 private:
  static void ShellSort(std::vector<int>& arr);
  static void BatcherMerge(std::vector<int>& left, std::vector<int>& right, std::vector<int>& result);
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<int> local_input_;
  boost::mpi::communicator world_;
};
}  // namespace fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi