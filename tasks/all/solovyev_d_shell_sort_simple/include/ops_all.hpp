#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace solovyev_d_shell_sort_simple_all {

class TaskALL : public ppc::core::Task {
 public:
  explicit TaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  void ShellSort(std::vector<int>& data) const;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  boost::mpi::communicator world_;
  int num_threads_ = 0;
};

}  // namespace solovyev_d_shell_sort_simple_all