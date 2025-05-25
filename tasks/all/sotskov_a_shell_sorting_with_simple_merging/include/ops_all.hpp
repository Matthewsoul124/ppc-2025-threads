#pragma once

#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_all {

void ShellSort(std::vector<int>& arr, size_t left, size_t right);
void ParallelMerge(std::vector<int>& arr, size_t left, size_t mid, size_t right);
void ShellSortWithSimpleMerging(std::vector<int>& arr);

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  boost::mpi::communicator world_;
  int rank_ = world_.rank();
  int size_ = world_.size();
  void CalculateDistribution(int total, std::vector<int>& counts, std::vector<int>& displs) const;
};

}  // namespace sotskov_a_shell_sorting_with_simple_merging_all