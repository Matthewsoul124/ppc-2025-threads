#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace kalyakina_a_shell_with_simple_merge_all {

class ShellSortALL : public ppc::core::Task {
  static std::vector<unsigned int> CalculationOfGapLengths(unsigned int size);
  void ShellSort(std::vector<int>& vec);

 public:
  static std::vector<int> SimpleMergeSort(const std::vector<int>& vec1, const std::vector<int>& vec2);

  explicit ShellSortALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<int> input_;
  std::vector<int> output_;
  std::vector<unsigned int> Sedgwick_sequence_;
  boost::mpi::communicator world_;
};

}  // namespace kalyakina_a_shell_with_simple_merge_all