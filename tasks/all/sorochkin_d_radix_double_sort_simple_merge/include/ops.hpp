#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace sorochkin_d_radix_double_sort_simple_merge_all {

class SortTask : public ppc::core::Task {
 public:
  explicit SortTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  void Squash(boost::mpi::communicator& group);
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  std::vector<double> procchunk_;
  boost::mpi::communicator world_;
};

}  // namespace sorochkin_d_radix_double_sort_simple_merge_all