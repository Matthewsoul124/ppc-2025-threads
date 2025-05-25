#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace khovansky_d_double_radix_batcher_all {

class RadixAll : public ppc::core::Task {
 public:
  explicit RadixAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, output_;
  boost::mpi::communicator world_;
};
}  // namespace khovansky_d_double_radix_batcher_all