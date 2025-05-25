#pragma once

#include <utility>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace borisov_s_strassen_all {

class ParallelStrassenMpiStl : public ppc::core::Task {
 public:
  explicit ParallelStrassenMpiStl(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_;
  std::vector<double> a_pad_, b_pad_;
  std::vector<double> output_;

  int rowsA_ = 0;
  int colsA_ = 0;
  int rowsB_ = 0;
  int colsB_ = 0;
  int m_ = 0;

  boost::mpi::communicator world_;
};

}  // namespace borisov_s_strassen_all