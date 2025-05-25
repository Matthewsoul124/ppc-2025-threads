#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace titov_s_image_filter_horiz_gaussian3x3_all {

class GaussianFilterALL : public ppc::core::Task {
 public:
  explicit GaussianFilterALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool DistributeData(int world_rank, int world_size, int height, int width, int start_row, int end_row,
                      std::vector<double> &local_input);
  void ProcessRows(const std::vector<double> &local_input, std::vector<double> &local_output, int width, int local_rows,
                   int num_threads);
  bool CollectResults(int world_rank, int world_size, int height, int width, int start_row,
                      const std::vector<double> &local_output);

 private:
  std::vector<double> input_;
  std::vector<double> output_;
  int width_;
  int height_;
  std::vector<int> kernel_;
  boost::mpi::communicator world_;
};

}  // namespace titov_s_image_filter_horiz_gaussian3x3_all