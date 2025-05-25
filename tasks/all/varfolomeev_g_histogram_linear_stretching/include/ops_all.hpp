#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace varfolomeev_g_histogram_linear_stretching_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<uint8_t> input_image_, result_image_;
  const boost::mpi::communicator world_;

  void ScatterData(std::vector<uint8_t>& local_data);
  void FindMinMax(const std::vector<uint8_t>& local_data, int& global_min, int& global_max);
  void GatherResults(const std::vector<uint8_t>& local_data);
};

}  // namespace varfolomeev_g_histogram_linear_stretching_all