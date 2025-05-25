#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace komshina_d_image_filtering_vertical_gaussian_all {

class TestTaskALL : public ppc::core::Task {
 public:
  explicit TestTaskALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<unsigned char> DistributeInputToProcesses(int rank, int size, std::size_t k_radius,
                                                        std::size_t local_height, std::size_t offset,
                                                        std::size_t halo_top, std::size_t halo_bottom);
  std::vector<unsigned char> input_, output_;
  std::vector<float> kernel_;
  std::size_t height_{}, width_{};
  boost::mpi::communicator world_;
};

}  // namespace komshina_d_image_filtering_vertical_gaussian_all