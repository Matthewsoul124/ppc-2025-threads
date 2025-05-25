#pragma once

#include <utility>
#include <vector>

#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

namespace moiseev_a_mult_mat_mpi {

class MultMatMPI : public ppc::core::Task {
 public:
  explicit MultMatMPI(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void DetermineGrid(int world_size, int& p, int& active_procs, int& block) const;

  void DistributeMatrices(boost::mpi::communicator& world, int world_rank, int active_procs, int p, int block,
                          std::vector<double>& a_block, std::vector<double>& b_block) const;

  static void ComputeLocalBlock(boost::mpi::communicator& active, int my_row, int my_col, int p, int block,
                                std::vector<double>& a_block, std::vector<double>& b_block,
                                std::vector<double>& c_block);

  void GatherResult(boost::mpi::communicator& world, int world_rank, int active_procs, int p, int block,
                    const std::vector<double>& c_block);

  std::vector<double> matrix_a_, matrix_b_, matrix_c_;
  int matrix_size_{};
  int num_blocks_{};
  int block_size_{};
};

}  // namespace moiseev_a_mult_mat_mpi