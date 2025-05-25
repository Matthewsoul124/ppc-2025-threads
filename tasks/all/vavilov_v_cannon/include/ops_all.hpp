#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace vavilov_v_cannon_all {
class CannonALL : public ppc::core::Task {
 public:
  explicit CannonALL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  int N_;
  int block_size_;
  int num_blocks_;
  std::vector<double> A_;
  std::vector<double> B_;
  std::vector<double> C_;
  boost::mpi::communicator world_;

  void InitialShift(std::vector<double>& local_a, std::vector<double>& local_b);
  void BlockMultiply(const std::vector<double>& local_a, const std::vector<double>& local_b,
                     std::vector<double>& local_c);
  void ShiftBlocks(std::vector<double>& local_a, std::vector<double>& local_b);
  static int FindOptimalGridSize(int size, int n);
  static void TakeBlock(const std::vector<double>& matrix, double* block, int n, int k, int block_row, int block_col);
  void GatherResults(std::vector<double>& tmp_c, int block_size_sq);
  void PrepareScatterData(std::vector<double>& scatter_a, std::vector<double>& scatter_b, int active_procs,
                          int block_size_sq);
};
}  // namespace vavilov_v_cannon_all
