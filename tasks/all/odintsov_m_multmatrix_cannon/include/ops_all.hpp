#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace odintsov_m_mulmatrix_cannon_all {

class MulMatrixCannonALL : public ppc::core::Task {
 public:
  explicit MulMatrixCannonALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  static void ShiftRow(std::vector<double>& matrix, int root, int row, int shift);
  static void ShiftColumn(std::vector<double>& matrix, int root, int col, int shift);
  void ShiftBlocksUp(std::vector<double>& matrix, int root, int block_sz) const;
  void ShiftBlocksLeft(std::vector<double>& matrix, int root, int block_sz) const;
  static bool IsSquere(unsigned int num);
  static int GetBlockSize(int n);
  static void InitializeShift(std::vector<double>& matrix, int root, int grid_size, int block_sz, bool is_row_shift);
  static void ProcessBlockMul(int bi, int bj_start, int bj_end, int root, int block_sz,
                              const std::vector<double>& matrix_a, const std::vector<double>& matrix_b,
                              std::vector<double>& local_c);
  static void ProcessBlockSTL(int bi, int num_blocks, int root, int block_sz, const std::vector<double>& matrix_a,
                              const std::vector<double>& matrix_b, std::vector<double>& local_c);

  std::vector<double> matrixA_, matrixB_;
  unsigned int szA_ = 0, szB_ = 0;
  int block_sz_ = 0;
  std::vector<double> matrixC_;
  boost::mpi::communicator com_;
};

}  // namespace odintsov_m_mulmatrix_cannon_all
