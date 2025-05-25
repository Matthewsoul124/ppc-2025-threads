#pragma once

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <utility>
#include <vector>

#include "all/sadikov_I_SparseMatrixMultiplication/include/SparseMatrix.hpp"
#include "core/task/include/task.hpp"

namespace sadikov_i_sparse_matrix_multiplication_task_all {
class CCSMatrixALL : public ppc::core::Task {
  SparseMatrix m_fMatrix_;
  SparseMatrix m_sMatrix_;
  SparseMatrix m_answerMatrix_;
  boost::mpi::communicator m_world_;
  std::vector<int> m_displacements_;
  MatrixComponents m_intermediate_data_;
  std::pair<std::vector<int>, std::vector<int>> m_sizes_;
  void CalculateDisplacements();

 public:
  explicit CCSMatrixALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};
}  // namespace sadikov_i_sparse_matrix_multiplication_task_all