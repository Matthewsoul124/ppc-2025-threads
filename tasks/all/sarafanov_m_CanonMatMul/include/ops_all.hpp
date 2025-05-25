#pragma once

#include <boost/mpi/communicator.hpp>
#include <utility>
#include <vector>

#include "all/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"
#include "core/task/include/task.hpp"

namespace sarafanov_m_canon_mat_mul_all {
struct IndexesPair {
  int first = 0;
  int second = 0;
  IndexesPair() = default;
  IndexesPair(int first, int second);
  template <typename Archive>
  // NOLINTNEXTLINE(readability-identifier-naming)
  void serialize(Archive& archive, const unsigned int) {
    archive & first;
    archive & second;
  }
};
class CanonMatMulALL : public ppc::core::Task {
  CanonMatrix a_matrix_;
  CanonMatrix b_matrix_;
  CanonMatrix c_matrix_;
  CanonMatrix matrix_sum_on_process_;
  boost::mpi::communicator world_;
  std::vector<IndexesPair> indexes_;

 public:
  explicit CanonMatMulALL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  bool CheckSquareSize(int number);
  void CalculateIndexes();
  static std::vector<double> ConvertToSquareMatrix(int need_size, MatrixType type, const std::vector<double>& matrx);
};
}  // namespace sarafanov_m_canon_mat_mul_all