#include <complex>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace yasakova_t_sparse_matrix_multiplication_stl {

struct CompressedRowStorage {
  std::vector<std::complex<double>> nonZeroValues;
  std::vector<int> columnIndices;
  std::vector<int> rowPointers;
  int rowCount;
  int columnCount;
  CompressedRowStorage() : nonZeroValues({}), columnIndices({}), rowPointers({}), rowCount(0), columnCount(0) {};
  CompressedRowStorage(int rows, int cols) : rowCount(rows), columnCount(cols) { rowPointers.resize(rows + 1, 0); }

  void InsertElement(int row_idx, std::complex<double> val, int col_idx);
  CompressedRowStorage(const CompressedRowStorage& other) = default;
  CompressedRowStorage& operator=(const CompressedRowStorage& other) = default;
  static void DisplayMatrix(const CompressedRowStorage& matrix);
};
std::vector<std::complex<double>> ConvertToDense(const CompressedRowStorage& mat);
CompressedRowStorage ConvertToSparse(std::vector<std::complex<double>>& vec);
bool CompareMatrices(const CompressedRowStorage& left_matrix, const CompressedRowStorage& right_matrix);
bool AreClose(const std::complex<double>& left_matrix, const std::complex<double>& right_matrix, double epsilon);
class SparseMatrixMultiTask : public ppc::core::Task {
 public:
  explicit SparseMatrixMultiTask(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<std::complex<double>> inputData_, resultData_;
  CompressedRowStorage firstMatrix_, secondMatrix_;
};

}  // namespace yasakova_t_sparse_matrix_multiplication_stl