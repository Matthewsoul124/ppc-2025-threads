#pragma once

#include <cstddef>
#include <optional>
#include <utility>
#include <vector>

namespace sadikov_i_sparse_matrix_multiplication_task_all {
struct MatrixComponents {
  std::vector<double> m_values;
  std::vector<int> m_rows;
  std::vector<int> m_elementsSum;
  void Resize(size_t values_size, std::optional<size_t> sums_size) noexcept {
    m_values.resize(values_size);
    m_rows.resize(values_size);
    if (sums_size.has_value()) {
      m_elementsSum.resize(sums_size.value());
    }
  }
};
class SparseMatrix {
  int m_rowsCount_ = 0;
  int m_columnsCount_ = 0;
  MatrixComponents m_compontents_;
  static int GetElementsCount(int index, const std::vector<int>& elements_sum);
  static double CalculateSum(const SparseMatrix& fmatrix, const SparseMatrix& smatrix,
                             const std::vector<int>& felements_sum, const std::vector<int>& selements_sum, int i_index,
                             int j_index);

 public:
  SparseMatrix() = default;
  SparseMatrix(int rows_count, int columns_count, MatrixComponents components) noexcept
      : m_rowsCount_(rows_count), m_columnsCount_(columns_count), m_compontents_(std::move(components)) {};
  [[nodiscard]] const MatrixComponents& GetMatrixComponents() const noexcept { return m_compontents_; }
  void SetMatrixData(const MatrixComponents& components, int rows_count, int columns_count);
  [[nodiscard]] const std::vector<double>& GetValues() const noexcept { return m_compontents_.m_values; }
  [[nodiscard]] const std::vector<int>& GetRows() const noexcept { return m_compontents_.m_rows; }
  [[nodiscard]] const std::vector<int>& GetElementsSum() const noexcept { return m_compontents_.m_elementsSum; }
  [[nodiscard]] int GetColumnsCount() const noexcept { return m_columnsCount_; }
  [[nodiscard]] int GetRowsCount() const noexcept { return m_rowsCount_; }
  static SparseMatrix MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values);
  static SparseMatrix Transpose(const SparseMatrix& matrix);
  static MatrixComponents Multiplicate(const SparseMatrix& first_matrix, const SparseMatrix& second_matrix,
                                       int second_displacement, int barier);
  template <typename Archive>
  // NOLINTNEXTLINE(readability-identifier-naming)
  void serialize(Archive& ar, const unsigned int) {
    ar & m_compontents_.m_values;
    ar & m_compontents_.m_rows;
    ar & m_compontents_.m_elementsSum;
    ar & m_rowsCount_;
    ar & m_columnsCount_;
  }
};

std::vector<double> FromSparseMatrix(const SparseMatrix& matrix);

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count);
}  // namespace sadikov_i_sparse_matrix_multiplication_task_all