#pragma once

#include <complex>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "mpi.h"

struct Matrix {
  uint32_t rows;
  uint32_t cols;
  std::vector<std::complex<double>> data;

  std::complex<double>& Get(uint32_t row, uint32_t col) { return data[(row * cols) + col]; }

  bool operator==(const Matrix& other) const noexcept {
    return rows == other.rows && cols == other.cols && data == other.data;
  }
};

inline Matrix MultiplyMat(Matrix& lhs, Matrix& rhs) {
  Matrix res{.rows = lhs.rows, .cols = rhs.cols, .data = std::vector<std::complex<double>>(lhs.rows * rhs.cols)};
  for (uint32_t i = 0; i < lhs.rows; i++) {
    for (uint32_t j = 0; j < rhs.cols; j++) {
      res.Get(i, j) = 0;
      for (uint32_t k = 0; k < rhs.rows; k++) {
        res.Get(i, j) += lhs.Get(i, k) * rhs.Get(k, j);
      }
    }
  }
  return res;
}

struct MatrixCRS {
  std::vector<std::complex<double>> data;

  std::uint32_t cols_count;
  std::vector<std::uint32_t> rowptr;
  std::vector<std::uint32_t> colind;

  //

  [[nodiscard]] std::uint32_t GetRows() const { return rowptr.size() - 1; }
  [[nodiscard]] std::uint32_t GetCols() const { return cols_count; }

  using TSubrange = std::pair<int, int>;

  void Bcast(MPI_Comm comm, int root) {
    int sizes[3] = {int(data.size()), int(rowptr.size()), int(colind.size())};
    MPI_Bcast(sizes, std::size(sizes), MPI_INT, root, comm);

    data.resize(sizes[0]);
    MPI_Bcast(data.data(), static_cast<int>(data.size() * sizeof(decltype(data)::value_type)), MPI_CHAR, root, comm);

    MPI_Bcast(&cols_count, 1, MPI_UINT32_T, root, comm);

    rowptr.resize(sizes[1]);
    MPI_Bcast(rowptr.data(), static_cast<int>(rowptr.size()), MPI_UNSIGNED, root, comm);

    colind.resize(sizes[2]);
    MPI_Bcast(colind.data(), static_cast<int>(colind.size()), MPI_UNSIGNED, root, comm);
  }

  void Send(MPI_Comm comm, int dest) const {
    int sizes[3] = {int(data.size()), int(rowptr.size()), int(colind.size())};
    MPI_Send(sizes, std::size(sizes), MPI_INT, dest, 0, comm);

    MPI_Send(data.data(), static_cast<int>(data.size() * sizeof(decltype(data)::value_type)), MPI_CHAR, dest, 0, comm);
    MPI_Send(&cols_count, 1, MPI_UINT32_T, dest, 0, comm);
    MPI_Send(rowptr.data(), static_cast<int>(rowptr.size()), MPI_UNSIGNED, dest, 0, comm);
    MPI_Send(colind.data(), static_cast<int>(colind.size()), MPI_UNSIGNED, dest, 0, comm);
  }

  [[nodiscard]] MatrixCRS ExtractPart(const TSubrange& row_range) const {
    MatrixCRS res;
    const TSubrange idx_range{rowptr[row_range.first], rowptr[row_range.second]};

    res.data.assign(data.data() + idx_range.first, data.data() + idx_range.second);
    res.cols_count = cols_count;
    res.rowptr.assign(rowptr.data() + row_range.first, rowptr.data() + (row_range.second - row_range.first + 1));
    res.colind.assign(colind.data() + idx_range.first, colind.data() + (idx_range.second - idx_range.first));

    return res;
  }

  void GetOffsets(const TSubrange& row_range, int (&offsets_buf)[2]) const {
    offsets_buf[0] = row_range.first;
    offsets_buf[1] = static_cast<int>(rowptr[row_range.first]);
  }

  void SendPart(MPI_Comm comm, int dest, const TSubrange& row_range) const {
    const TSubrange idx_range{rowptr[row_range.first], rowptr[row_range.second]};

    int sizes[2] = {idx_range.second - idx_range.first, row_range.second - row_range.first + 1};
    MPI_Send(sizes, std::size(sizes), MPI_INT, dest, 0, comm);

    MPI_Send(data.data() + idx_range.first, sizes[0] * static_cast<int>(sizeof(decltype(data)::value_type)), MPI_CHAR,
             dest, 0, comm);
    MPI_Send(&cols_count, 1, MPI_UINT32_T, dest, 0, comm);
    MPI_Send(rowptr.data() + row_range.first, sizes[1], MPI_UNSIGNED, dest, 0, comm);
    MPI_Send(colind.data() + idx_range.first, sizes[0], MPI_UNSIGNED, dest, 0, comm);
  }

  static MatrixCRS Recv(MPI_Comm comm, int source) {
    MatrixCRS res{};

    int sizes[2]{};
    MPI_Recv(sizes, std::size(sizes), MPI_INT, source, 0, comm, MPI_STATUS_IGNORE);

    res.data.resize(sizes[0]);
    MPI_Recv(res.data.data(), sizes[0] * static_cast<int>(sizeof(decltype(data)::value_type)), MPI_CHAR, source, 0,
             comm, MPI_STATUS_IGNORE);

    MPI_Recv(&res.cols_count, 1, MPI_UINT32_T, source, 0, comm, MPI_STATUS_IGNORE);

    res.rowptr.resize(sizes[1]);
    MPI_Recv(res.rowptr.data(), sizes[1], MPI_UNSIGNED, source, 0, comm, MPI_STATUS_IGNORE);

    res.colind.resize(sizes[0]);
    MPI_Recv(res.colind.data(), sizes[0], MPI_UNSIGNED, source, 0, comm, MPI_STATUS_IGNORE);

    return res;
  }

  bool operator==(const MatrixCRS& other) const noexcept {
    return cols_count == other.cols_count && rowptr == other.rowptr && colind == other.colind && data == other.data;
  }
};

inline MatrixCRS RegularToCRS(const Matrix& matrix) {
  MatrixCRS result;
  result.rowptr.resize(matrix.rows + 1);
  result.cols_count = matrix.cols;

  uint32_t i = 0;
  for (uint32_t row = 0; row < matrix.rows; ++row) {
    uint32_t nz = 0;
    for (uint32_t col = 0; col < matrix.cols; ++col) {
      if (const auto& element = matrix.data[i++]; element != 0.0) {
        ++nz;
        result.colind.push_back(col);
        result.data.push_back(element);
      }
    }
    result.rowptr[row + 1] = result.rowptr[row] + nz;
  }

  return result;
}

inline Matrix CRSToRegular(const MatrixCRS& crs) {
  Matrix matrix{.rows = crs.GetRows(),
                .cols = crs.GetCols(),
                .data = std::vector<std::complex<double>>(crs.GetRows() * crs.GetCols())};
  for (uint32_t row = 0; row < matrix.rows; ++row) {
    for (uint32_t i = crs.rowptr[row]; i < crs.rowptr[row + 1]; ++i) {
      matrix.Get(row, crs.colind[i]) = crs.data[i];
    }
  }
  return matrix;
}

namespace tyurin_m_matmul_crs_complex_all {

class TestTaskAll : public ppc::core::Task {
 public:
  explicit TestTaskAll(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  }
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  MatrixCRS global_lhs_;
  MatrixCRS rhs_;

  std::vector<MatrixCRS> procres_;

  int rank_;
  MatrixCRS Scatter(int& row_offset, int& idx_offset);
  std::vector<MatrixCRS> Gather(MatrixCRS&& local_res) const;
};

}  // namespace tyurin_m_matmul_crs_complex_all