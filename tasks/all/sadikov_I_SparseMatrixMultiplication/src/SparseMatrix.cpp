#include "all/sadikov_I_SparseMatrixMultiplication/include/SparseMatrix.hpp"

#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <optional>
#include <vector>

#include "core/util/include/util.hpp"
#include "oneapi/tbb/parallel_for.h"

namespace sadikov_i_sparse_matrix_multiplication_task_all {

void SparseMatrix::SetMatrixData(const MatrixComponents& components, int rows_count, int columns_count) {
  m_compontents_ = components;
  m_rowsCount_ = rows_count;
  m_columnsCount_ = columns_count;
}
SparseMatrix SparseMatrix::Transpose(const SparseMatrix& matrix) {
  MatrixComponents components;
  auto max_size = std::max(matrix.GetRowsCount(), matrix.GetColumnsCount());
  std::vector<std::vector<double>> intermediate_values(max_size);
  std::vector<std::vector<int>> intermediate_indexes(max_size);
  int counter = 0;
  for (size_t i = 0; i < matrix.GetElementsSum().size(); ++i) {
    auto limit = i == 0 ? matrix.GetElementsSum()[0] : matrix.GetElementsSum()[i] - matrix.GetElementsSum()[i - 1];
    for (int j = 0; j < limit; ++j) {
      intermediate_values[matrix.GetRows()[counter]].emplace_back(matrix.GetValues()[counter]);
      intermediate_indexes[matrix.GetRows()[counter]].emplace_back(i);
      counter++;
    }
  }
  for (size_t i = 0; i < intermediate_values.size(); ++i) {
    for (size_t j = 0; j < intermediate_values[i].size(); ++j) {
      components.m_values.emplace_back(intermediate_values[i][j]);
      components.m_rows.emplace_back(intermediate_indexes[i][j]);
    }
    i == 0 ? components.m_elementsSum.emplace_back(intermediate_values[i].size())
           : components.m_elementsSum.emplace_back(intermediate_values[i].size() + components.m_elementsSum[i - 1]);
  }
  return {matrix.GetColumnsCount(), matrix.GetRowsCount(), components};
}
double SparseMatrix::CalculateSum(const SparseMatrix& fmatrix, const SparseMatrix& smatrix,
                                  const std::vector<int>& felements_sum, const std::vector<int>& selements_sum,
                                  int i_index, int j_index) {
  int fmatrix_elements_count = GetElementsCount(j_index, felements_sum);
  int smatrix_elements_count = GetElementsCount(i_index, selements_sum);
  int fmatrix_start_index = j_index != 0 ? felements_sum[j_index] - fmatrix_elements_count : 0;
  int smatrix_start_index = i_index != 0 ? selements_sum[i_index] - smatrix_elements_count : 0;
  double sum = 0.0;
  for (int i = 0; i < fmatrix_elements_count; i++) {
    for (int j = 0; j < smatrix_elements_count; j++) {
      if (fmatrix.GetRows()[fmatrix_start_index + i] == smatrix.GetRows()[smatrix_start_index + j]) {
        sum += fmatrix.GetValues()[i + fmatrix_start_index] * smatrix.GetValues()[j + smatrix_start_index];
      }
    }
  }
  return sum;
}

MatrixComponents SparseMatrix::Multiplicate(const SparseMatrix& first_matrix, const SparseMatrix& second_matrix,
                                            int second_displacement, int barier) {
  MatrixComponents component;
  component.Resize(first_matrix.GetElementsSum().size() * second_matrix.GetElementsSum().size(),
                   second_matrix.GetElementsSum().size());
  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  arena.execute([&] {
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(second_displacement, barier,
                                           (barier - second_displacement) / ppc::util::GetPPCNumThreads()),
        [&](const oneapi::tbb::blocked_range<size_t>& range) {
          for (size_t i = range.begin(); i != range.end(); ++i) {
            if (component.m_elementsSum[i] == 0) {
              component.m_elementsSum[i]++;
            }
            for (size_t j = 0; j < first_matrix.GetElementsSum().size(); ++j) {
              double sum = CalculateSum(first_matrix, second_matrix, first_matrix.GetElementsSum(),
                                        second_matrix.GetElementsSum(), static_cast<int>(i), static_cast<int>(j));
              if (sum != 0) {
                component.m_values[(i * first_matrix.GetElementsSum().size()) + j] = sum;
                component.m_rows[(i * first_matrix.GetElementsSum().size()) + j] = static_cast<int>(j) + 1;
                component.m_elementsSum[i]++;
              }
            }
          }
        });
  });
  return {component};
}

SparseMatrix SparseMatrix::MatrixToSparse(int rows_count, int columns_count, const std::vector<double>& values) {
  MatrixComponents compontents;
  compontents.m_elementsSum.resize(columns_count);
  for (int i = 0; i < columns_count; ++i) {
    for (int j = 0; j < rows_count; ++j) {
      if (values[i + (columns_count * j)] != 0) {
        compontents.m_values.emplace_back(values[i + (columns_count * j)]);
        compontents.m_rows.emplace_back(j);
        compontents.m_elementsSum[i]++;
      }
    }
    if (i != columns_count - 1) {
      compontents.m_elementsSum[i + 1] = compontents.m_elementsSum[i];
    }
  }
  return {rows_count, columns_count, compontents};
}

std::vector<double> FromSparseMatrix(const SparseMatrix& matrix) {
  std::vector<double> simple_matrix(matrix.GetRowsCount() * matrix.GetColumnsCount());
  int counter = 0;
  for (size_t i = 0; i < matrix.GetElementsSum().size(); ++i) {
    auto limit = i == 0 ? matrix.GetElementsSum()[0] : matrix.GetElementsSum()[i] - matrix.GetElementsSum()[i - 1];
    for (int j = 0; j < limit; ++j) {
      simple_matrix[i + (matrix.GetColumnsCount() * matrix.GetRows()[counter])] = matrix.GetValues()[counter];
      counter++;
    }
  }
  return simple_matrix;
}

int SparseMatrix::GetElementsCount(int index, const std::vector<int>& elements_sum) {
  if (index == 0) {
    return elements_sum[index];
  }
  return elements_sum[index] - elements_sum[index - 1];
}

std::vector<double> BaseMatrixMultiplication(const std::vector<double>& fmatrix, int fmatrix_rows_count,
                                             int fmatrix_columns_count, const std::vector<double>& smatrix,
                                             int smatrix_rows_count, int smatrix_columns_count) {
  if (fmatrix_rows_count == smatrix_columns_count) {
    std::vector<double> answer(fmatrix_rows_count * smatrix_columns_count);
    for (int i = 0; i < fmatrix_rows_count; i++) {
      for (int j = 0; j < smatrix_columns_count; j++) {
        for (int n = 0; n < smatrix_rows_count; n++) {
          answer[j + (i * smatrix_columns_count)] +=
              fmatrix[(i * fmatrix_columns_count) + n] * smatrix[(n * smatrix_columns_count) + j];
        }
      }
    }
    return answer;
  }
  return {};
}
}  // namespace sadikov_i_sparse_matrix_multiplication_task_all