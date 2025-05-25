#include "all/sarafanov_m_CanonMatMul/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <utility>
#include <vector>

#include "all/sarafanov_m_CanonMatMul/include/CanonMatrix.hpp"
#include "core/util/include/util.hpp"

sarafanov_m_canon_mat_mul_all::IndexesPair::IndexesPair(int first, int second) : first(first), second(second) {}

void sarafanov_m_canon_mat_mul_all::CanonMatMulALL::CalculateIndexes() {
  int part = static_cast<int>(a_matrix_.GetSqrtSize()) / world_.size();
  int balance = static_cast<int>(a_matrix_.GetSqrtSize()) % world_.size();
  for (int i = 0; i < world_.size(); ++i) {
    if (i == 0) {
      indexes_.emplace_back(0, part);
    } else {
      indexes_.emplace_back(indexes_[i - 1].second, indexes_[i - 1].second + part);
    }
    if (balance != 0) {
      indexes_[i].second++;
      balance--;
    }
  }
}

bool sarafanov_m_canon_mat_mul_all::CanonMatMulALL::PreProcessingImpl() {
  a_matrix_.ClearMatrix();
  b_matrix_.ClearMatrix();
  c_matrix_.ClearMatrix();
  indexes_.clear();
  matrix_sum_on_process_.ClearMatrix();
  if (world_.rank() == 0) {
    int rows = static_cast<int>(task_data->inputs_count[0]);
    int columns = static_cast<int>(task_data->inputs_count[1]);
    std::vector<double> matrix_a(rows * columns);
    auto *in = reinterpret_cast<double *>(task_data->inputs[0]);
    std::copy(in, in + matrix_a.size(), matrix_a.begin());
    if (!CheckSquareSize(0)) {
      matrix_a = ConvertToSquareMatrix(std::max(rows, columns),
                                       rows > columns ? MatrixType::kRowMatrix : MatrixType::kColumnMatrix, matrix_a);
    }
    a_matrix_.SetBaseMatrix(std::move(matrix_a));
    a_matrix_.PreRoutine(MatrixType::kRowMatrix);
    rows = static_cast<int>(task_data->inputs_count[2]);
    columns = static_cast<int>(task_data->inputs_count[3]);
    std::vector<double> matrix_b(rows * columns);
    in = reinterpret_cast<double *>(task_data->inputs[1]);
    std::copy(in, in + matrix_b.size(), matrix_b.begin());
    if (!CheckSquareSize(2)) {
      matrix_b = ConvertToSquareMatrix(std::max(rows, columns),
                                       rows > columns ? MatrixType::kRowMatrix : MatrixType::kColumnMatrix, matrix_b);
    }
    b_matrix_.SetBaseMatrix(std::move(matrix_b));
    b_matrix_.PreRoutine(MatrixType::kColumnMatrix);
    CalculateIndexes();
  }
  return true;
}

std::vector<double> sarafanov_m_canon_mat_mul_all::CanonMatMulALL::ConvertToSquareMatrix(
    int need_size, MatrixType type, const std::vector<double> &matrx) {
  std::vector<double> matrix;
  int rows_counter = 0;
  int zero_columns = 0;
  switch (type) {
    case MatrixType::kRowMatrix:
      rows_counter = 1;
      zero_columns = need_size - (static_cast<int>(matrx.size()) / need_size);
      for (int i = 0; i < static_cast<int>(matrx.size()); ++i) {
        if ((need_size - zero_columns) * rows_counter - i == 0) {
          rows_counter++;
          for (int j = 0; j < zero_columns; ++j) {
            matrix.emplace_back(0.0);
          }
        }
        matrix.emplace_back(matrx[i]);
      }
      for (int i = 0; i < zero_columns; ++i) {
        matrix.emplace_back(0.0);
      }
      break;
    case MatrixType::kColumnMatrix:
      matrix = matrx;
      int zero_rows = need_size - (static_cast<int>(matrx.size()) / need_size);
      for (int i = 0; i < zero_rows * need_size; ++i) {
        matrix.emplace_back(0.0);
      }
      break;
  }
  return matrix;
}

bool sarafanov_m_canon_mat_mul_all::CanonMatMulALL::CheckSquareSize(int number) {
  return task_data->inputs_count[number] == task_data->inputs_count[number + 1];
}

bool sarafanov_m_canon_mat_mul_all::CanonMatMulALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return std::max(task_data->inputs_count[0], task_data->inputs_count[1]) *
               std::max(task_data->inputs_count[2], task_data->inputs_count[3]) ==
           task_data->outputs_count[0];
  }
  return true;
}

bool sarafanov_m_canon_mat_mul_all::CanonMatMulALL::RunImpl() {
  c_matrix_.ClearMatrix();
  boost::mpi::broadcast(world_, a_matrix_, 0);
  boost::mpi::broadcast(world_, b_matrix_, 0);
  boost::mpi::broadcast(world_, indexes_, 0);
  std::vector<CanonMatrix> mul_results(ppc::util::GetPPCNumThreads());
#pragma omp parallel
  {
#pragma omp for
    for (int i = indexes_[world_.rank()].first; i < indexes_[world_.rank()].second; ++i) {
      mul_results[omp_get_thread_num()] += a_matrix_.MultiplicateMatrix(b_matrix_, i);
    }
  }
  std::ranges::for_each(mul_results, [&](auto &matrix) {
    if (!matrix.IsEmpty()) {
      matrix_sum_on_process_ += matrix;
    }
  });
  if (matrix_sum_on_process_.IsEmpty()) {
    matrix_sum_on_process_.SetBaseMatrix(std::vector<double>(a_matrix_.GetSize()));
  }
  if (world_.rank() == 0) {
    std::vector<double> intermediate_values(a_matrix_.GetSize() * world_.size());
    auto sizes = std::vector<int>(world_.size(), static_cast<int>(a_matrix_.GetSize()));
    boost::mpi::gatherv(world_, matrix_sum_on_process_.GetMatrix(), intermediate_values.data(), sizes, 0);
    std::vector<double> answer(a_matrix_.GetSize());
    for (int i = 0; i < static_cast<int>(answer.size()); ++i) {
      for (int j = 0; j < world_.size(); ++j) {
        answer[i] += intermediate_values[i + (j * a_matrix_.GetSize())];
      }
    }
    c_matrix_.SetBaseMatrix(std::move(answer));
  } else {
    boost::mpi::gatherv(world_, matrix_sum_on_process_.GetMatrix(), 0);
  }
  return true;
}

bool sarafanov_m_canon_mat_mul_all::CanonMatMulALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(c_matrix_.GetMatrix(), reinterpret_cast<double *>(task_data->outputs[0]));
  }
  return true;
}
