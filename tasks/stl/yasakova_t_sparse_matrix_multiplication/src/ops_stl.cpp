#include "stl/yasakova_t_sparse_matrix_multiplication/include/ops_stl.hpp"

#include <cmath>
#include <complex>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

void yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage::InsertElement(int row_idx,
                                                                                      std::complex<double> val,
                                                                                      int col_idx) {
  for (int j = rowPointers[row_idx]; j < rowPointers[row_idx + 1]; ++j) {
    if (columnIndices[j] == col_idx) {
      nonZeroValues[j] += val;
      return;
    }
  }
  columnIndices.emplace_back(col_idx);
  nonZeroValues.emplace_back(val);
  for (int i = row_idx + 1; i <= rowCount; ++i) {
    rowPointers[i]++;
  }
}

void yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage::DisplayMatrix(
    const yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage& matrix) {
  for (int i = 0; i < matrix.rowCount; ++i) {
    for (int j = matrix.rowPointers[i]; j < matrix.rowPointers[i + 1]; ++j) {
      std::cout << "Element at (" << i << ", " << matrix.columnIndices[j] << ") = " << matrix.nonZeroValues[j] << '\n';
    }
  }
}

bool yasakova_t_sparse_matrix_multiplication_stl::AreClose(const std::complex<double>& left_matrix,
                                                           const std::complex<double>& right_matrix, double epsilon) {
  return std::abs(left_matrix.real() - right_matrix.real()) < epsilon &&
         std::abs(left_matrix.imag() - right_matrix.imag()) < epsilon;
}

std::vector<std::complex<double>> yasakova_t_sparse_matrix_multiplication_stl::ConvertToDense(
    const CompressedRowStorage& mat) {
  std::vector<std::complex<double>> actual_result = {};
  actual_result.reserve(5 + mat.nonZeroValues.size() + mat.columnIndices.size() + mat.rowPointers.size());
  actual_result.emplace_back((double)mat.rowCount);
  actual_result.emplace_back((double)mat.columnCount);
  actual_result.emplace_back((double)mat.nonZeroValues.size());
  actual_result.emplace_back((double)mat.columnIndices.size());
  actual_result.emplace_back((double)mat.rowPointers.size());
  for (unsigned int i = 0; i < (unsigned int)mat.nonZeroValues.size(); i++) {
    actual_result.emplace_back(mat.nonZeroValues[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.columnIndices.size(); i++) {
    actual_result.emplace_back(mat.columnIndices[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.rowPointers.size(); i++) {
    actual_result.emplace_back(mat.rowPointers[i]);
  }
  return actual_result;
}
bool yasakova_t_sparse_matrix_multiplication_stl::CompareMatrices(
    const yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage& left_matrix,
    const yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage& right_matrix) {
  if (left_matrix.columnCount != right_matrix.columnCount || left_matrix.rowCount != right_matrix.rowCount) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)left_matrix.rowCount; ++i) {
    unsigned int this_row_start = left_matrix.rowPointers[i];
    unsigned int this_row_end = left_matrix.rowPointers[i + 1];
    unsigned int other_row_start = right_matrix.rowPointers[i];
    unsigned int other_row_end = right_matrix.rowPointers[i + 1];
    if ((this_row_end - this_row_start) != (other_row_end - other_row_start)) {
      return false;
    }
    for (unsigned int j = this_row_start; j < this_row_end; ++j) {
      bool found = false;
      for (unsigned int k = other_row_start; k < other_row_end; ++k) {
        if (left_matrix.columnIndices[j] == right_matrix.columnIndices[k] &&
            AreClose(left_matrix.nonZeroValues[j], right_matrix.nonZeroValues[k], 0.000001)) {
          found = true;
          break;
        }
      }
      if (!found) {
        return false;
      }
    }
  }
  return true;
}
yasakova_t_sparse_matrix_multiplication_stl::CompressedRowStorage
yasakova_t_sparse_matrix_multiplication_stl::ConvertToSparse(std::vector<std::complex<double>>& vec) {
  CompressedRowStorage actual_result;
  actual_result.rowCount = (int)vec[0].real();
  actual_result.columnCount = (int)vec[1].real();
  auto values_size = (unsigned int)vec[2].real();
  auto col_indices_size = (unsigned int)vec[3].real();
  auto row_ptr_size = (unsigned int)vec[4].real();
  actual_result.nonZeroValues.reserve(values_size);
  actual_result.columnIndices.reserve(col_indices_size);
  actual_result.rowPointers.reserve(row_ptr_size);
  for (unsigned int i = 0; i < values_size; i++) {
    actual_result.nonZeroValues.emplace_back(vec[5 + i]);
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    actual_result.columnIndices.emplace_back((int)vec[5 + values_size + i].real());
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    actual_result.rowPointers.emplace_back((int)vec[5 + values_size + col_indices_size + i].real());
  }
  return actual_result;
}

bool yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask::PreProcessingImpl() {
  // Init val for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto* input_ptr = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);
  inputData_ = std::vector<std::complex<double>>(input_ptr, input_ptr + input_size);
  std::vector<std::complex<double>> matrix_a = {};
  std::vector<std::complex<double>> matrix_b = {};
  matrix_a.reserve(5 + (unsigned int)(inputData_[2].real() + inputData_[3].real() + inputData_[4].real()));
  matrix_b.reserve(inputData_.size() -
                   (unsigned int)(5 + inputData_[2].real() + inputData_[3].real() + inputData_[4].real()));
  for (unsigned int i = 0; i < (unsigned int)(5 + inputData_[2].real() + inputData_[3].real() + inputData_[4].real());
       i++) {
    matrix_a.emplace_back(inputData_[i]);
  }
  for (auto i = (unsigned int)(5 + inputData_[2].real() + inputData_[3].real() + inputData_[4].real());
       i < (unsigned int)inputData_.size(); i++) {
    matrix_b.emplace_back(inputData_[i]);
  }
  firstMatrix_ = ConvertToSparse(matrix_a);
  secondMatrix_ = ConvertToSparse(matrix_b);
  return true;
}

bool yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask::ValidationImpl() {
  // Check equality of counts elements
  unsigned int input_size = task_data->inputs_count[0];
  auto* input_ptr = reinterpret_cast<std::complex<double>*>(task_data->inputs[0]);
  std::vector<std::complex<double>> vec = std::vector<std::complex<double>>(input_ptr, input_ptr + input_size);
  return !(vec[1] != vec[5 + (int)(vec[2].real() + vec[3].real() + vec[4].real())].real());
}

bool yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask::RunImpl() {
  CompressedRowStorage expected_result(firstMatrix_.rowCount, secondMatrix_.columnCount);
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);
  std::vector<CompressedRowStorage> thread_results(
      num_threads, CompressedRowStorage(firstMatrix_.rowCount, secondMatrix_.columnCount));

  auto multiply_task = [&](unsigned int start_row, unsigned int end_row, int thread_index) {
    CompressedRowStorage local_result(firstMatrix_.rowCount, secondMatrix_.columnCount);
    for (unsigned int i = start_row; i < end_row; ++i) {
      for (unsigned int j = firstMatrix_.rowPointers[i]; j < (unsigned int)firstMatrix_.rowPointers[i + 1]; ++j) {
        unsigned int col_a = firstMatrix_.columnIndices[j];
        std::complex<double> value_a = firstMatrix_.nonZeroValues[j];
        for (unsigned int k = secondMatrix_.rowPointers[col_a]; k < (unsigned int)secondMatrix_.rowPointers[col_a + 1];
             ++k) {
          unsigned int col_b = secondMatrix_.columnIndices[k];
          std::complex<double> value_b = secondMatrix_.nonZeroValues[k];

          local_result.InsertElement((int)i, value_a * value_b, (int)col_b);
        }
      }
    }
    thread_results[thread_index] = local_result;
  };

  unsigned int rows_per_thread = firstMatrix_.rowCount / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    unsigned int start_row = t * rows_per_thread;
    unsigned int end_row = (t == num_threads - 1) ? firstMatrix_.rowCount : start_row + rows_per_thread;
    threads[t] = std::thread(multiply_task, start_row, end_row, t);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& local_result : thread_results) {
    for (unsigned int i = 0; i < (unsigned int)local_result.rowCount; ++i) {
      for (unsigned int j = local_result.rowPointers[i]; j < (unsigned int)local_result.rowPointers[i + 1]; ++j) {
        expected_result.InsertElement((int)i, local_result.nonZeroValues[j], local_result.columnIndices[j]);
      }
    }
  }

  resultData_ = ConvertToDense(expected_result);
  return true;
}

bool yasakova_t_sparse_matrix_multiplication_stl::SparseMatrixMultiTask::PostProcessingImpl() {
  for (size_t i = 0; i < resultData_.size(); i++) {
    reinterpret_cast<std::complex<double>*>(task_data->outputs[0])[i] = resultData_[i];
  }
  return true;
}