#include "stl/kolodkin_g_multiplication_matrix_CRS/include/ops_stl.hpp"

#include <cmath>
#include <complex>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <iostream>
#include <thread>
#include <vector>

void kolodkin_g_multiplication_matrix_stl::SparseMatrixCRS::AddValue(int row, Complex value, int col) {
  for (int j = rowPtr[row]; j < rowPtr[row + 1]; ++j) {
    if (colIndices[j] == col) {
      values[j] += value;
      return;
    }
  }
  colIndices.emplace_back(col);
  values.emplace_back(value);
  for (int i = row + 1; i <= numRows; ++i) {
    rowPtr[i]++;
  }
}

void kolodkin_g_multiplication_matrix_stl::SparseMatrixCRS::PrintSparseMatrix(
    const kolodkin_g_multiplication_matrix_stl::SparseMatrixCRS& matrix) {
  for (int i = 0; i < matrix.numRows; ++i) {
    for (int j = matrix.rowPtr[i]; j < matrix.rowPtr[i + 1]; ++j) {
      std::cout << "Element at (" << i << ", " << matrix.colIndices[j] << ") = " << matrix.values[j] << '\n';
    }
  }
}

bool kolodkin_g_multiplication_matrix_stl::AreEqualElems(const Complex& a, const Complex& b, double epsilon) {
  return std::abs(a.real() - b.real()) < epsilon && std::abs(a.imag() - b.imag()) < epsilon;
}

std::vector<Complex> kolodkin_g_multiplication_matrix_stl::ParseMatrixIntoVec(const SparseMatrixCRS& mat) {
  std::vector<Complex> res = {};
  res.reserve(5 + mat.values.size() + mat.colIndices.size() + mat.rowPtr.size());
  res.emplace_back((double)mat.numRows);
  res.emplace_back((double)mat.numCols);
  res.emplace_back((double)mat.values.size());
  res.emplace_back((double)mat.colIndices.size());
  res.emplace_back((double)mat.rowPtr.size());
  for (unsigned int i = 0; i < (unsigned int)mat.values.size(); i++) {
    res.emplace_back(mat.values[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.colIndices.size(); i++) {
    res.emplace_back(mat.colIndices[i]);
  }
  for (unsigned int i = 0; i < (unsigned int)mat.rowPtr.size(); i++) {
    res.emplace_back(mat.rowPtr[i]);
  }
  return res;
}
bool kolodkin_g_multiplication_matrix_stl::CheckMatrixesEquality(
    const kolodkin_g_multiplication_matrix_stl::SparseMatrixCRS& a,
    const kolodkin_g_multiplication_matrix_stl::SparseMatrixCRS& b) {
  if (a.numCols != b.numCols || a.numRows != b.numRows) {
    return false;
  }
  for (unsigned int i = 0; i < (unsigned int)a.numRows; ++i) {
    unsigned int this_row_start = a.rowPtr[i];
    unsigned int this_row_end = a.rowPtr[i + 1];
    unsigned int other_row_start = b.rowPtr[i];
    unsigned int other_row_end = b.rowPtr[i + 1];
    if ((this_row_end - this_row_start) != (other_row_end - other_row_start)) {
      return false;
    }
    for (unsigned int j = this_row_start; j < this_row_end; ++j) {
      bool found = false;
      for (unsigned int k = other_row_start; k < other_row_end; ++k) {
        if (a.colIndices[j] == b.colIndices[k] && AreEqualElems(a.values[j], b.values[k], 0.000001)) {
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
kolodkin_g_multiplication_matrix_stl::SparseMatrixCRS kolodkin_g_multiplication_matrix_stl::ParseVectorIntoMatrix(
    std::vector<Complex>& vec) {
  SparseMatrixCRS res;
  res.numRows = (int)vec[0].real();
  res.numCols = (int)vec[1].real();
  auto values_size = (unsigned int)vec[2].real();
  auto col_indices_size = (unsigned int)vec[3].real();
  auto row_ptr_size = (unsigned int)vec[4].real();
  res.values.reserve(values_size);
  res.colIndices.reserve(col_indices_size);
  res.rowPtr.reserve(row_ptr_size);
  for (unsigned int i = 0; i < values_size; i++) {
    res.values.emplace_back(vec[5 + i]);
  }
  for (unsigned int i = 0; i < col_indices_size; i++) {
    res.colIndices.emplace_back((int)vec[5 + values_size + i].real());
  }
  for (unsigned int i = 0; i < row_ptr_size; i++) {
    res.rowPtr.emplace_back((int)vec[5 + values_size + col_indices_size + i].real());
  }
  return res;
}

bool kolodkin_g_multiplication_matrix_stl::TestTaskSTL::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  input_ = std::vector<Complex>(in_ptr, in_ptr + input_size);
  std::vector<Complex> matrix_a = {};
  std::vector<Complex> matrix_b = {};
  matrix_a.reserve(5 + (unsigned int)(input_[2].real() + input_[3].real() + input_[4].real()));
  matrix_b.reserve(input_.size() - (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()));
  for (unsigned int i = 0; i < (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real()); i++) {
    matrix_a.emplace_back(input_[i]);
  }
  for (auto i = (unsigned int)(5 + input_[2].real() + input_[3].real() + input_[4].real());
       i < (unsigned int)input_.size(); i++) {
    matrix_b.emplace_back(input_[i]);
  }
  A_ = ParseVectorIntoMatrix(matrix_a);
  B_ = ParseVectorIntoMatrix(matrix_b);
  return true;
}

bool kolodkin_g_multiplication_matrix_stl::TestTaskSTL::ValidationImpl() {
  // Check equality of counts elements
  unsigned int input_size = task_data->inputs_count[0];
  auto* in_ptr = reinterpret_cast<Complex*>(task_data->inputs[0]);
  std::vector<Complex> vec = std::vector<Complex>(in_ptr, in_ptr + input_size);
  return !(vec[1] != vec[5 + (int)(vec[2].real() + vec[3].real() + vec[4].real())].real());
}

bool kolodkin_g_multiplication_matrix_stl::TestTaskSTL::RunImpl() {
  SparseMatrixCRS c(A_.numRows, B_.numCols);
  const int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads(num_threads);
  std::vector<SparseMatrixCRS> local_results(num_threads, SparseMatrixCRS(A_.numRows, B_.numCols));

  auto worker = [&](unsigned int start_row, unsigned int end_row, int thread_index) {
    SparseMatrixCRS local_c(A_.numRows, B_.numCols);
    for (unsigned int i = start_row; i < end_row; ++i) {
      for (unsigned int j = A_.rowPtr[i]; j < (unsigned int)A_.rowPtr[i + 1]; ++j) {
        unsigned int col_a = A_.colIndices[j];
        Complex value_a = A_.values[j];
        for (unsigned int k = B_.rowPtr[col_a]; k < (unsigned int)B_.rowPtr[col_a + 1]; ++k) {
          unsigned int col_b = B_.colIndices[k];
          Complex value_b = B_.values[k];

          local_c.AddValue((int)i, value_a * value_b, (int)col_b);
        }
      }
    }
    local_results[thread_index] = local_c;
  };

  unsigned int rows_per_thread = A_.numRows / num_threads;
  for (int t = 0; t < num_threads; ++t) {
    unsigned int start_row = t * rows_per_thread;
    unsigned int end_row = (t == num_threads - 1) ? A_.numRows : start_row + rows_per_thread;
    threads[t] = std::thread(worker, start_row, end_row, t);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  for (const auto& local_c : local_results) {
    for (unsigned int i = 0; i < (unsigned int)local_c.numRows; ++i) {
      for (unsigned int j = local_c.rowPtr[i]; j < (unsigned int)local_c.rowPtr[i + 1]; ++j) {
        c.AddValue((int)i, local_c.values[j], local_c.colIndices[j]);
      }
    }
  }

  output_ = ParseMatrixIntoVec(c);
  return true;
}

bool kolodkin_g_multiplication_matrix_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<Complex*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
