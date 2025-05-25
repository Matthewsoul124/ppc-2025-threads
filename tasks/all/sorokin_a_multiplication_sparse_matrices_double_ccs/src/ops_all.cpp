#include "all/sorokin_a_multiplication_sparse_matrices_double_ccs/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <numeric>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gather.hpp"
#include "boost/mpi/collectives/gatherv.hpp"

namespace {

void ComputeLocalNNZ(int start_col, int local_cols, const std::vector<int>& b_col_ptr,
                     const std::vector<int>& b_row_indices, const std::vector<int>& a_col_ptr,
                     const std::vector<int>& a_row_indices, int m, std::vector<int>& local_nnz) {
  for (int j_local = 0; j_local < local_cols; ++j_local) {
    int j = start_col + j_local;
    std::vector<bool> temp_used(m, false);
    for (int t = b_col_ptr[j]; t < b_col_ptr[j + 1]; ++t) {
      int row_b = b_row_indices[t];
      for (int i = a_col_ptr[row_b]; i < a_col_ptr[row_b + 1]; ++i) {
        int row_a = a_row_indices[i];
        if (!temp_used[row_a]) {
          temp_used[row_a] = true;
          local_nnz[j_local]++;
        }
      }
    }
  }
}

void GatherNNZData(boost::mpi::communicator& world, const std::vector<int>& local_nnz, int local_cols, int start_col,
                   std::vector<int>& nnz_per_column, std::vector<int>& recv_counts, std::vector<int>& displs) {
  int rank = world.rank();
  int size = world.size();

  if (rank == 0) {
    std::vector<int> all_start_cols(size);
    std::vector<int> all_local_cols(size);
    boost::mpi::gather(world, start_col, all_start_cols, 0);
    boost::mpi::gather(world, local_cols, all_local_cols, 0);

    for (int p = 0; p < size; ++p) {
      recv_counts[p] = all_local_cols[p];
      displs[p] = all_start_cols[p];
    }
  } else {
    boost::mpi::gather(world, start_col, 0);
    boost::mpi::gather(world, local_cols, 0);
  }

  if (rank == 0) {
    boost::mpi::gatherv(world.split(0), local_nnz.data(), local_cols, nnz_per_column.data(), recv_counts, displs, 0);
  } else {
    boost::mpi::gatherv(world.split(0), local_nnz.data(), local_cols, 0);
  }
}

void BuildCColPtr(boost::mpi::communicator& world, const std::vector<int>& nnz_per_column,
                  std::vector<int>& c_col_ptr) {
  size_t n = nnz_per_column.size();
  if (world.rank() == 0) {
    c_col_ptr.resize(n + 1);
    c_col_ptr[0] = 0;
    std::partial_sum(nnz_per_column.begin(), nnz_per_column.end(), c_col_ptr.begin() + 1);
  }
  boost::mpi::broadcast(world, c_col_ptr, 0);
}

void ComputeCValuesAndIndices(const std::vector<double>& a_values, const std::vector<int>& a_row_indices,
                              const std::vector<int>& a_col_ptr, const std::vector<double>& b_values,
                              const std::vector<int>& b_row_indices, const std::vector<int>& b_col_ptr, int m, int n,
                              std::vector<double>& c_values, std::vector<int>& c_row_indices,
                              const std::vector<int>& c_col_ptr) {
  int total_nnz = c_col_ptr[n];
  c_values.resize(total_nnz);
  c_row_indices.resize(total_nnz);

#pragma omp parallel for
  for (int j = 0; j < n; ++j) {
    std::vector<double> temp_values(m, 0.0);
    std::vector<bool> temp_used(m, false);

    for (int t = b_col_ptr[j]; t < b_col_ptr[j + 1]; ++t) {
      int row_b = b_row_indices[t];
      double val_b = b_values[t];
      for (int i = a_col_ptr[row_b]; i < a_col_ptr[row_b + 1]; ++i) {
        int row_a = a_row_indices[i];
        temp_values[row_a] += a_values[i] * val_b;
        temp_used[row_a] = true;
      }
    }

    int pos = c_col_ptr[j];
    int count = 0;
    for (int i = 0; i < m; ++i) {
      if (temp_used[i]) {
        c_row_indices[pos + count] = i;
        c_values[pos + count] = temp_values[i];
        count++;
      }
    }
  }
}

}  // namespace

namespace sorokin_a_multiplication_sparse_matrices_double_ccs_all {

void MultiplyCCS(boost::mpi::communicator& world, const std::vector<double>& a_values,
                 const std::vector<int>& a_row_indices, int m, const std::vector<int>& a_col_ptr,
                 const std::vector<double>& b_values, const std::vector<int>& b_row_indices, int k,
                 const std::vector<int>& b_col_ptr, std::vector<double>& c_values, std::vector<int>& c_row_indices,
                 int n, std::vector<int>& c_col_ptr) {
  int rank = world.rank();
  int size = world.size();

  int cols_per_process = n / size;
  int remainder = n % size;
  int start_col = (rank * cols_per_process) + std::min(rank, remainder);
  int end_col = start_col + cols_per_process + (rank < remainder ? 1 : 0);
  int local_cols = end_col - start_col;

  std::vector<int> local_nnz(local_cols, 0);
  ComputeLocalNNZ(start_col, local_cols, b_col_ptr, b_row_indices, a_col_ptr, a_row_indices, m, local_nnz);

  std::vector<int> nnz_per_column(n, 0);
  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  GatherNNZData(world, local_nnz, local_cols, start_col, nnz_per_column, recv_counts, displs);

  BuildCColPtr(world, nnz_per_column, c_col_ptr);

  if (rank == 0) {
    ComputeCValuesAndIndices(a_values, a_row_indices, a_col_ptr, b_values, b_row_indices, b_col_ptr, m, n, c_values,
                             c_row_indices, c_col_ptr);
  } else {
    c_col_ptr.clear();
    c_values.clear();
    c_row_indices.clear();
  }

  world.barrier();
}

}  // namespace sorokin_a_multiplication_sparse_matrices_double_ccs_all

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::PreProcessingImpl() {
  // Init value for input and output
  M_ = static_cast<int>(task_data->inputs_count[0]);
  K_ = static_cast<int>(task_data->inputs_count[1]);
  N_ = static_cast<int>(task_data->inputs_count[2]);
  auto* current_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  A_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[3]);
  current_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  std::vector<double> a_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[4]);
  A_row_indices_.resize(a_row_indices_d.size());
  std::ranges::transform(a_row_indices_d.begin(), a_row_indices_d.end(), A_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[2]);
  std::vector<double> a_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[5]);
  A_col_ptr_.resize(a_col_ptr_d.size());
  std::ranges::transform(a_col_ptr_d.begin(), a_col_ptr_d.end(), A_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[3]);
  B_values_ = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[6]);
  current_ptr = reinterpret_cast<double*>(task_data->inputs[4]);
  std::vector<double> b_row_indices_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[7]);
  B_row_indices_.resize(b_row_indices_d.size());
  std::ranges::transform(b_row_indices_d.begin(), b_row_indices_d.end(), B_row_indices_.begin(),
                         [](double x) { return static_cast<int>(x); });
  current_ptr = reinterpret_cast<double*>(task_data->inputs[5]);
  std::vector<double> b_col_ptr_d = std::vector<double>(current_ptr, current_ptr + task_data->inputs_count[8]);
  B_col_ptr_.resize(b_col_ptr_d.size());
  std::ranges::transform(b_col_ptr_d.begin(), b_col_ptr_d.end(), B_col_ptr_.begin(),
                         [](double x) { return static_cast<int>(x); });
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0 && task_data->inputs_count[2] > 0;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::RunImpl() {
  MultiplyCCS(world_, A_values_, A_row_indices_, M_, A_col_ptr_, B_values_, B_row_indices_, K_, B_col_ptr_, C_values_,
              C_row_indices_, N_, C_col_ptr_);
  return true;
}

bool sorokin_a_multiplication_sparse_matrices_double_ccs_all::TestTaskALL::PostProcessingImpl() {
  std::vector<double> c_row_indices_d(C_row_indices_.size());
  std::vector<double> c_col_ptr_d(C_col_ptr_.size());
  std::ranges::transform(C_row_indices_.begin(), C_row_indices_.end(), c_row_indices_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  std::ranges::transform(C_col_ptr_.begin(), C_col_ptr_.end(), c_col_ptr_d.begin(),
                         [](int x) { return static_cast<double>(x); });
  for (size_t i = 0; i < C_values_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = C_values_[i];
  }
  for (size_t i = 0; i < c_row_indices_d.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[1])[i] = c_row_indices_d[i];
  }
  for (size_t i = 0; i < c_col_ptr_d.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[2])[i] = c_col_ptr_d[i];
  }
  return true;
}
