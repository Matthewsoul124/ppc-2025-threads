#include "all/kondratev_ya_ccs_complex_multiplication/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

// NOLINTBEGIN(misc-include-cleaner)
#include <boost/serialization/complex.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
// NOLINTEND(misc-include-cleaner)

#include "core/util/include/util.hpp"

bool kondratev_ya_ccs_complex_multiplication_all::IsZero(const std::complex<double> &value) {
  return std::norm(value) < kEpsilonForZero;
}

bool kondratev_ya_ccs_complex_multiplication_all::IsEqual(const std::complex<double> &a,
                                                          const std::complex<double> &b) {
  return std::norm(a - b) <= kEpsilonForZero;
}

std::pair<int, int> kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::GetLocalColumnRange(int rank,
                                                                                                  int total_cols) {
  boost::mpi::communicator world;
  int world_size = world.size();

  int cols_per_proc = total_cols / world_size;
  int remainder = total_cols % world_size;

  int start_col = (rank * cols_per_proc) + std::min(rank, remainder);
  int cols_count = cols_per_proc + (rank < remainder ? 1 : 0);

  return {start_col, start_col + cols_count};
}

bool kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    a_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[0]);
    b_ = *reinterpret_cast<CCSMatrix *>(task_data->inputs[1]);

    if (a_.rows == 0 || a_.cols == 0 || b_.rows == 0 || b_.cols == 0 || a_.cols != b_.rows) {
      return false;
    }

    c_ = CCSMatrix({a_.rows, b_.cols});
  }
  return true;
}

bool kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == 2 && task_data->outputs_count[0] == 1 && task_data->inputs[0] != nullptr &&
           task_data->inputs[1] != nullptr && task_data->outputs[0] != nullptr;
  }
  return true;
}

void kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::CreateLocalMatrix(
    int rank, int size, const std::vector<std::pair<int, int>> &all_local_cols, CCSMatrix &out) {
  int current_process_start_col = all_local_cols[rank].first;
  int current_process_end_col = all_local_cols[rank].second;
  int num_local_cols = current_process_end_col - current_process_start_col;

  CCSMatrix local_b({b_.rows, num_local_cols});
  local_b.col_ptrs.resize(num_local_cols + 1, 0);

  local_b.values.reserve((b_.values.size() * num_local_cols / b_.cols) + 1);
  local_b.row_index.reserve((b_.values.size() * num_local_cols / b_.cols) + 1);

  for (int i = 0; i < num_local_cols; ++i) {
    int original_col_idx = current_process_start_col + i;
    local_b.col_ptrs[i] = static_cast<int>(local_b.values.size());

    int start_val_idx = b_.col_ptrs[original_col_idx];
    int end_val_idx = b_.col_ptrs[original_col_idx + 1];

    for (int k = start_val_idx; k < end_val_idx; ++k) {
      local_b.values.push_back(b_.values[k]);
      local_b.row_index.push_back(b_.row_index[k]);
    }
  }
  local_b.col_ptrs[num_local_cols] = static_cast<int>(local_b.values.size());

  out = local_b;
}

void kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::MergeResults(
    const std::vector<CCSMatrix> &all_results, const std::vector<std::pair<int, int>> &all_local_cols) {
  c_.col_ptrs.resize(b_.cols + 1, 0);
  size_t total_nnz_estimate = std::accumulate(
      all_results.begin(), all_results.end(), 0,
      [](size_t sum, const auto &block) { return sum + (block.col_ptrs.empty() ? 0 : block.col_ptrs.back()); });
  c_.values.reserve(total_nnz_estimate);
  c_.row_index.reserve(total_nnz_estimate);

  int size = world_.size();
  std::vector<std::vector<ColumnUpdateData>> thread_local_updates(size);
  ProcessBlocksInParallel(all_results, all_local_cols, thread_local_updates, size);

  std::vector<ColumnUpdateData> all_column_updates;
  if (b_.cols > 0) {
    all_column_updates.reserve(b_.cols);
  }

  for (int p_val = 0; p_val < size; p_val++) {
    all_column_updates.insert(all_column_updates.end(), std::make_move_iterator(thread_local_updates[p_val].begin()),
                              std::make_move_iterator(thread_local_updates[p_val].end()));
  }

  UpdateResultMatrix(all_column_updates);
}

void kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::ProcessBlocksInParallel(
    const std::vector<CCSMatrix> &all_results, const std::vector<std::pair<int, int>> &all_local_cols,
    std::vector<std::vector<ColumnUpdateData>> &thread_local_updates, int size) {
  std::vector<std::thread> data_preparation_threads;

  int requested_threads = ppc::util::GetPPCNumThreads();
  int num_worker_threads = requested_threads > 0 ? requested_threads : 1;

  if (num_worker_threads > 0) {
    data_preparation_threads.reserve(num_worker_threads);
  }

  for (int t_id = 0; t_id < num_worker_threads; ++t_id) {
    data_preparation_threads.emplace_back([&, t_id, num_worker_threads, size]() {
      for (int p_val = t_id; p_val < size; p_val += num_worker_threads) {
        const auto &received_block = all_results[p_val];
        int p_global_start_col = all_local_cols[p_val].first;
        if (received_block.cols > 0) {
          thread_local_updates[p_val].reserve(received_block.cols);
        }

        for (int local_col_idx = 0; local_col_idx < received_block.cols; local_col_idx++) {
          ColumnUpdateData update_data;
          update_data.global_col_idx = p_global_start_col + local_col_idx;

          const int start_data_idx = received_block.col_ptrs[local_col_idx];
          const int end_data_idx = received_block.col_ptrs[local_col_idx + 1];

          if (start_data_idx < end_data_idx) {
            update_data.values_to_insert.assign(received_block.values.begin() + start_data_idx,
                                                received_block.values.begin() + end_data_idx);
            update_data.row_indices_to_insert.assign(received_block.row_index.begin() + start_data_idx,
                                                     received_block.row_index.begin() + end_data_idx);
          }
          thread_local_updates[p_val].emplace_back(std::move(update_data));
        }
      }
    });
  }

  for (auto &t : data_preparation_threads) {
    t.join();
  }
}

void kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::UpdateResultMatrix(
    const std::vector<ColumnUpdateData> &all_column_updates) {
  for (const auto &update_data : all_column_updates) {
    if (update_data.global_col_idx < static_cast<int>(c_.col_ptrs.size())) {
      c_.col_ptrs[update_data.global_col_idx] = static_cast<int>(c_.values.size());
    }
    if (!update_data.values_to_insert.empty()) {
      c_.values.insert(c_.values.end(), update_data.values_to_insert.begin(), update_data.values_to_insert.end());
      c_.row_index.insert(c_.row_index.end(), update_data.row_indices_to_insert.begin(),
                          update_data.row_indices_to_insert.end());
    }
  }

  c_.col_ptrs[b_.cols] = static_cast<int>(c_.values.size());
}

bool kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  boost::mpi::broadcast(world_, a_, 0);
  boost::mpi::broadcast(world_, b_, 0);

  std::vector<std::pair<int, int>> all_local_cols(size);
  for (int p = 0; p < size; p++) {
    auto [start_col, end_col] = GetLocalColumnRange(p, b_.cols);
    all_local_cols[p] = {start_col, end_col};
  }

  CCSMatrix local_b;
  CreateLocalMatrix(rank, size, all_local_cols, local_b);
  CCSMatrix computed_local_block = a_ * local_b;

  std::vector<CCSMatrix> all_results;
  boost::mpi::gather(world_, computed_local_block, all_results, 0);

  if (rank == 0) {
    MergeResults(all_results, all_local_cols);
  }

  return true;
}

bool kondratev_ya_ccs_complex_multiplication_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<CCSMatrix *>(task_data->outputs[0]) = c_;
  }
  return true;
}

kondratev_ya_ccs_complex_multiplication_all::CCSMatrix
kondratev_ya_ccs_complex_multiplication_all::CCSMatrix::operator*(const CCSMatrix &other) const {
  CCSMatrix result({rows, other.cols});
  result.values.reserve(std::min(rows * other.cols, static_cast<int>(values.size() * other.values.size())));
  result.row_index.reserve(result.values.capacity());

  std::vector<std::complex<double>> temp_col(rows, std::complex<double>(0.0, 0.0));

  for (int result_col = 0; result_col < other.cols; result_col++) {
    for (int k = other.col_ptrs[result_col]; k < other.col_ptrs[result_col + 1]; k++) {
      int row_other = other.row_index[k];
      std::complex<double> val_other = other.values[k];

      for (int i = col_ptrs[row_other]; i < col_ptrs[row_other + 1]; i++) {
        int row_this = row_index[i];
        temp_col[row_this] += values[i] * val_other;
      }
    }

    result.col_ptrs[result_col] = static_cast<int>(result.values.size());
    for (int i = 0; i < rows; i++) {
      if (!IsZero(temp_col[i])) {
        result.values.emplace_back(temp_col[i]);
        result.row_index.emplace_back(i);

        temp_col[i] = std::complex<double>(0.0, 0.0);
      }
    }
  }

  result.col_ptrs[other.cols] = static_cast<int>(result.values.size());

  return result;
}