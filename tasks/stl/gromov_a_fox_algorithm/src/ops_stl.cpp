#include "stl/gromov_a_fox_algorithm/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
void FoxBlockMul(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int n,
                 int block_size, int stage, int i, int j) {
  int start_k = stage * block_size;
  int end_k = std::min((stage + 1) * block_size, n);

  for (int bk = start_k; bk < end_k; ++bk) {
    for (int bi = i; bi < i + block_size && bi < n; ++bi) {
      for (int bj = j; bj < j + block_size && bj < n; ++bj) {
        c[(bi * n) + bj] += a[(bi * n) + bk] * b[(bk * n) + bj];
      }
    }
  }
}
}  // namespace

bool gromov_a_fox_algorithm_stl::TestTaskSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }

  unsigned int matrix_size = input_size / 2;
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  A_ = std::vector<double>(in_ptr, in_ptr + matrix_size);
  B_ = std::vector<double>(in_ptr + matrix_size, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<double>(output_size, 0.0);

  n_ = static_cast<int>(std::sqrt(matrix_size));
  if (n_ * n_ != static_cast<int>(matrix_size)) {
    return false;
  }

  block_size_ = static_cast<int>(std::sqrt(n_));
  for (int i = block_size_; i >= 1; --i) {
    if (n_ % i == 0) {
      block_size_ = i;
      break;
    }
  }
  for (int i = block_size_ + 1; i <= n_; ++i) {
    if (n_ % i == 0) {
      if (std::abs(i - static_cast<int>(std::sqrt(n_))) < std::abs(block_size_ - static_cast<int>(std::sqrt(n_)))) {
        block_size_ = i;
      }
      break;
    }
  }
  return block_size_ > 0;
}

bool gromov_a_fox_algorithm_stl::TestTaskSTL::ValidationImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  if (input_size % 2 != 0) {
    return false;
  }
  unsigned int matrix_size = input_size / 2;
  auto sqrt_matrix_size = static_cast<unsigned int>(std::sqrt(matrix_size));
  return matrix_size == task_data->outputs_count[0] && sqrt_matrix_size * sqrt_matrix_size == matrix_size;
}

bool gromov_a_fox_algorithm_stl::TestTaskSTL::RunImpl() {
  int num_blocks = (n_ + block_size_ - 1) / block_size_;
  const int num_threads = std::min(ppc::util::GetPPCNumThreads(), 8);
  std::vector<std::thread> threads;

  int num_block_rows = num_blocks;
  int num_block_cols = num_blocks;
  int total_tasks = num_block_rows * num_block_cols;

  for (int stage = 0; stage < num_blocks; ++stage) {
    threads.clear();
    threads.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
      threads.emplace_back([&, t, stage]() {
        for (int task_id = t; task_id < total_tasks; task_id += num_threads) {
          int i = (task_id / num_block_cols) * block_size_;
          int j = (task_id % num_block_cols) * block_size_;
          if (i < n_ && j < n_) {
            FoxBlockMul(A_, B_, output_, n_, block_size_, stage, i, j);
          }
        }
      });
    }

    for (auto& thread : threads) {
      thread.join();
    }
  }

  return true;
}

bool gromov_a_fox_algorithm_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}