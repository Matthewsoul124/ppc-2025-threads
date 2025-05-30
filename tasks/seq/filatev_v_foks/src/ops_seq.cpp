#include "seq/filatev_v_foks/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool filatev_v_foks_seq::Focks::PreProcessingImpl() {
  size_block_ = task_data->inputs_count[4];
  size_a_.n = task_data->inputs_count[0];
  size_a_.m = task_data->inputs_count[1];
  size_b_.n = task_data->inputs_count[2];
  size_b_.m = task_data->inputs_count[3];

  size_c_.n = task_data->outputs_count[0];
  size_c_.m = task_data->outputs_count[1];

  size_ = std::max(size_a_.n, size_a_.m);
  size_ = std::max(size_, size_b_.n);
  size_ = std::max(size_, size_b_.m);

  size_ = (size_ % size_block_ == 0) ? size_ : ((size_ % size_block_) + 1) * size_block_;

  matrix_a_.assign(size_ * size_, 0);
  matrix_b_.assign(size_ * size_, 0);

  auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
  auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);

  for (size_t i = 0; i < size_a_.m; ++i) {
    std::copy(temp_a + (i * size_a_.n), temp_a + ((i + 1) * size_a_.n), matrix_a_.data() + (i * size_));
  }
  for (size_t i = 0; i < size_b_.m; ++i) {
    std::copy(temp_b + (i * size_b_.n), temp_b + ((i + 1) * size_b_.n), matrix_b_.data() + (i * size_));
  }

  return true;
}

bool filatev_v_foks_seq::Focks::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->inputs_count[3] &&
         task_data->inputs_count[1] == task_data->inputs_count[2] &&
         task_data->outputs_count[0] == task_data->inputs_count[2] &&
         task_data->outputs_count[1] == task_data->inputs_count[1] && task_data->inputs_count[4] > 0;
}

bool filatev_v_foks_seq::Focks::RunImpl() {
  matrix_c_.assign(size_ * size_, 0);

  size_t grid_size = size_ / size_block_;

  for (size_t step = 0; step < grid_size; ++step) {
    for (size_t i = 0; i < grid_size; ++i) {
      for (size_t j = 0; j < grid_size; ++j) {
        size_t root = (i + step) % grid_size;
        for (size_t bi = 0; bi < size_block_; ++bi) {
          for (size_t bj = 0; bj < size_block_; ++bj) {
            for (size_t bk = 0; bk < size_block_; ++bk) {
              matrix_c_[((i * size_block_ + bi) * size_) + (j * size_block_) + bj] +=
                  matrix_a_[((i * size_block_ + bi) * size_) + (root * size_block_) + bk] *
                  matrix_b_[((root * size_block_ + bk) * size_) + (j * size_block_) + bj];
            }
          }
        }
      }
    }
  }

  return true;
}

bool filatev_v_foks_seq::Focks::PostProcessingImpl() {
  auto *temp = reinterpret_cast<double *>(task_data->outputs[0]);
  for (size_t i = 0; i < size_c_.m; ++i) {
    std::copy(matrix_c_.data() + (i * size_), matrix_c_.data() + (i * size_) + size_c_.n, temp + (i * size_c_.n));
  }
  return true;
}
