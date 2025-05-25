#include "all/filatev_v_foks/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/serialization/vector.hpp>  // IWYU pragma: keep
#include <cmath>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool filatev_v_foks_all::Focks::PreProcessingImpl() {
  if (world_.rank() == 0) {
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
  }
  return true;
}

bool filatev_v_foks_all::Focks::ValidationImpl() {
  bool valid = true;
  if (world_.rank() == 0) {
    valid = task_data->inputs_count[0] == task_data->inputs_count[3] &&
            task_data->inputs_count[1] == task_data->inputs_count[2] &&
            task_data->outputs_count[0] == task_data->inputs_count[2] &&
            task_data->outputs_count[1] == task_data->inputs_count[1] && task_data->inputs_count[4] > 0;
  }
  boost::mpi::broadcast(world_, valid, 0);
  return valid;
}

void filatev_v_foks_all::Focks::Worker(size_t start_step, size_t end_step, size_t grid_size, std::mutex &mtx) {
  for (size_t step_i_j = start_step; step_i_j < end_step; ++step_i_j) {
    size_t step = step_i_j / (grid_size * grid_size);
    size_t i = (step_i_j % (grid_size * grid_size)) / grid_size;
    size_t j = step_i_j % grid_size;

    size_t root = (i + step) % grid_size;
    std::vector<double> local_block(size_block_ * size_block_, 0);

    for (size_t bi = 0; bi < size_block_; ++bi) {
      for (size_t bj = 0; bj < size_block_; ++bj) {
        for (size_t bk = 0; bk < size_block_; ++bk) {
          local_block[(bi * size_block_) + bj] +=
              matrix_a_[((i * size_block_ + bi) * size_) + (root * size_block_) + bk] *
              matrix_b_[((root * size_block_ + bk) * size_) + (j * size_block_) + bj];
        }
      }
    }

    std::lock_guard<std::mutex> lock(mtx);
    for (size_t bi = 0; bi < size_block_; ++bi) {
      for (size_t bj = 0; bj < size_block_; ++bj) {
        matrix_c_[((i * size_block_ + bi) * size_) + (j * size_block_) + bj] += local_block[(bi * size_block_) + bj];
      }
    }
  }
}

bool filatev_v_foks_all::Focks::RunImpl() {
  boost::mpi::broadcast(world_, size_block_, 0);
  boost::mpi::broadcast(world_, size_, 0);
  if (world_.rank() != 0) {
    matrix_a_.resize(size_ * size_);
    matrix_b_.resize(size_ * size_);
  }
  boost::mpi::broadcast(world_, matrix_a_, 0);
  boost::mpi::broadcast(world_, matrix_b_, 0);

  size_t grid_size = size_ / size_block_;

  matrix_c_.assign(size_ * size_, 0);

  size_t total_steps = grid_size * grid_size * grid_size;
  size_t steps_per_process = total_steps / world_.size();
  size_t remainder = total_steps % world_.size();

  size_t start_step = (world_.rank() * steps_per_process) + std::min<size_t>(world_.rank(), remainder);
  size_t end_step = start_step + steps_per_process + (static_cast<size_t>(world_.rank()) < remainder ? 1 : 0);

  size_t num_threads = ppc::util::GetPPCNumThreads();
  std::mutex mtx;

  if (end_step - start_step >= num_threads) {
    std::vector<std::thread> threads(num_threads);
    size_t steps_per_thread = (end_step - start_step) / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
      size_t thread_start = start_step + (t * steps_per_thread);
      size_t thread_end = (t == num_threads - 1) ? end_step : thread_start + steps_per_thread;
      threads[t] = std::thread(&Focks::Worker, this, thread_start, thread_end, grid_size, std::ref(mtx));
    }

    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    Worker(start_step, end_step, grid_size, mtx);
  }

  if (world_.rank() == 0) {
    std::vector<double> temp_matrix(size_ * size_);
    for (int src = 1; src < world_.size(); ++src) {
      world_.recv(src, 2, temp_matrix);
      for (size_t i = 0; i < size_ * size_; ++i) {
        matrix_c_[i] += temp_matrix[i];
      }
    }
  } else {
    world_.send(0, 2, matrix_c_);
  }

  world_.barrier();

  return true;
}

bool filatev_v_foks_all::Focks::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto *temp = reinterpret_cast<double *>(task_data->outputs[0]);
    for (size_t i = 0; i < size_c_.m; ++i) {
      std::copy(matrix_c_.data() + (i * size_), matrix_c_.data() + (i * size_) + size_c_.n, temp + (i * size_c_.n));
    }
  }
  return true;
}
