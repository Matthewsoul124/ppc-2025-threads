#include "all/vavilov_v_cannon/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/scatter.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/request.hpp>
#include <cmath>
#include <vector>

namespace mpi = boost::mpi;

int vavilov_v_cannon_all::CannonALL::FindOptimalGridSize(int size, int n) {
  int grid = std::floor(std::sqrt(size));
  while (grid > 0) {
    if (n % grid == 0) {
      break;
    }
    --grid;
  }
  return grid > 0 ? grid : 1;
}

void vavilov_v_cannon_all::CannonALL::TakeBlock(const std::vector<double>& matrix, double* block, int n, int k,
                                                int block_row, int block_col) {
#pragma omp parallel for
  for (int i = 0; i < k; ++i) {
    for (int j = 0; j < k; ++j) {
      block[(i * k) + j] = matrix[(((block_row * k) + i) * n) + ((block_col * k) + j)];
    }
  }
}

bool vavilov_v_cannon_all::CannonALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    N_ = static_cast<int>(std::sqrt(task_data->inputs_count[0]));
    num_blocks_ = static_cast<int>(std::sqrt(N_));
    block_size_ = N_ / num_blocks_;

    auto* a = reinterpret_cast<double*>(task_data->inputs[0]);
    auto* b = reinterpret_cast<double*>(task_data->inputs[1]);
    A_.assign(a, a + (N_ * N_));
    B_.assign(b, b + (N_ * N_));
    C_.assign(N_ * N_, 0);
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->inputs_count[1] &&
           task_data->outputs_count[0] == task_data->inputs_count[0];
  }
  return true;
}

void vavilov_v_cannon_all::CannonALL::InitialShift(std::vector<double>& local_a, std::vector<double>& local_b) {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  int row = rank / grid_size;
  int col = rank % grid_size;

  int send_rank_a = (row * grid_size) + ((col + grid_size - 1) % grid_size);
  int recv_rank_a = (row * grid_size) + ((col + 1) % grid_size);

  int send_rank_b = col + (grid_size * ((row + grid_size - 1) % grid_size));
  int recv_rank_b = col + (grid_size * ((row + 1) % grid_size));

  std::vector<double> tmp_a(block_size_ * block_size_);
  std::vector<double> tmp_b(block_size_ * block_size_);
  for (int i = 0; i < row; ++i) {
    world_.send(send_rank_a, 0, local_a.data(), block_size_ * block_size_);
    world_.recv(recv_rank_a, 0, tmp_a.data(), block_size_ * block_size_);
    std::swap(local_a, tmp_a);
  }

  for (int i = 0; i < col; ++i) {
    world_.send(send_rank_b, 1, local_b.data(), block_size_ * block_size_);
    world_.recv(recv_rank_b, 1, tmp_b.data(), block_size_ * block_size_);
    std::swap(local_b, tmp_b);
  }
}

void vavilov_v_cannon_all::CannonALL::ShiftBlocks(std::vector<double>& local_a, std::vector<double>& local_b) {
  int rank = world_.rank();
  int grid_size = num_blocks_;
  int row = rank / grid_size;
  int col = rank % grid_size;

  int send_rank_a = (row * grid_size) + ((col + grid_size - 1) % grid_size);
  int recv_rank_a = (row * grid_size) + ((col + 1) % grid_size);

  int send_rank_b = col + (grid_size * ((row + grid_size - 1) % grid_size));
  int recv_rank_b = col + (grid_size * ((row + 1) % grid_size));

  std::vector<double> tmp_a(block_size_ * block_size_);
  std::vector<double> tmp_b(block_size_ * block_size_);

  world_.send(send_rank_a, 2, local_a.data(), block_size_ * block_size_);
  world_.recv(recv_rank_a, 2, tmp_a.data(), block_size_ * block_size_);
  std::swap(local_a, tmp_a);

  world_.send(send_rank_b, 3, local_b.data(), block_size_ * block_size_);
  world_.recv(recv_rank_b, 3, tmp_b.data(), block_size_ * block_size_);
  std::swap(local_b, tmp_b);
}

void vavilov_v_cannon_all::CannonALL::BlockMultiply(const std::vector<double>& local_a,
                                                    const std::vector<double>& local_b, std::vector<double>& local_c) {
#pragma omp parallel for
  for (int i = 0; i < block_size_; ++i) {
    for (int j = 0; j < block_size_; ++j) {
      double temp = 0.0;
      for (int k = 0; k < block_size_; ++k) {
        temp += local_a[(i * block_size_) + k] * local_b[(k * block_size_) + j];
      }
      local_c[(i * block_size_) + j] += temp;
    }
  }
}

void vavilov_v_cannon_all::CannonALL::GatherResults(std::vector<double>& tmp_c, int block_size_sq) {
#pragma omp parallel for
  for (int block_row = 0; block_row < num_blocks_; ++block_row) {
    for (int block_col = 0; block_col < num_blocks_; ++block_col) {
      int block_rank = (block_row * num_blocks_) + block_col;
      int block_index = block_rank * block_size_sq;
      for (int i = 0; i < block_size_; ++i) {
        for (int j = 0; j < block_size_; ++j) {
          int global_row = (block_row * block_size_) + i;
          int global_col = (block_col * block_size_) + j;
          C_[(global_row * N_) + global_col] = tmp_c[block_index + (i * block_size_) + j];
        }
      }
    }
  }
}

void vavilov_v_cannon_all::CannonALL::PrepareScatterData(std::vector<double>& scatter_a, std::vector<double>& scatter_b,
                                                         int active_procs, int block_size_sq) {
  scatter_a.resize(active_procs * block_size_sq);
  scatter_b.resize(active_procs * block_size_sq);
  int index = 0;
  for (int block_row = 0; block_row < num_blocks_; ++block_row) {
    for (int block_col = 0; block_col < num_blocks_; ++block_col) {
      TakeBlock(A_, scatter_a.data() + index, N_, block_size_, block_row, block_col);
      TakeBlock(B_, scatter_b.data() + index, N_, block_size_, block_row, block_col);
      index += block_size_sq;
    }
  }
}

bool vavilov_v_cannon_all::CannonALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  mpi::broadcast(world_, N_, 0);

  num_blocks_ = FindOptimalGridSize(size, N_);
  block_size_ = N_ / num_blocks_;
  int block_size_sq = block_size_ * block_size_;

  int active_procs = num_blocks_ * num_blocks_;
  mpi::communicator active_world = world_.split(rank < active_procs ? 0 : MPI_UNDEFINED);
  if (rank >= active_procs) {
    return true;
  }

  rank = active_world.rank();

  std::vector<double> local_a(block_size_sq);
  std::vector<double> local_b(block_size_sq);
  std::vector<double> local_c(block_size_sq, 0);

  std::vector<double> scatter_a;
  std::vector<double> scatter_b;
  if (rank == 0) {
    PrepareScatterData(scatter_a, scatter_b, active_procs, block_size_sq);
  }

  mpi::scatter(active_world, scatter_a.data(), local_a.data(), block_size_sq, 0);
  mpi::scatter(active_world, scatter_b.data(), local_b.data(), block_size_sq, 0);

  InitialShift(local_a, local_b);

  BlockMultiply(local_a, local_b, local_c);
  for (int iter = 0; iter < num_blocks_ - 1; ++iter) {
    ShiftBlocks(local_a, local_b);
    BlockMultiply(local_a, local_b, local_c);
  }

  std::vector<double> tmp_c;
  if (rank == 0) {
    tmp_c.resize(active_procs * block_size_sq);
  }
  mpi::gather(active_world, local_c.data(), block_size_sq, tmp_c.data(), 0);
  if (rank == 0) {
    GatherResults(tmp_c, block_size_sq);
  }

  return true;
}

bool vavilov_v_cannon_all::CannonALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(C_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
