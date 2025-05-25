#include "all/moiseev_a_mult_mat/include/ops_mpi.hpp"

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <cmath>
#include <vector>

bool moiseev_a_mult_mat_mpi::MultMatMPI::PreProcessingImpl() {
  unsigned input_size_a = task_data->inputs_count[0];
  unsigned input_size_b = task_data->inputs_count[1];
  auto* in_ptr_a = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* in_ptr_b = reinterpret_cast<double*>(task_data->inputs[1]);

  matrix_a_.assign(in_ptr_a, in_ptr_a + input_size_a);
  matrix_b_.assign(in_ptr_b, in_ptr_b + input_size_b);

  unsigned output_size = task_data->outputs_count[0];
  matrix_c_.assign(output_size, 0.0);

  matrix_size_ = static_cast<int>(std::sqrt(input_size_a));
  block_size_ = static_cast<int>(std::sqrt(matrix_size_));
  if (matrix_size_ % block_size_ != 0) {
    block_size_ = 1;
  }
  num_blocks_ = matrix_size_ / block_size_;
  return true;
}

bool moiseev_a_mult_mat_mpi::MultMatMPI::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool moiseev_a_mult_mat_mpi::MultMatMPI::RunImpl() {
  boost::mpi::communicator world;
  int world_size = world.size();
  int world_rank = world.rank();

  int p = 0;
  int active_procs = 0;
  int block = 0;
  DetermineGrid(world_size, p, active_procs, block);

  boost::mpi::communicator active = world.split(world_rank < active_procs ? 0 : 1, world_rank);
  bool is_active = (world_rank < active_procs);

  std::vector<double> a_block(block * block);
  std::vector<double> b_block(block * block);
  std::vector<double> c_block(block * block, 0.0);

  if (is_active) {
    DistributeMatrices(world, world_rank, active_procs, p, block, a_block, b_block);

    int my_row = world_rank / p;
    int my_col = world_rank % p;
    ComputeLocalBlock(active, my_row, my_col, p, block, a_block, b_block, c_block);

    GatherResult(world, world_rank, active_procs, p, block, c_block);
  }

  boost::mpi::broadcast(world, matrix_c_, 0);
  return true;
}

bool moiseev_a_mult_mat_mpi::MultMatMPI::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(matrix_c_, out_ptr);
  return true;
}

void moiseev_a_mult_mat_mpi::MultMatMPI::DetermineGrid(int world_size, int& p, int& active_procs, int& block) const {
  p = static_cast<int>(std::sqrt(world_size));
  while (p > 1 && (matrix_size_ % p != 0 || p * p > world_size)) {
    --p;
  }
  p = std::max(p, 1);
  active_procs = p * p;
  block = matrix_size_ / p;
}

void moiseev_a_mult_mat_mpi::MultMatMPI::DistributeMatrices(boost::mpi::communicator& world, int world_rank,
                                                            int active_procs, int p, int block,
                                                            std::vector<double>& a_block,
                                                            std::vector<double>& b_block) const {
  if (world_rank == 0) {
    for (int proc = 0; proc < active_procs; ++proc) {
      int row = proc / p;
      int col = proc % p;
      std::vector<double> tmp(block * block);
      for (int i = 0; i < block; ++i) {
        int src = ((row * block + i) * matrix_size_) + (col * block);
        std::copy_n(&matrix_a_[src], block, &tmp[i * block]);
      }
      if (proc == 0) {
        a_block = tmp;
      } else {
        world.send(proc, 0, tmp);
      }
    }
    for (int proc = 0; proc < active_procs; ++proc) {
      int row = proc / p;
      int col = proc % p;
      std::vector<double> tmp(block * block);
      for (int i = 0; i < block; ++i) {
        int src = ((row * block + i) * matrix_size_) + (col * block);
        std::copy_n(&matrix_b_[src], block, &tmp[i * block]);
      }
      if (proc == 0) {
        b_block = tmp;
      } else {
        world.send(proc, 1, tmp);
      }
    }
  } else {
    world.recv(0, 0, a_block);
    world.recv(0, 1, b_block);
  }
}

void moiseev_a_mult_mat_mpi::MultMatMPI::ComputeLocalBlock(boost::mpi::communicator& active, int my_row, int my_col,
                                                           int p, int block, std::vector<double>& a_block,
                                                           std::vector<double>& b_block, std::vector<double>& c_block) {
  auto orig_a = a_block;
  boost::mpi::communicator row_comm = active.split(my_row, my_col);

  for (int step = 0; step < p; ++step) {
    a_block = orig_a;
    int root = (my_row + step) % p;
    boost::mpi::broadcast(row_comm, a_block, root);

#pragma omp parallel for
    for (int i = 0; i < block; ++i) {
      for (int j = 0; j < block; ++j) {
        double sum = 0;
        for (int k = 0; k < block; ++k) {
          sum += a_block[(i * block) + k] * b_block[(k * block) + j];
        }
        c_block[(i * block) + j] += sum;
      }
    }

    int prev = (my_row - 1 + p) % p;
    int next = (my_row + 1) % p;
    active.sendrecv((prev * p) + my_col, 2, b_block, (next * p) + my_col, 2, b_block);
  }
}

void moiseev_a_mult_mat_mpi::MultMatMPI::GatherResult(boost::mpi::communicator& world, int world_rank, int active_procs,
                                                      int p, int block, const std::vector<double>& c_block) {
  if (world_rank == 0) {
    for (int i = 0; i < block; ++i) {
      int dst = ((i + 0 * block) * matrix_size_) + (0 * block);
      std::copy_n(&c_block[i * block], block, &matrix_c_[dst]);
    }
    for (int proc = 1; proc < active_procs; ++proc) {
      std::vector<double> tmp(block * block);
      world.recv(proc, 3, tmp);
      int row = proc / p;
      int col = proc % p;
      for (int i = 0; i < block; ++i) {
        int dst = ((i + row * block) * matrix_size_) + (col * block);
        std::copy_n(&tmp[i * block], block, &matrix_c_[dst]);
      }
    }
  } else {
    world.send(0, 3, c_block);
  }
}
