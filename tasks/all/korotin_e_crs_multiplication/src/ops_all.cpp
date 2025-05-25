#include "all/korotin_e_crs_multiplication/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gather.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

using namespace std::chrono_literals;

bool korotin_e_crs_multiplication_all::CrsMultiplicationALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    A_N_ = task_data->inputs_count[0];
    auto *in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[0]);
    A_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + A_N_);

    A_Nz_ = task_data->inputs_count[1];
    in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[1]);
    A_col_ = std::vector<unsigned int>(in_ptr, in_ptr + A_Nz_);

    auto *val_ptr = reinterpret_cast<double *>(task_data->inputs[2]);
    A_val_ = std::vector<double>(val_ptr, val_ptr + A_Nz_);

    B_N_ = task_data->inputs_count[3];
    in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[3]);
    B_rI_ = std::vector<unsigned int>(in_ptr, in_ptr + B_N_);

    B_Nz_ = task_data->inputs_count[4];
    in_ptr = reinterpret_cast<unsigned int *>(task_data->inputs[4]);
    B_col_ = std::vector<unsigned int>(in_ptr, in_ptr + B_Nz_);

    val_ptr = reinterpret_cast<double *>(task_data->inputs[5]);
    B_val_ = std::vector<double>(val_ptr, val_ptr + B_Nz_);
    output_size_ = task_data->outputs_count[0];
  }

  return true;
}

bool korotin_e_crs_multiplication_all::CrsMultiplicationALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[1] == task_data->inputs_count[2] &&
           task_data->inputs_count[4] == task_data->inputs_count[5] &&
           task_data->inputs_count[0] == task_data->outputs_count[0] &&
           *std::max_element(reinterpret_cast<unsigned int *>(task_data->inputs[1]),
                             reinterpret_cast<unsigned int *>(task_data->inputs[1]) + task_data->inputs_count[1]) <=
               task_data->inputs_count[3] - 2;
  }
  return true;
}

void korotin_e_crs_multiplication_all::CrsMultiplicationALL::MulTask(size_t l, size_t r, std::vector<double> &local_val,
                                                                     std::vector<unsigned int> &local_col,
                                                                     std::vector<unsigned int> &temp_r_i,
                                                                     const std::vector<unsigned int> &tr_i,
                                                                     const std::vector<unsigned int> &tcol,
                                                                     const std::vector<double> &tval) {
  for (size_t k = l; k < r; ++k) {
    for (size_t s = 0; s < tr_i.size() - 1; ++s) {
      double sum = 0;
      unsigned int ai = A_rI_[k];
      unsigned int bt = tr_i[s];
      while (ai < A_rI_[k + 1] && bt < tr_i[s + 1]) {
        if (A_col_[ai] == tcol[bt]) {
          sum += A_val_[ai] * tval[bt];
          ai++;
          bt++;
        } else if (A_col_[ai] < tcol[bt]) {
          ai++;
        } else {
          bt++;
        }
      }
      if (sum != 0) {
        local_val.push_back(sum);
        local_col.push_back(s);
        temp_r_i[k + 1]++;
      }
    }
  }
}

void korotin_e_crs_multiplication_all::CrsMultiplicationALL::TrpB(std::vector<unsigned int> &tr_i,
                                                                  std::vector<unsigned int> &tcol,
                                                                  std::vector<double> &tval) {
  unsigned int k = 0;
  unsigned int s = 0;
  for (k = 0; k < B_Nz_; k++) {
    tr_i[B_col_[k] + 1]++;
  }

  for (k = 1; k < tr_i.size(); k++) {
    tr_i[k] += tr_i[k - 1];
  }

  for (k = 0; k < B_N_ - 1; k++) {
    for (s = B_rI_[k]; s < B_rI_[k + 1]; s++) {
      tval[tr_i[B_col_[s]]] = B_val_[s];
      tcol[tr_i[B_col_[s]]] = k;
      tr_i[B_col_[s]]++;
    }
  }

  for (k = tr_i.size() - 1; k > 0; k--) {
    tr_i[k] = tr_i[k - 1];
  }
  tr_i[0] = 0;
}

bool korotin_e_crs_multiplication_all::CrsMultiplicationALL::RunImpl() {
  unsigned int i = 0;
  unsigned int tr_i_sz = 0;

  if (world_.rank() == 0) {
    tr_i_sz = *std::ranges::max_element(B_col_.begin(), B_col_.end());
  }

  boost::mpi::broadcast(world_, tr_i_sz, 0);
  boost::mpi::broadcast(world_, A_N_, 0);
  boost::mpi::broadcast(world_, A_Nz_, 0);
  boost::mpi::broadcast(world_, B_Nz_, 0);
  boost::mpi::broadcast(world_, output_size_, 0);

  std::vector<unsigned int> tr_i(tr_i_sz + 2, 0);
  std::vector<unsigned int> tcol(B_Nz_, 0);
  std::vector<double> tval(B_Nz_, 0);

  if (world_.rank() == 0) {
    CrsMultiplicationALL::TrpB(tr_i, tcol, tval);
  } else {
    A_rI_ = std::vector<unsigned int>(A_N_);
    A_col_ = std::vector<unsigned int>(A_Nz_);
    A_val_ = std::vector<double>(A_Nz_);
  }

  boost::mpi::broadcast(world_, A_rI_.data(), static_cast<int>(A_rI_.size()), 0);
  boost::mpi::broadcast(world_, A_col_.data(), static_cast<int>(A_col_.size()), 0);
  boost::mpi::broadcast(world_, A_val_.data(), static_cast<int>(A_val_.size()), 0);
  boost::mpi::broadcast(world_, tr_i.data(), static_cast<int>(tr_i.size()), 0);
  boost::mpi::broadcast(world_, tcol.data(), static_cast<int>(tcol.size()), 0);
  boost::mpi::broadcast(world_, tval.data(), static_cast<int>(tval.size()), 0);

  unsigned int local_a_n = (A_N_ - 1) / world_.size();
  size_t start = 0;
  if (world_.rank() < static_cast<int>((A_N_ - 1) % world_.size())) {
    local_a_n++;
    start = local_a_n * world_.rank();
  } else {
    start = (local_a_n * world_.rank()) + ((A_N_ - 1) % world_.size());
  }

  output_rI_ = std::vector<unsigned int>(output_size_, 0);
  output_col_.clear();
  output_val_.clear();
  unsigned int magic_const = ppc::util::GetPPCNumThreads();
  std::vector<std::vector<double>> local_val(magic_const);
  std::vector<std::vector<unsigned int>> local_col(magic_const);
  std::vector<unsigned int> temp_r_i(A_N_, 0);
  std::vector<std::thread> threads;

  std::vector<size_t> delta(magic_const, local_a_n / magic_const);
  for (i = 0; i < local_a_n % magic_const; ++i) {
    delta[i]++;
  }
  delta[0] += start;
  for (i = 1; i < magic_const; ++i) {
    delta[i] += delta[i - 1];
  }

  threads.emplace_back(&CrsMultiplicationALL::MulTask, this, start, delta[0], std::ref(local_val[0]),
                       std::ref(local_col[0]), std::ref(temp_r_i), std::ref(tr_i), std::ref(tcol), std::ref(tval));
  for (i = 1; i < magic_const; ++i) {
    threads.emplace_back(&CrsMultiplicationALL::MulTask, this, delta[i - 1], delta[i], std::ref(local_val[i]),
                         std::ref(local_col[i]), std::ref(temp_r_i), std::ref(tr_i), std::ref(tcol), std::ref(tval));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  for (unsigned int t = 0; t < magic_const; ++t) {
    output_val_.insert(output_val_.end(), local_val[t].begin(), local_val[t].end());
    output_col_.insert(output_col_.end(), local_col[t].begin(), local_col[t].end());
  }

  std::vector<std::vector<double>> gathered_val;
  std::vector<std::vector<unsigned int>> gathered_col;
  std::vector<unsigned int> temp_r_i_all(A_N_, 0);
  boost::mpi::gather(world_, output_col_, gathered_col, 0);
  boost::mpi::gather(world_, output_val_, gathered_val, 0);
  boost::mpi::reduce(world_, temp_r_i, temp_r_i_all, std::plus<>(), 0);

  if (world_.rank() == 0) {
    output_val_.clear();
    output_col_.clear();
    for (int t = 0; t < world_.size(); ++t) {
      output_val_.insert(output_val_.end(), gathered_val[t].begin(), gathered_val[t].end());
      output_col_.insert(output_col_.end(), gathered_col[t].begin(), gathered_col[t].end());
    }

    for (i = 1; i < A_N_; ++i) {
      output_rI_[i] += output_rI_[i - 1] + temp_r_i_all[i];
    }
  }

  return true;
}

bool korotin_e_crs_multiplication_all::CrsMultiplicationALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_rI_.size(); i++) {
      reinterpret_cast<unsigned int *>(task_data->outputs[0])[i] = output_rI_[i];
    }
    for (size_t i = 0; i < output_col_.size(); i++) {
      reinterpret_cast<unsigned int *>(task_data->outputs[1])[i] = output_col_[i];
      reinterpret_cast<double *>(task_data->outputs[2])[i] = output_val_[i];
    }
    task_data->outputs_count.emplace_back(output_col_.size());
    task_data->outputs_count.emplace_back(output_val_.size());
  }
  return true;
}
