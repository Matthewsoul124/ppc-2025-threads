#include "stl/solovev_a_ccs_mmult_sparse/include/ccs_mmult_sparse.hpp"

#include <complex>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

void solovev_a_matrix_stl::SeqMatMultCcs::ProcessPhase1(solovev_a_matrix_stl::SeqMatMultCcs* self, int col,
                                                        std::vector<int>& available) {
  for (int i = self->M2_->col_p[col]; i < self->M2_->col_p[col + 1]; ++i) {
    int r = self->M2_->row[i];
    if (r < 0 || r >= self->M1_->c_n) {
      continue;
    }
    for (int j = self->M1_->col_p[r]; j < self->M1_->col_p[r + 1]; ++j) {
      int rr = self->M1_->row[j];
      if (rr >= 0 && rr < self->r_n_) {
        available[rr] = 1;
      }
    }
  }
  self->counts_[col] = std::accumulate(available.begin(), available.end(), 0);
}

void solovev_a_matrix_stl::SeqMatMultCcs::ProcessPhase2(solovev_a_matrix_stl::SeqMatMultCcs* self, int col,
                                                        std::vector<int>& available,
                                                        std::vector<std::complex<double>>& cask) {
  cask.assign(self->r_n_, {0.0, 0.0});
  for (int i = self->M2_->col_p[col]; i < self->M2_->col_p[col + 1]; ++i) {
    int r = self->M2_->row[i];
    if (r < 0 || r >= self->M1_->c_n) {
      continue;
    }
    auto v2 = self->M2_->val[i];
    for (int j = self->M1_->col_p[r]; j < self->M1_->col_p[r + 1]; ++j) {
      int rr = self->M1_->row[j];
      if (rr >= 0 && rr < self->r_n_) {
        cask[rr] += self->M1_->val[j] * v2;
        available[rr] = 1;
      }
    }
  }
  int pos = self->M3_->col_p[col];
  for (int rr = 0; rr < self->r_n_; ++rr) {
    if (available[rr] != 0) {
      self->M3_->row[pos] = rr;
      self->M3_->val[pos++] = cask[rr];
    }
  }
}

void solovev_a_matrix_stl::SeqMatMultCcs::NotifyCompletion(solovev_a_matrix_stl::SeqMatMultCcs* self) {
  int done = self->completed_.fetch_add(1) + 1;
  if (done == self->c_n_) {
    std::lock_guard<std::mutex> lk(self->mtx_);
    self->cv_done_.notify_all();
  }
}

void solovev_a_matrix_stl::SeqMatMultCcs::WorkerLoop(solovev_a_matrix_stl::SeqMatMultCcs* self) {
  static thread_local std::vector<int> available;
  static thread_local std::vector<std::complex<double>> cask;
  while (true) {
    int col = self->next_col_.fetch_add(1);
    if (col >= self->c_n_) {
      break;
    }
    available.assign(self->r_n_, 0);
    if (self->phase_ == 1) {
      ProcessPhase1(self, col, available);
    } else if (self->phase_ == 2) {
      ProcessPhase2(self, col, available, cask);
    }
    NotifyCompletion(self);
  }
}

bool solovev_a_matrix_stl::SeqMatMultCcs::PreProcessingImpl() {
  M1_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
  M2_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
  M3_ = reinterpret_cast<MatrixInCcsSparse*>(task_data->outputs[0]);
  return true;
}

bool solovev_a_matrix_stl::SeqMatMultCcs::ValidationImpl() {
  auto* m1 = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[0]);
  auto* m2 = reinterpret_cast<MatrixInCcsSparse*>(task_data->inputs[1]);
  return (m1->c_n == m2->r_n);
}

bool solovev_a_matrix_stl::SeqMatMultCcs::RunImpl() {
  if (M1_ == nullptr || M2_ == nullptr || M3_ == nullptr) {
    return false;
  }

  r_n_ = M1_->r_n;
  c_n_ = M2_->c_n;
  M3_->r_n = r_n_;
  M3_->c_n = c_n_;

  counts_.assign(c_n_, 0);
  M3_->col_p.assign(c_n_ + 1, 0);

  unsigned num_threads = ppc::util::GetPPCNumThreads();
  next_col_.store(0);
  completed_.store(0);
  phase_ = 1;
  workers_.clear();
  for (unsigned i = 0; i < num_threads; ++i) {
    workers_.emplace_back(WorkerLoop, this);
  }

  {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [&]() { return completed_.load() >= c_n_; });
  }

  for (auto& th : workers_) {
    if (th.joinable()) {
      th.join();
    }
  }
  for (int i = 0; i < c_n_; ++i) {
    M3_->col_p[i + 1] = M3_->col_p[i] + counts_[i];
  }

  int total = M3_->col_p[c_n_];
  if (total < 0) {
    return false;
  }
  M3_->n_z = total;
  M3_->row.resize(total);
  M3_->val.resize(total);

  next_col_.store(0);
  completed_.store(0);
  phase_ = 2;
  workers_.clear();
  for (unsigned i = 0; i < num_threads; ++i) {
    workers_.emplace_back(WorkerLoop, this);
  }

  {
    std::unique_lock<std::mutex> lk(mtx_);
    cv_done_.wait(lk, [&]() { return completed_.load() >= c_n_; });
  }

  for (auto& th : workers_) {
    if (th.joinable()) {
      th.join();
    }
  }
  workers_.clear();

  return true;
}

bool solovev_a_matrix_stl::SeqMatMultCcs::PostProcessingImpl() { return true; }
