#include "all/filateva_e_simpson/include/ops_all.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>  // IWYU pragma: keep
#include <cmath>
#include <core/util/include/util.hpp>
#include <cstddef>
#include <functional>
#include <vector>

bool filateva_e_simpson_all::Simpson::PreProcessingImpl() {
  if (world_.rank() == 0) {
    mer_ = task_data->inputs_count[0];
    steps_ = task_data->inputs_count[1];

    auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
    a_.insert(a_.end(), temp_a, temp_a + mer_);

    auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);
    b_.insert(b_.end(), temp_b, temp_b + mer_);
  }
  return true;
}

bool filateva_e_simpson_all::Simpson::ValidationImpl() {
  bool valid = true;
  if (world_.rank() == 0) {
    size_t mer = task_data->inputs_count[0];
    auto *temp_a = reinterpret_cast<double *>(task_data->inputs[0]);
    auto *temp_b = reinterpret_cast<double *>(task_data->inputs[1]);
    if (task_data->inputs_count[1] % 2 == 1) {
      valid = false;
    }
    for (size_t i = 0; i < mer; i++) {
      if (temp_b[i] <= temp_a[i]) {
        valid = false;
        break;
      }
    }
  }
  boost::mpi::broadcast(world_, valid, 0);
  return valid;
}

double filateva_e_simpson_all::Simpson::IntegralFunc(long start, long end) {
  double res = 0.0;
  const int num_threads = ppc::util::GetPPCNumThreads();

  oneapi::tbb::task_arena arena(num_threads);

  arena.execute([&] {
    return res = oneapi::tbb::parallel_reduce(
               tbb::blocked_range<long>(start, end), 0.0,
               [&](const tbb::blocked_range<long> &r, double local_res) {
                 for (long i = r.begin(); i != r.end(); ++i) {
                   unsigned long temp = i;
                   std::vector<double> param(mer_);
                   double weight = 1.0;
                   for (size_t m = 0; m < mer_; m++) {
                     size_t shag_i = temp % (steps_ + 1);
                     temp /= (steps_ + 1);
                     param[m] = a_[m] + h_[m] * static_cast<double>(shag_i);
                     if (shag_i == 0 || shag_i == steps_) {
                       continue;
                     }
                     weight *= (2.0 + static_cast<double>(shag_i % 2) * 2);
                   }
                   local_res += weight * f_(param);
                 }
                 return local_res;
               },
               [](double a, double b) { return a + b; });
  });

  return res;
}

void filateva_e_simpson_all::Simpson::SetFunc(Func f) { f_ = f; }

bool filateva_e_simpson_all::Simpson::RunImpl() {
  boost::mpi::broadcast(world_, mer_, 0);
  boost::mpi::broadcast(world_, steps_, 0);
  boost::mpi::broadcast(world_, a_, 0);
  const int num_proc = world_.size();
  long del = (long)std::pow(steps_ + 1, mer_) / num_proc;
  long ost = num_proc - ((long)std::pow(steps_ + 1, mer_) % num_proc);

  if (world_.rank() == 0) {
    h_.resize(mer_);
    for (size_t i = 0; i < mer_; i++) {
      h_[i] = static_cast<double>(b_[i] - a_[i]) / static_cast<double>(steps_);
    }
    res_ = 0.0;
  }
  boost::mpi::broadcast(world_, h_, 0);

  long start = (world_.rank() < ost) ? del * world_.rank() : (del * ost) + ((del + 1) * (world_.rank() - ost));
  long end = start + del + (world_.rank() < ost ? 0 : 1);
  double local_res = IntegralFunc(start, end);

  reduce(world_, local_res, res_, std::plus<>(), 0);

  if (world_.rank() == 0) {
    for (size_t i = 0; i < mer_; i++) {
      res_ *= (h_[i] / 3.0);
    }
  }
  return true;
}

bool filateva_e_simpson_all::Simpson::PostProcessingImpl() {
  if (world_.rank() == 0) {
    reinterpret_cast<double *>(task_data->outputs[0])[0] = res_;
  }
  return true;
}
