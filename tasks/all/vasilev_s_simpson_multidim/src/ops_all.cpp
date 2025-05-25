#include "../include/ops_all.hpp"

#include <oneapi/tbb/parallel_reduce.h>
#include <oneapi/tbb/task_arena.h>
#include <tbb/tbb.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include "core/util/include/util.hpp"

bool vasilev_s_simpson_multidim::SimpsonTaskAll::ValidationImpl() {
  if (world_.rank() != 0) {
    return task_data->inputs.size() == 1;  // func ptr
  }

  const auto arity = task_data->inputs_count[0];

  const bool inputs_are_present = task_data->inputs.size() == 3 && arity > 0;
  const bool outputs_are_present = task_data->outputs.size() == 1 && task_data->outputs_count[0] == 1;
  if (!inputs_are_present || !outputs_are_present) {
    return false;
  }

  const auto* bounds = reinterpret_cast<Bound*>(task_data->inputs[0]);
  return std::all_of(bounds, bounds + arity, [](const auto& b) { return b.lo <= b.hi; });
}

void vasilev_s_simpson_multidim::SimpsonTaskAll::CalcSteps() {
  steps_.resize(arity_);
  std::ranges::transform(bounds_, steps_.begin(), [n = approxs_](const auto& b) { return (b.hi - b.lo) / n; });
}

bool vasilev_s_simpson_multidim::SimpsonTaskAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    arity_ = task_data->inputs_count[0];
    const auto* bsrc = reinterpret_cast<Bound*>(task_data->inputs[0]);
    bounds_.assign(bsrc, bsrc + arity_);

    func_ = reinterpret_cast<IntegrandFunction>(task_data->inputs[1]);
    approxs_ = *reinterpret_cast<std::size_t*>(task_data->inputs[2]);

    CalcSteps();
    scale_ = std::accumulate(steps_.begin(), steps_.end(), 1., [](double cur, double step) { return cur * step / 3.; });
  } else {
    func_ = reinterpret_cast<IntegrandFunction>(task_data->inputs[0]);
  }

  return true;
}

bool vasilev_s_simpson_multidim::SimpsonTaskAll::RunImpl() {
  boost::mpi::broadcast(world_, arity_, 0);
  boost::mpi::broadcast(world_, approxs_, 0);
  boost::mpi::broadcast(world_, bounds_, 0);

  if (world_.rank() != 0) {
    CalcSteps();
  }

  gridcap_ = static_cast<std::size_t>(std::pow(approxs_, arity_));
  const auto per = gridcap_ / world_.size();

  const auto begin = per * world_.rank();
  const auto iterlen = per + ((world_.rank() == (world_.size() - 1)) ? (gridcap_ % world_.size()) : 0);

  oneapi::tbb::task_arena arena(ppc::util::GetPPCNumThreads());
  double isum = arena.execute([&] {
    return oneapi::tbb::parallel_reduce(
        oneapi::tbb::blocked_range<std::size_t>(begin, begin + iterlen, iterlen / arena.max_concurrency()), 0.,
        [&](const tbb::blocked_range<std::size_t>& r, double threadsum) {
          std::vector<double> coordbuf(arity_);
          for (std::size_t ip = r.begin(); ip < r.end(); ip++) {
            auto p = ip;
            double coefficient = 1.;
            for (size_t k = 0; k < coordbuf.size(); k++) {
              const auto pos{p % approxs_};
              coordbuf[k] = bounds_[k].lo + (double(pos) * (bounds_[k].hi - bounds_[k].lo) / double(approxs_));
              p /= static_cast<decltype(p)>(approxs_);
              if (pos == 0 || pos == (approxs_ - 1)) {
                continue;
              }
              if (pos % 2 != 0) {
                coefficient *= 4.;
              } else {
                coefficient *= 2.;
              }
            }

            threadsum += coefficient * func_(coordbuf);
          }

          return threadsum;
        },
        std::plus<>());
  });

  double global_isum = 0.;
  boost::mpi::reduce(world_, isum, global_isum, std::plus{}, 0);

  if (world_.rank() == 0) {
    result_ = global_isum * scale_;
  }

  return true;
}

bool vasilev_s_simpson_multidim::SimpsonTaskAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  }
  return true;
}
