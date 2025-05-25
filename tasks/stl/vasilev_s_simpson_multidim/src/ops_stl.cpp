#include "../include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <future>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool vasilev_s_simpson_multidim::SimpsonTaskStl::ValidationImpl() {
  const auto arity = task_data->inputs_count[0];

  const bool inputs_are_present = task_data->inputs.size() == 3 && arity > 0;
  const bool outputs_are_present = task_data->outputs.size() == 1 && task_data->outputs_count[0] == 1;
  if (!inputs_are_present || !outputs_are_present) {
    return false;
  }

  const auto* bounds = reinterpret_cast<Bound*>(task_data->inputs[0]);
  return std::all_of(bounds, bounds + arity, [](const auto& b) { return b.lo <= b.hi; });
}

bool vasilev_s_simpson_multidim::SimpsonTaskStl::PreProcessingImpl() {
  arity_ = task_data->inputs_count[0];
  const auto* bsrc = reinterpret_cast<Bound*>(task_data->inputs[0]);
  bounds_.assign(bsrc, bsrc + arity_);

  func_ = reinterpret_cast<IntegrandFunction>(task_data->inputs[1]);
  approxs_ = *reinterpret_cast<std::size_t*>(task_data->inputs[2]);

  steps_.resize(arity_);
  std::ranges::transform(bounds_, steps_.begin(), [n = approxs_](const auto& b) { return (b.hi - b.lo) / n; });

  gridcap_ = static_cast<std::size_t>(std::pow(approxs_, arity_));
  scale_ = std::accumulate(steps_.begin(), steps_.end(), 1., [](double cur, double step) { return cur * step / 3.; });

  return true;
}

void vasilev_s_simpson_multidim::SimpsonTaskStl::ComputeThreadSum(std::pair<std::size_t, std::size_t> range,
                                                                  std::promise<double>&& promise) {
  double threadsum = 0.;
  std::vector<double> coordbuf(arity_);
  for (std::size_t ip = range.first; ip < range.second; ip++) {
    auto p = ip;
    double coefficient = 1.;
    for (size_t k = 0; k < coordbuf.size(); k++) {
      const auto pos = p % approxs_;
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
  promise.set_value(threadsum);
}

bool vasilev_s_simpson_multidim::SimpsonTaskStl::RunImpl() {
  const auto ws = static_cast<std::size_t>(ppc::util::GetPPCNumThreads());
  std::vector<std::future<double>> futures(ws);
  std::vector<std::thread> threads(ws);

  const auto per = gridcap_ / ws;
  const auto extra = gridcap_ % ws;

  std::size_t iptr = 0;
  for (std::size_t i = 0; i < ws; i++) {
    const auto leap = per + ((i < extra) ? 1 : 0);

    std::promise<double> promise;
    futures[i] = promise.get_future();
    threads[i] =
        std::thread(&SimpsonTaskStl::ComputeThreadSum, this, std::make_pair(iptr, iptr + leap), std::move(promise));
    iptr += leap;
  }

  double isum = 0.0;
  for (std::size_t i = 0; i < ws; i++) {
    threads[i].join();
    isum += futures[i].get();
  }

  result_ = isum * scale_;

  return true;
}

bool vasilev_s_simpson_multidim::SimpsonTaskStl::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}
