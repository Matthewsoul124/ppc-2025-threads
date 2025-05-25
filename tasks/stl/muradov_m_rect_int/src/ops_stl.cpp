#include "stl/muradov_m_rect_int/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

bool muradov_m_rect_int_stl::RectIntTaskSTLPar::ValidationImpl() {
  return task_data->inputs_count[0] == 1 && task_data->inputs_count[1] > 0 && task_data->outputs_count[0] == 1;
}

bool muradov_m_rect_int_stl::RectIntTaskSTLPar::PreProcessingImpl() {
  auto* p_grains = reinterpret_cast<int*>(task_data->inputs[0]);
  grains_ = *p_grains;

  auto* p_bounds = reinterpret_cast<std::pair<double, double>*>(task_data->inputs[1]);
  bounds_.assign(p_bounds, p_bounds + task_data->inputs_count[1]);

  return true;
}

bool muradov_m_rect_int_stl::RectIntTaskSTLPar::RunImpl() {
  res_ = 0;

  const auto dims = static_cast<std::size_t>(bounds_.size());

  double hh = 1.0;
  for (std::size_t i = 0; i < dims; i++) {
    hh *= (bounds_[i].second - bounds_[i].first) / grains_;
  }
  int pts = static_cast<int>(std::pow(grains_, dims));

  int threads = ppc::util::GetPPCNumThreads();
  std::vector<int> jobs(threads, pts / threads);
  if (threads > 0) {
    jobs[0] += pts % threads;
  }

  std::vector<double> partial_sums(threads, 0.0);
  std::vector<std::thread> ts(threads);
  int left = 0;
  for (int id = 0; id < threads; id++) {
    ts[id] = std::thread(
        [&](int start_pt, int end_pt, double& partial_sum) {  // NOLINT(bugprone-easily-swappable-parameters)
          double sum = 0.0;
          FunArgs args(dims);
          for (int i = start_pt; i < end_pt; i++) {
            auto j = i;
            for (size_t k = 0; k < dims; k++) {
              args[k] = bounds_[k].first;
              args[k] += (j % grains_) * (bounds_[k].second - bounds_[k].first) / grains_;
              j /= grains_;
            }
            sum += fun_(args);
          }
          partial_sum = sum;
        },
        left, left + jobs[id], std::ref(partial_sums[id]));
    left += jobs[id];
  }

  double total_sum = 0.0;
  for (int i = 0; i < threads; i++) {
    ts[i].join();
    total_sum += partial_sums[i];
  }
  res_ = total_sum * hh;

  return true;
}

bool muradov_m_rect_int_stl::RectIntTaskSTLPar::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = res_;
  return true;
}
