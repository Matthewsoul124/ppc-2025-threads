#include "stl/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl {

bool TestTaskSTL::PreProcessingImpl() {
  auto* lower_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  auto* upper_ptr = reinterpret_cast<double*>(task_data->inputs[1]);
  auto* steps_ptr = reinterpret_cast<int*>(task_data->inputs[2]);

  dimensions_ = static_cast<int>(task_data->inputs_count[0] / sizeof(double));
  if (dimensions_ == 0) {
    return false;
  }

  lower_limits_ = std::vector<double>(lower_ptr, lower_ptr + dimensions_);
  upper_limits_ = std::vector<double>(upper_ptr, upper_ptr + dimensions_);
  steps_ = std::vector<int>(steps_ptr, steps_ptr + dimensions_);

  result_ = 0.0;
  return true;
}

bool TestTaskSTL::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] / sizeof(double) == task_data->inputs_count[2] / sizeof(int));
}

namespace {

double TrapezoidalIntegration(const std::function<double(const std::vector<double>&)>& func,
                              const std::vector<double>& lower, const std::vector<double>& upper,
                              const std::vector<int>& steps, size_t current_dim, const std::vector<double>& point);

inline double CalculateWeight(int i, int steps_dim) {
  return (static_cast<unsigned>(i - 1) >= static_cast<unsigned>(steps_dim - 1)) ? 0.5 : 1.0;
}

double SequentialIntegration(const std::function<double(const std::vector<double>&)>& func,
                             const std::vector<double>& lower, const std::vector<double>& upper,
                             const std::vector<int>& steps, size_t current_dim, std::vector<double>& point, double h) {
  double sum = 0.0;
  for (int i = 0; i <= steps[current_dim]; ++i) {
    point[current_dim] = lower[current_dim] + i * h;
    double weight = CalculateWeight(i, steps[current_dim]);
    sum += weight * TrapezoidalIntegration(func, lower, upper, steps, current_dim + 1, point);
  }
  return sum * h;
}

void ThreadIntegration(const std::function<double(const std::vector<double>&)>& func, const std::vector<double>& lower,
                       const std::vector<double>& upper, const std::vector<int>& steps, size_t current_dim,
                       const std::vector<double>& new_point, double h, int start, int end,
                       std::vector<double>& partial_sums) {
  auto local_point = new_point;
  for (int i = start; i < end; ++i) {
    local_point[current_dim] = lower[current_dim] + i * h;
    double weight = CalculateWeight(i, steps[current_dim]);
    partial_sums[i] = weight * TrapezoidalIntegration(func, lower, upper, steps, current_dim + 1, local_point);
  }
}

double ParallelIntegration(const std::function<double(const std::vector<double>&)>& func,
                           const std::vector<double>& lower, const std::vector<double>& upper,
                           const std::vector<int>& steps, size_t current_dim, const std::vector<double>& new_point,
                           double h) {
  std::vector<double> partial_sums(steps[current_dim] + 1, 0.0);
  const int num_threads = std::min(ppc::util::GetPPCNumThreads(), 4);
  std::vector<std::thread> threads(num_threads);
  int chunk_size = (steps[current_dim] + 1) / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    int start = t * chunk_size;
    int end = (t == num_threads - 1) ? steps[current_dim] + 1 : (t + 1) * chunk_size;

    threads[t] = std::thread(ThreadIntegration, std::ref(func), std::ref(lower), std::ref(upper), std::ref(steps),
                             current_dim, std::ref(new_point), h, start, end, std::ref(partial_sums));
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  return std::accumulate(partial_sums.begin(), partial_sums.end(), 0.0) * h;
}

double TrapezoidalIntegration(const std::function<double(const std::vector<double>&)>& func,
                              const std::vector<double>& lower, const std::vector<double>& upper,
                              const std::vector<int>& steps, size_t current_dim, const std::vector<double>& point) {
  if (current_dim == lower.size()) {
    return func(point);
  }

  if (steps[current_dim] <= 0) {
    return 0.0;
  }

  double h = (upper[current_dim] - lower[current_dim]) / steps[current_dim];
  if (std::isnan(h) || std::isinf(h)) {
    return 0.0;
  }

  std::vector<double> new_point = point;
  new_point.push_back(0.0);

  if (current_dim >= lower.size() - 1 || steps[current_dim] < 1000) {
    return SequentialIntegration(func, lower, upper, steps, current_dim, new_point, h);
  }

  return ParallelIntegration(func, lower, upper, steps, current_dim, new_point, h);
}

}  // namespace

bool TestTaskSTL::RunImpl() {
  if (!function_) {
    return false;
  }
  std::vector<double> point;
  result_ = TrapezoidalIntegration(function_, lower_limits_, upper_limits_, steps_, 0, point);
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  if (!task_data->outputs.empty() && task_data->outputs[0] != nullptr) {
    *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  }
  return true;
}

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl