#include "seq/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_seq.hpp"

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq {

bool TestTaskSequential::PreProcessingImpl() {
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

bool TestTaskSequential::ValidationImpl() {
  return (task_data->inputs_count[0] == task_data->inputs_count[1]) &&
         (task_data->inputs_count[0] / sizeof(double) == task_data->inputs_count[2] / sizeof(int));
}

namespace {

double TrapezoidalIntegration(const std::function<double(const std::vector<double>&)>& func,
                              const std::vector<double>& lower, const std::vector<double>& upper,
                              const std::vector<int>& steps, size_t current_dim, std::vector<double> point) {
  if (current_dim == lower.size()) {
    return func(point);
  }

  double h = (upper[current_dim] - lower[current_dim]) / steps[current_dim];
  double sum = 0.0;

  point.push_back(0.0);
  for (int i = 0; i <= steps[current_dim]; ++i) {
    point[current_dim] = lower[current_dim] + i * h;
    double weight = (i == 0 || i == steps[current_dim]) ? 0.5 : 1.0;
    sum += weight * TrapezoidalIntegration(func, lower, upper, steps, current_dim + 1, point);
  }

  return sum * h;
}

}  // namespace

bool TestTaskSequential::RunImpl() {
  if (!function_) {
    return false;
  }
  std::vector<double> point;
  result_ = TrapezoidalIntegration(function_, lower_limits_, upper_limits_, steps_, 0, point);
  return true;
}

bool TestTaskSequential::PostProcessingImpl() {
  *reinterpret_cast<double*>(task_data->outputs[0]) = result_;
  return true;
}

}  // namespace prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq