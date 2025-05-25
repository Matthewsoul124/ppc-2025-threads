#include "all/chernykh_a_multidimensional_integral_rectangle/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(misc-include-cleaner)
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <vector>

namespace chernykh_a_multidimensional_integral_rectangle_all {

double Dimension::GetLowerBound() const { return lower_bound_; }

double Dimension::GetUpperBound() const { return upper_bound_; }

int Dimension::GetStepsCount() const { return steps_count_; }

double Dimension::GetStepSize() const { return (upper_bound_ - lower_bound_) / steps_count_; }

bool Dimension::IsValid() const { return lower_bound_ < upper_bound_ && steps_count_ > 0; }

bool AllTask::ValidationImpl() {
  if (world_.rank() == 0) {
    auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
    uint32_t dims_size = task_data->inputs_count[0];
    return dims_size > 0 &&
           std::all_of(dims_ptr, dims_ptr + dims_size, [](const Dimension &dim) -> bool { return dim.IsValid(); });
  }
  return true;
}

bool AllTask::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto *dims_ptr = reinterpret_cast<Dimension *>(task_data->inputs[0]);
    uint32_t dims_size = task_data->inputs_count[0];
    dims_.assign(dims_ptr, dims_ptr + dims_size);
  }
  return true;
}

bool AllTask::RunImpl() {
  boost::mpi::broadcast(world_, dims_, 0);

  int total_points = GetTotalPoints();
  int points_per_process = total_points / world_.size();
  int extra_points = total_points % world_.size();
  int start_point = (world_.rank() * points_per_process) + std::min(world_.rank(), extra_points);
  int end_point = start_point + points_per_process + (world_.rank() < extra_points ? 1 : 0);

  double partial_sum = 0.0;
#pragma omp parallel
  {
    auto thread_point = Point(dims_.size());
#pragma omp for reduction(+ : partial_sum)
    for (int i = start_point; i < end_point; i++) {
      FillPoint(i, thread_point);
      partial_sum += func_(thread_point);
    }
  }

  boost::mpi::reduce(world_, partial_sum, result_, std::plus(), 0);

  if (world_.rank() == 0) {
    result_ *= GetScalingFactor();
  }
  return true;
}

bool AllTask::PostProcessingImpl() {
  if (world_.rank() == 0) {
    *reinterpret_cast<double *>(task_data->outputs[0]) = result_;
  }
  return true;
}

void AllTask::FillPoint(int index, Point &point) const {
  for (size_t i = 0; i < dims_.size(); i++) {
    int coordinate_index = index % dims_[i].GetStepsCount();
    point[i] = dims_[i].GetLowerBound() + (coordinate_index + 1) * dims_[i].GetStepSize();
    index /= dims_[i].GetStepsCount();
  }
}

int AllTask::GetTotalPoints() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1,
                         [](int accum, const Dimension &dim) -> int { return accum * dim.GetStepsCount(); });
}

double AllTask::GetScalingFactor() const {
  return std::accumulate(dims_.begin(), dims_.end(), 1.0,
                         [](double accum, const Dimension &dim) -> double { return accum * dim.GetStepSize(); });
}

}  // namespace chernykh_a_multidimensional_integral_rectangle_all
