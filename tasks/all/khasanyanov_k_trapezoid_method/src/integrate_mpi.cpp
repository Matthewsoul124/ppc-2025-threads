#include "../include/integrate_mpi.hpp"

#include <algorithm>
#include <boost/serialization/utility.hpp>  // NOLINT(*-include-cleaner)
#include <boost/serialization/vector.hpp>   // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/reduce.hpp"
#include "core/task/include/task.hpp"

using namespace khasanyanov_k_trapezoid_method_all;

const int TrapezoidalMethodALL::kDefaultSteps = 10;
const int TrapezoidalMethodALL::kMaxSteps = 250;

void TrapezoidalMethodALL::CreateTaskData(std::shared_ptr<ppc::core::TaskData> &task_data, TaskContext &context,
                                          double *out) {
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(&context));
  task_data->inputs_count.emplace_back(context.bounds.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out));
  task_data->outputs_count.emplace_back(1);
}

bool TrapezoidalMethodALL::ValidationImpl() {
  if (comm_.rank() == 0) {
    auto *data = reinterpret_cast<TaskContext *>(task_data->inputs[0]);
    return data != nullptr && task_data->inputs_count[0] > 0 && task_data->outputs[0] != nullptr;
  }
  return true;
}

bool TrapezoidalMethodALL::PreProcessingImpl() {
  if (comm_.rank() == 0) {
    data_ = *reinterpret_cast<TaskContext *>(task_data->inputs[0]);
  }
  return true;
}
bool TrapezoidalMethodALL::RunImpl() {
  res_ = TrapezoidalMethodOmp(data_.bounds, 100);

  return true;
}
bool TrapezoidalMethodALL::PostProcessingImpl() {
  if (comm_.rank() == 0) {
    *reinterpret_cast<double *>(task_data->outputs[0]) = res_;
  }
  return true;
}

double TrapezoidalMethodALL::TrapezoidalMethodOmp(const IntegrationBounds &bounds, int steps) {
  const int rank = comm_.rank();
  const int size = comm_.size();

  IntegrationBounds local_bounds;
  if (rank == 0) {
    local_bounds = bounds;
  }

  boost::mpi::broadcast(comm_, local_bounds, 0);

  const size_t dimension = local_bounds.size();
  std::vector<double> h(dimension);
  double cell_volume = 1.0;
  int total_points = 1;

  for (size_t i = 0; i < dimension; ++i) {
    const auto &[a, b] = local_bounds[i];
    h[i] = (b - a) / steps;
    cell_volume *= h[i];
    total_points *= (steps + 1);
  }

  const int chunk_size = total_points / size;
  const int remainder = total_points % size;
  const int start_idx = (rank * chunk_size) + std::min(rank, remainder);
  const int end_idx = start_idx + chunk_size + (rank < remainder ? 1 : 0);

  double local_sum = 0.0;
#pragma omp parallel for reduction(+ : local_sum)
  for (int idx = start_idx; idx < end_idx; ++idx) {
    std::vector<double> point(dimension);
    int temp = idx;
    int boundary_count = 0;

    for (size_t dim = 0; dim < dimension; ++dim) {
      const int steps_per_dim = steps + 1;
      const int step = temp % steps_per_dim;
      temp /= steps_per_dim;

      const auto &[a, _] = local_bounds[dim];
      point[dim] = a + step * h[dim];

      if (step == 0 || step == steps) {
        boundary_count++;
      }
    }

    const double weight = std::pow(0.5, boundary_count);
    local_sum += function_(point) * weight;
  }

  double global_sum = 0.0;
  boost::mpi::reduce(comm_, local_sum, global_sum, std::plus<>(), 0);

  return (rank == 0) ? (global_sum * cell_volume) : 0.0;
}