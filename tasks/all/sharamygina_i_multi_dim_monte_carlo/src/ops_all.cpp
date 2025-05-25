#include "all/sharamygina_i_multi_dim_monte_carlo/include/ops_all.h"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cstdlib>
#include <ctime>
#include <functional>
#include <random>
#include <vector>

bool sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask::PreProcessingImpl() {
  int rank = world_.rank();

  if (rank == 0) {
    auto* raw_bounds = reinterpret_cast<double*>(task_data->inputs[0]);
    size_t total_bounds_count = task_data->inputs_count[0];
    boundaries_.resize(total_bounds_count);
    std::copy(raw_bounds, raw_bounds + total_bounds_count, boundaries_.begin());

    int* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    number_of_iterations_ = *iter_ptr;
  }

  boost::mpi::broadcast(world_, boundaries_, 0);
  boost::mpi::broadcast(world_, number_of_iterations_, 0);

  return true;
}

bool sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data && !task_data->outputs_count.empty() && !task_data->outputs.empty() &&
           !task_data->inputs.empty() && task_data->outputs_count[0] == 1 && (task_data->inputs_count.size() == 2) &&
           (task_data->inputs_count[0] % 2 == 0) && (task_data->inputs_count[1] == 1);
  }
  return true;
}

bool sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  size_t dimension = boundaries_.size() / 2;

  std::mt19937 engine(static_cast<unsigned long>(std::time(nullptr)) + rank + size);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  int iterations_per_process = number_of_iterations_ / size;
  int remainder = number_of_iterations_ % size;

  int start_iter = (rank * iterations_per_process) + std::min(rank, remainder);
  int end_iter = start_iter + iterations_per_process + (rank < remainder ? 1 : 0);

  double accumulator = 0.0;

  std::vector<double> random_point(dimension);

#pragma omp parallel for reduction(+ : accumulator)
  for (int n = start_iter; n < end_iter; ++n) {
    for (size_t i = 0; i < dimension; ++i) {
      double low = boundaries_[2 * i];
      double high = boundaries_[(2 * i) + 1];
      double t = distribution(engine);
      random_point[i] = low + (high - low) * t;
    }
    accumulator += integrating_function_(random_point);
  }

  double global_accumulator = 0.0;
  boost::mpi::reduce(world_, accumulator, global_accumulator, std::plus<>(), 0);

  if (rank == 0) {
    double volume = 1.0;
    for (size_t i = 0; i < dimension; ++i) {
      double edge_length = boundaries_[(2 * i) + 1] - boundaries_[2 * i];
      volume *= edge_length;
    }
    final_result_ = (global_accumulator / static_cast<double>(number_of_iterations_)) * volume;
  }

  return true;
}

bool sharamygina_i_multi_dim_monte_carlo_all::MultiDimMonteCarloTask::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    output_ptr[0] = final_result_;
  }
  return true;
}