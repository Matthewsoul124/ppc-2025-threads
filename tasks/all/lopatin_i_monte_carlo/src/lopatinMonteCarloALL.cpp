#include "all/lopatin_i_monte_carlo/include/lopatinMonteCarloALL.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/reduce.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <cmath>
#include <cstddef>
#include <ctime>
#include <functional>
#include <random>
#include <vector>

namespace lopatin_i_monte_carlo_all {

bool TestTaskAll::ValidationImpl() {
  if (world_.rank() == 0) {
    const bool outputs_valid = !task_data->outputs_count.empty() && task_data->outputs_count[0] == 1;
    const bool inputs_valid = task_data->inputs_count.size() == 2 &&
                              (task_data->inputs_count[0] % 2 == 0) &&  // odd num of bounds
                              task_data->inputs_count[1] == 1;          // iterations num

    auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    const int iterations = *iter_ptr;
    const bool iter_valid = iterations > 0;
    return outputs_valid && inputs_valid && iter_valid;
  }
  return true;
}

bool TestTaskAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto* bounds_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    size_t bounds_size = task_data->inputs_count[0];
    integrationBounds_.resize(bounds_size);
    std::copy(bounds_ptr, bounds_ptr + bounds_size, integrationBounds_.begin());

    auto* iter_ptr = reinterpret_cast<int*>(task_data->inputs[1]);
    iterations_ = *iter_ptr;
  }
  return true;
}

bool TestTaskAll::RunImpl() {
  boost::mpi::broadcast(world_, integrationBounds_, 0);
  boost::mpi::broadcast(world_, iterations_, 0);

  const size_t d = integrationBounds_.size() / 2;  // dimensions

  // integration volume
  double volume = 1.0;
  for (size_t j = 0; j < d; ++j) {
    volume *= (integrationBounds_[(2 * j) + 1] - integrationBounds_[2 * j]);
  }

  // distributing iterations
  const int world_size = world_.size();
  const int world_rank = world_.rank();
  const int local_iterations = (iterations_ / world_size) + (world_rank < (iterations_ % world_size) ? 1 : 0);

  double local_sum = 0.0;
#pragma omp parallel reduction(+ : local_sum)
  {
    // init random numbers generator
    std::random_device rd;
    std::seed_seq seed{rd(), static_cast<unsigned>(std::time(nullptr)), static_cast<unsigned>(world_rank),
                       static_cast<unsigned>(omp_get_thread_num())};
    std::mt19937 local_rnd(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);

#pragma omp for
    for (int i = 0; i < local_iterations; ++i) {
      std::vector<double> point(d);
      for (size_t j = 0; j < d; ++j) {
        const double min = integrationBounds_[2 * j];
        const double max = integrationBounds_[(2 * j) + 1];
        point[j] = min + (max - min) * dis(local_rnd);
      }
      local_sum += integrand_(point);
    }
  }

  double global_sum = 0.0;
  boost::mpi::reduce(world_, local_sum, global_sum, std::plus<>(), 0);

  if (world_rank == 0) {
    result_ = (global_sum / iterations_) * volume;
  }

  return true;
}

bool TestTaskAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    *output_ptr = result_;
  }
  return true;
}

}  // namespace lopatin_i_monte_carlo_all
