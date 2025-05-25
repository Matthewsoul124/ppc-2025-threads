#include "all/malyshev_a_increase_contrast_by_histogram/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "boost/mpi/collectives/all_reduce.hpp"
#include "boost/mpi/collectives/broadcast.hpp"
#include "boost/mpi/collectives/gatherv.hpp"
#include "boost/mpi/collectives/scatterv.hpp"
#include "boost/serialization/utility.hpp"  // NOLINT(misc-include-cleaner)

bool malyshev_a_increase_contrast_by_histogram_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    data_.assign(task_data->inputs[0], task_data->inputs[0] + task_data->inputs_count[0]);
  }

  return world_.rank() != 0 || !data_.empty();
}

bool malyshev_a_increase_contrast_by_histogram_all::TestTaskALL::ValidationImpl() {
  return world_.rank() != 0 ||
         (task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr && task_data->inputs_count.size() == 1 &&
          task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0]);
}

bool malyshev_a_increase_contrast_by_histogram_all::TestTaskALL::RunImpl() {
  int rank = world_.rank();
  int size = world_.size();

  int data_size = 0;
  if (rank == 0) {
    data_size = static_cast<int>(data_.size());
  }
  boost::mpi::broadcast(world_, data_size, 0);
  int chunk_size = data_size / size;
  int local_size = (rank == size - 1) ? (data_size - (rank * chunk_size)) : chunk_size;

  std::vector<int> sendcounts(size);
  std::fill(sendcounts.begin(), sendcounts.end() - 1, chunk_size);
  sendcounts.back() = data_size - (size - 1) * chunk_size;

  std::vector<uint8_t> local_data;
  local_data.resize(local_size);
  boost::mpi::scatterv(world_, data_, sendcounts, local_data.data(), 0);

  auto local_min = std::numeric_limits<uint8_t>::max();
  auto local_max = std::numeric_limits<uint8_t>::min();

#ifdef _MSC_VER
#pragma omp parallel
  {
    auto thread_min = std::numeric_limits<uint8_t>::max();
    auto thread_max = std::numeric_limits<uint8_t>::min();

#pragma omp for nowait
    for (int i = 0; i < local_size; i++) {
      thread_min = std::min(thread_min, local_data[i]);
      thread_max = std::max(thread_max, local_data[i]);
    }

#pragma omp critical
    {
      local_min = std::min(local_min, thread_min);
      local_max = std::max(local_max, thread_max);
    }
  }
#else
#pragma omp parallel for reduction(min : local_min) reduction(max : local_max)
  for (int i = 0; i < local_size; i++) {
    local_min = std::min(local_min, local_data[i]);
    local_max = std::max(local_max, local_data[i]);
  }
#endif
  std::pair<uint8_t, uint8_t> local_minmax = {local_min, local_max};
  std::pair<uint8_t, uint8_t> global_minmax;

  boost::mpi::all_reduce(world_, local_minmax, global_minmax,
                         [](const std::pair<uint8_t, uint8_t>& a, const std::pair<uint8_t, uint8_t>& b) {
                           return std::make_pair(std::min(a.first, b.first), std::max(a.second, b.second));
                         });

  auto min_value = global_minmax.first;
  auto max_value = global_minmax.second;

  if (min_value == max_value) {
    return true;
  }

  auto spectrum = std::numeric_limits<uint8_t>::max() - std::numeric_limits<uint8_t>::min();
  auto range = max_value - min_value;

#pragma omp parallel for
  for (int i = 0; i < local_size; i++) {
    local_data[i] = static_cast<uint8_t>((local_data[i] - min_value) * spectrum / range);
  }

  boost::mpi::gatherv(world_, local_data, data_.data(), sendcounts, 0);

  return true;
}

bool malyshev_a_increase_contrast_by_histogram_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(data_, task_data->outputs[0]);
  }

  return true;
}
