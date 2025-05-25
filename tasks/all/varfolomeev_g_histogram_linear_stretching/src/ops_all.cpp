#include "all/varfolomeev_g_histogram_linear_stretching/include/ops_all.hpp"

#include <omp.h>

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/operations.hpp>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
void StretchHistogram(std::vector<uint8_t>& local_data, int global_min, int global_max) {
  if (global_min != global_max) {
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(local_data.size()); ++i) {
      local_data[i] =
          static_cast<uint8_t>(std::round((local_data[i] - global_min) * 255.0 / (global_max - global_min)));
    }
  }
}
}  // namespace

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::ScatterData(std::vector<uint8_t>& local_data) {
  if (world_.rank() == 0) {
    int total_size = static_cast<int>(input_image_.size());
    int world_size = world_.size();

    std::vector<int> counts(world_.size());
    std::vector<int> displs(world_.size());

    int base_count = total_size / world_size;
    int remainder = total_size % world_size;

    for (int i = 0; i < world_size; ++i) {
      counts[i] = base_count + (i < remainder ? 1 : 0);
      displs[i] = (i == 0) ? 0 : (displs[i - 1] + counts[i - 1]);
    }

    for (int proc = 1; proc < world_.size(); ++proc) {
      std::vector<uint8_t> proc_data(input_image_.begin() + displs[proc],
                                     input_image_.begin() + displs[proc] + counts[proc]);
      world_.send(proc, 0, proc_data);
    }
    local_data.assign(input_image_.begin(), input_image_.begin() + counts[0]);
  } else {
    world_.recv(0, 0, local_data);
  }
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::GatherResults(const std::vector<uint8_t>& local_data) {
  int total_size = static_cast<int>(input_image_.size());
  int world_size = world_.size();

  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);

  int base_count = total_size / world_size;
  int remainder = total_size % world_size;

  for (int i = 0; i < world_size; ++i) {
    counts[i] = base_count + (i < remainder ? 1 : 0);
    displs[i] = (i == 0) ? 0 : (displs[i - 1] + counts[i - 1]);
  }

  if (world_.rank() == 0) {
    result_image_.resize(total_size);
  }

  boost::mpi::gatherv(world_, local_data.data(), static_cast<int>(local_data.size()), result_image_.data(), counts,
                      displs, 0);
}

void varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::FindMinMax(const std::vector<uint8_t>& local_data,
                                                                            int& global_min, int& global_max) {
  int local_min = 255;
  int local_max = 0;

  if (!local_data.empty()) {
    local_min = *std::ranges::min_element(local_data);
    local_max = *std::ranges::max_element(local_data);
  }

  boost::mpi::all_reduce(world_, local_min, global_min, boost::mpi::minimum<int>());
  boost::mpi::all_reduce(world_, local_max, global_max, boost::mpi::maximum<int>());

  if (global_min == global_max) {
    global_min = 0;
    global_max = 255;
  }
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    return task_data->inputs_count[0] == task_data->outputs_count[0] && task_data->inputs_count[0] > 0;
  }
  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::PreProcessingImpl() {
  input_image_.clear();
  result_image_.clear();
  if (world_.rank() == 0) {
    input_image_.resize(task_data->inputs_count[0]);
    auto* input_ptr = reinterpret_cast<uint8_t*>(task_data->inputs[0]);
    std::ranges::copy(input_ptr, input_ptr + task_data->inputs_count[0], input_image_.begin());
  }

  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::RunImpl() {
  std::vector<uint8_t> local_data;
  ScatterData(local_data);

  int global_min = 0;
  int global_max = 255;
  FindMinMax(local_data, global_min, global_max);

  StretchHistogram(local_data, global_min, global_max);

  GatherResults(local_data);

  return true;
}

bool varfolomeev_g_histogram_linear_stretching_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    auto* output_ptr = reinterpret_cast<uint8_t*>(task_data->outputs[0]);
    std::ranges::copy(result_image_.begin(), result_image_.end(), output_ptr);
  }
  return true;
}
