#include "all/kozlova_e_contrast_enhancement/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/all_reduce.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/operations.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

bool kozlova_e_contrast_enhancement_all::TestTaskAll::PreProcessingImpl() {
  if (world_.rank() == 0) {
    auto *input_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
    size_t size = task_data->inputs_count[0];
    width_ = task_data->inputs_count[1];
    height_ = task_data->inputs_count[2];
    output_.resize(size, 0);
    input_.resize(size);
    std::copy(input_ptr, input_ptr + size, input_.begin());
  }

  return true;
}

bool kozlova_e_contrast_enhancement_all::TestTaskAll::ValidationImpl() {
  if (world_.rank() == 0) {
    size_t size = task_data->inputs_count[0];
    size_t check_width = task_data->inputs_count[1];
    size_t check_height = task_data->inputs_count[2];
    return size == task_data->outputs_count[0] && size > 0 && (size % 2 == 0) && check_width >= 1 &&
           check_height >= 1 && (size == check_height * check_width);
  }
  return true;
}

bool kozlova_e_contrast_enhancement_all::TestTaskAll::RunImpl() {
  int rank = world_.rank();
  int proc_count = world_.size();

  size_t input_size = 0;
  if (rank == 0) {
    input_size = input_.size();
  }

  boost::mpi::broadcast(world_, input_size, 0);
  input_.resize(input_size);
  boost::mpi::broadcast(world_, input_.data(), (int)input_.size(), 0);

  int local_size = (int)input_size / proc_count;
  int remainder = (int)input_size % proc_count;
  std::vector<int> counts(proc_count, local_size);
  for (int i = 0; i < remainder; ++i) {
    counts[i]++;
  }

  std::vector<int> displs(proc_count, 0);
  for (int i = 1; i < proc_count; ++i) {
    displs[i] = displs[i - 1] + counts[i - 1];
  }

  std::vector<uint8_t> local_input(counts[rank]);

  boost::mpi::scatterv(world_, input_.data(), counts, displs, local_input.data(), counts[rank], 0);

  uint8_t local_min = 255;
  uint8_t local_max = 0;
  if (!local_input.empty()) {
    local_min = *std::ranges::min_element(local_input);
    local_max = *std::ranges::max_element(local_input);
  }

  uint8_t global_min = 0;
  uint8_t global_max = 0;

  boost::mpi::all_reduce(world_, local_min, global_min, boost::mpi::minimum<uint8_t>());
  boost::mpi::all_reduce(world_, local_max, global_max, boost::mpi::maximum<uint8_t>());
  boost::mpi::broadcast(world_, global_max, 0);
  boost::mpi::broadcast(world_, global_min, 0);

  std::vector<uint8_t> local_output(local_input.size());

  if (global_min == global_max) {
    std::ranges::copy(local_input, local_output.data());
  } else {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < static_cast<int>(local_input.size()); ++i) {
      local_output[i] = static_cast<uint8_t>(((local_input[i] - global_min) / (double)(global_max - global_min)) * 255);
      local_output[i] = std::clamp(static_cast<int>(local_output[i]), 0, 255);
    }
  }

  std::vector<uint8_t> gathered;
  if (rank == 0) {
    gathered.resize(input_size);
  }

  boost::mpi::gatherv(world_, local_output.data(), counts[rank], gathered.data(), counts, displs, 0);

  if (rank == 0) {
    output_ = std::move(gathered);
  }

  return rank == 0;
}

bool kozlova_e_contrast_enhancement_all::TestTaskAll::PostProcessingImpl() {
  if (world_.rank() == 0) {
    for (size_t i = 0; i < output_.size(); ++i) {
      reinterpret_cast<uint8_t *>(task_data->outputs[0])[i] = output_[i];
    }
  }
  return true;
}