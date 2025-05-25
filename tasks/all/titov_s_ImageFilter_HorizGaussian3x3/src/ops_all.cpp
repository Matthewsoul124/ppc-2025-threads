#include "all/titov_s_ImageFilter_HorizGaussian3x3/include/ops_all.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);

  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  if (kernel_.size() != 3 || sqrt_size * sqrt_size != size) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::DistributeData(int world_rank, int world_size,
                                                                                   int height, int width, int start_row,
                                                                                   int end_row,
                                                                                   std::vector<double> &local_input) {
  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);

  int offset = 0;
  for (int p = 0; p < world_size; ++p) {
    int rows_for_p = (height / world_size) + (p < (height % world_size) ? 1 : 0);
    counts[p] = rows_for_p * width;
    displs[p] = offset;
    offset += counts[p];
  }

  if (world_rank == 0) {
    for (int p = 1; p < world_size; ++p) {
      world_.send(p, 0, &input_[displs[p]], counts[p]);
    }
    std::ranges::copy(input_.begin(), input_.begin() + counts[0], local_input.begin());
  } else {
    world_.recv(0, 0, local_input.data(), counts[world_rank]);
  }

  return true;
}

void titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::ProcessRows(const std::vector<double> &local_input,
                                                                                std::vector<double> &local_output,
                                                                                int width, int local_rows,
                                                                                int num_threads) {
  const double sum = kernel_[0] + kernel_[1] + kernel_[2];
  const double k0 = kernel_[0];
  const double k1 = kernel_[1];
  const double k2 = kernel_[2];

  auto process_row = [&](int i) {
    const int row_offset = i * width;
    const double *in_row = local_input.data() + row_offset;
    double *out_row = local_output.data() + row_offset;

    out_row[0] = (in_row[0] * k1 + in_row[1] * k2) / sum;

    for (int j = 1; j < width - 1; ++j) {
      out_row[j] = (in_row[j - 1] * k0 + in_row[j] * k1 + in_row[j + 1] * k2) / sum;
    }

    out_row[width - 1] = (in_row[width - 2] * k0 + in_row[width - 1] * k1) / sum;
  };

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::atomic<int> next_row{0};

  for (int t = 0; t < num_threads; ++t) {
    threads.emplace_back([&] {
      while (true) {
        int row = next_row.fetch_add(1, std::memory_order_relaxed);
        if (row >= local_rows) {
          break;
        }
        process_row(row);
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::CollectResults(
    int world_rank, int world_size, int height, int width, int start_row, const std::vector<double> &local_output) {
  std::vector<int> counts(world_size);
  std::vector<int> displs(world_size);

  int offset = 0;
  for (int p = 0; p < world_size; ++p) {
    int rows_for_p = (height / world_size) + (p < (height % world_size) ? 1 : 0);
    counts[p] = rows_for_p * width;
    displs[p] = offset;
    offset += counts[p];
  }

  if (world_rank == 0) {
    std::ranges::copy(local_output.begin(), local_output.end(), output_.begin() + displs[0]);

    for (int p = 1; p < world_size; ++p) {
      world_.recv(p, 0, &output_[displs[p]], counts[p]);
    }
  } else {
    world_.send(0, 0, local_output.data(), counts[world_rank]);
  }

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::RunImpl() {
  const int width = width_;
  const int height = height_;
  const int world_size = world_.size();
  const int world_rank = world_.rank();
  const int num_threads = ppc::util::GetPPCNumThreads();

  if (world_size > height) {
    return false;
  }

  const int rows_per_process = height / world_size;
  const int remainder = height % world_size;
  const int extra_row = (world_rank < remainder) ? 1 : 0;
  const int start_row = (world_rank * rows_per_process) + std::min(world_rank, remainder);
  const int end_row = start_row + rows_per_process + extra_row;
  const int local_rows = end_row - start_row;

  if (local_rows <= 0 || width <= 0) {
    return false;
  }

  std::vector<double> local_input(local_rows * width);
  std::vector<double> local_output(local_rows * width, 0.0);

  if (!DistributeData(world_rank, world_size, height, width, start_row, end_row, local_input)) {
    return false;
  }

  ProcessRows(local_input, local_output, width, local_rows, num_threads);

  if (!CollectResults(world_rank, world_size, height, width, start_row, local_output)) {
    return false;
  }

  world_.barrier();
  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_all::GaussianFilterALL::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);
  std::ranges::copy(output_, out_ptr);
  return true;
}