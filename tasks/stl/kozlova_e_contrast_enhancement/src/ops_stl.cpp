#include "stl/kozlova_e_contrast_enhancement/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool kozlova_e_contrast_enhancement_stl::TestTaskSTL::PreProcessingImpl() {
  auto *input_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  size_t size = task_data->inputs_count[0];
  width_ = task_data->inputs_count[1];
  height_ = task_data->inputs_count[2];
  output_.resize(size, 0);
  input_.resize(size);
  std::copy(input_ptr, input_ptr + size, input_.begin());

  return true;
}

bool kozlova_e_contrast_enhancement_stl::TestTaskSTL::ValidationImpl() {
  size_t size = task_data->inputs_count[0];
  size_t check_width = task_data->inputs_count[1];
  size_t check_height = task_data->inputs_count[2];
  return size == task_data->outputs_count[0] && size > 0 && (size % 2 == 0) && check_width >= 1 && check_height >= 1 &&
         (size == check_height * check_width);
}

bool kozlova_e_contrast_enhancement_stl::TestTaskSTL::RunImpl() {
  int min_value = *std::ranges::min_element(input_);
  if (min_value < 0) {
    throw "incorrect value";
  }
  int max_value = *std::ranges::max_element(input_);

  if (min_value == max_value) {
    std::ranges::copy(input_, output_.begin());
    return true;
  }

  auto normalize_pixel = [=](int value) {
    return std::clamp(static_cast<int>(((value - min_value) / static_cast<double>(max_value - min_value)) * 255), 0,
                      255);
  };

  size_t num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::clamp(num_threads, static_cast<size_t>(1), height_);
  std::vector<std::thread> threads(num_threads);

  size_t rows_per_thread = height_ / num_threads;
  size_t leftover_rows = height_ % num_threads;

  auto process_rows = [&](size_t start_row, size_t end_row) {
    for (size_t row = start_row; row < end_row; ++row) {
      for (size_t col = 0; col < width_; ++col) {
        size_t idx = (row * width_) + col;
        output_[idx] = normalize_pixel(input_[idx]);
      }
    }
  };

  size_t current_row = 0;
  for (size_t t = 0; t < num_threads; ++t) {
    size_t end_row = current_row + rows_per_thread + (t < leftover_rows ? 1 : 0);
    threads[t] = std::thread(process_rows, current_row, end_row);
    current_row = end_row;
  }

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  return true;
}

bool kozlova_e_contrast_enhancement_stl::TestTaskSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); ++i) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
