#include "stl/titov_s_ImageFilter_HorizGaussian3x3/include/ops_stl.hpp"

#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
  width_ = height_ = static_cast<int>(std::sqrt(input_size));
  input_.assign(in_ptr, in_ptr + input_size);

  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  output_ = std::vector<double>(input_size, 0.0);

  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::ValidationImpl() {
  auto *kernel_ptr = reinterpret_cast<int *>(task_data->inputs[1]);
  kernel_ = std::vector<int>(kernel_ptr, kernel_ptr + 3);
  size_t size = input_.size();
  auto sqrt_size = static_cast<size_t>(std::sqrt(size));
  if (kernel_.size() != 3 || sqrt_size * sqrt_size != size) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::RunImpl() {
  const double sum = kernel_[0] + kernel_[1] + kernel_[2];
  const int width = width_;
  const int height = height_;

  const auto num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  const int rows_per_thread = height / num_threads;

  for (int t = 0; t < num_threads; ++t) {
    const int start_row = t * rows_per_thread;
    const int end_row = (t == num_threads - 1) ? height : (start_row + rows_per_thread);

    threads.emplace_back([=, &input = input_, &output = output_, &kernel = kernel_] {
      for (int i = start_row; i < end_row; ++i) {
        const int row_offset = i * width;
        for (int j = 0; j < width; ++j) {
          double val = input[row_offset + j] * kernel[1];
          if (j > 0) {
            val += input[row_offset + j - 1] * kernel[0];
          }
          if (j < width - 1) {
            val += input[row_offset + j + 1] * kernel[2];
          }
          output[row_offset + j] = val / sum;
        }
      }
    });
  }

  for (auto &t : threads) {
    t.join();
  }
  return true;
}

bool titov_s_image_filter_horiz_gaussian3x3_stl::GaussianFilterSTL::PostProcessingImpl() {
  auto *out_ptr = reinterpret_cast<double *>(task_data->outputs[0]);

  for (size_t i = 0; i < output_.size(); i++) {
    out_ptr[i] = output_[i];
  }

  return true;
}
