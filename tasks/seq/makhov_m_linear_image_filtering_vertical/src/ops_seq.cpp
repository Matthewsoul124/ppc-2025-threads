#include "seq/makhov_m_linear_image_filtering_vertical/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

bool makhov_m_linear_image_filtering_vertical_seq::TaskSequential::PreProcessingImpl() {
  // Init value for input, output, kernel
  width_ = (int)(task_data->inputs_count[0]);
  height_ = (int)(task_data->inputs_count[1]);
  input_size_ = width_ * height_ * 3;
  auto *in_ptr = reinterpret_cast<uint8_t *>(task_data->inputs[0]);
  input_.assign(in_ptr, in_ptr + input_size_);
  output_.assign(input_.begin(), input_.end());
  kernel_ = {0.25F, 0.5F, 0.25F};  // [1, 2, 1] * 1/4
  return true;
}

bool makhov_m_linear_image_filtering_vertical_seq::TaskSequential::ValidationImpl() {
  // Check equality of counts elements
  return ((task_data->inputs_count[0] * task_data->inputs_count[1] * 3 >= 27) &&
          ((task_data->inputs_count[0] * task_data->inputs_count[1] * 3) == task_data->outputs_count[0]));
}

bool makhov_m_linear_image_filtering_vertical_seq::TaskSequential::RunImpl() {
  std::vector<uint8_t> temp(input_size_);
  makhov_m_linear_image_filtering_vertical_seq::TaskSequential::ApplyVerticalGaussian(input_, temp, width_, height_,
                                                                                      kernel_);
  makhov_m_linear_image_filtering_vertical_seq::TaskSequential::ApplyHorizontalGaussian(temp, output_, width_, height_,
                                                                                        kernel_);
  return true;
}

bool makhov_m_linear_image_filtering_vertical_seq::TaskSequential::PostProcessingImpl() {
  std::memcpy(task_data->outputs[0], output_.data(),
              std::min(output_.size(), static_cast<size_t>(task_data->outputs_count[0])));
  return true;
}

// Applying 1D Gaussian Kernel to a row (RGB version)
void makhov_m_linear_image_filtering_vertical_seq::TaskSequential::ApplyHorizontalGaussian(
    const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int height,
    const std::vector<float> &kernel) {
  const int kernel_radius = (int)(kernel.size() / 2);
  const int channels = 3;  // RGB = 3 канала

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float sum_r = 0.0F;
      float sum_g = 0.0F;
      float sum_b = 0.0F;

      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_x = std::clamp(x + k, 0, width - 1);
        int src_pos = (y * width + pixel_x) * channels;

        float weight = kernel[k + kernel_radius];
        sum_r += (float)src[src_pos] * weight;      // R
        sum_g += (float)src[src_pos + 1] * weight;  // G
        sum_b += (float)src[src_pos + 2] * weight;  // B
      }

      int dst_pos = (y * width + x) * channels;
      dst[dst_pos] = static_cast<uint8_t>(std::clamp(sum_r, 0.0F, 255.0F));
      dst[dst_pos + 1] = static_cast<uint8_t>(std::clamp(sum_g, 0.0F, 255.0F));
      dst[dst_pos + 2] = static_cast<uint8_t>(std::clamp(sum_b, 0.0F, 255.0F));
    }
  }
}

// Applying 1D Gaussian Kernel to a column (RGB version)
void makhov_m_linear_image_filtering_vertical_seq::TaskSequential::ApplyVerticalGaussian(
    const std::vector<uint8_t> &src, std::vector<uint8_t> &dst, int width, int height,
    const std::vector<float> &kernel) {
  const int kernel_radius = (int)(kernel.size() / 2);
  const int channels = 3;  // RGB = 3 канала

  for (int x = 0; x < width; ++x) {
    for (int y = 0; y < height; ++y) {
      float sum_r = 0.0F;
      float sum_g = 0.0F;
      float sum_b = 0.0F;

      for (int k = -kernel_radius; k <= kernel_radius; ++k) {
        int pixel_y = std::clamp(y + k, 0, height - 1);
        int src_pos = (pixel_y * width + x) * channels;

        float weight = kernel[k + kernel_radius];
        sum_r += (float)src[src_pos] * weight;      // R
        sum_g += (float)src[src_pos + 1] * weight;  // G
        sum_b += (float)src[src_pos + 2] * weight;  // B
      }

      int dst_pos = (y * width + x) * channels;
      dst[dst_pos] = static_cast<uint8_t>(std::clamp(sum_r, 0.0F, 255.0F));
      dst[dst_pos + 1] = static_cast<uint8_t>(std::clamp(sum_g, 0.0F, 255.0F));
      dst[dst_pos + 2] = static_cast<uint8_t>(std::clamp(sum_b, 0.0F, 255.0F));
    }
  }
}
