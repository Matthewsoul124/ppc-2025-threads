// Golovkin Maksim
#include "stl/golovkin_contrast_stretching/include/ops_stl.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSTL<PixelType>::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSTL<PixelType>::PreProcessingImpl() {
  image_size_ = task_data->inputs_count[0] / sizeof(PixelType);

  if (image_size_ == 0) {
    return true;
  }
  auto* input_ptr = reinterpret_cast<PixelType*>(task_data->inputs[0]);
  input_image_.assign(input_ptr, input_ptr + image_size_);
  output_image_.resize(image_size_);

  auto [min_it, max_it] = std::ranges::minmax_element(input_image_);
  min_val_ = *min_it;
  max_val_ = *max_it;

  return true;
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSTL<PixelType>::RunImpl() {
  if (image_size_ == 0) {
    return true;
  }
  if (min_val_ == max_val_) {
    std::ranges::fill(output_image_, 0);
    return true;
  }

  const double scale = 255.0 / (max_val_ - min_val_);

  unsigned int num_threads = 0;
  num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::min(num_threads, static_cast<unsigned int>(image_size_));

  if (num_threads == 0) {
    num_threads = 1;
  }

  size_t chunk_size = image_size_ / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start = t * chunk_size;
    size_t end = (t == num_threads - 1) ? image_size_ : start + chunk_size;

    threads.emplace_back([=, this] {
      for (size_t i = start; i < end; ++i) {
        double stretched = (input_image_[i] - min_val_) * scale;

        if constexpr (std::is_same_v<PixelType, uint8_t>) {
          output_image_[i] = static_cast<uint8_t>(std::clamp(static_cast<int>(stretched + 1e-9), 0, 255));
        } else if constexpr (std::is_same_v<PixelType, uint16_t>) {
          output_image_[i] = static_cast<uint16_t>(std::clamp(static_cast<int>(stretched + 1e-9), 0, 255));
        } else {
          output_image_[i] = static_cast<PixelType>(stretched);
        }
      }
    });
  }

  for (auto& thread : threads) {
    thread.join();
  }

  return true;
}

template <typename PixelType>
bool golovkin_contrast_stretching::ContrastStretchingSTL<PixelType>::PostProcessingImpl() {
  if (image_size_ == 0) {
    return true;
  }

  auto* output_ptr = reinterpret_cast<PixelType*>(task_data->outputs[0]);
  std::memcpy(output_ptr, output_image_.data(), output_image_.size() * sizeof(PixelType));
  return true;
}

template class golovkin_contrast_stretching::ContrastStretchingSTL<uint8_t>;
template class golovkin_contrast_stretching::ContrastStretchingSTL<uint16_t>;
template class golovkin_contrast_stretching::ContrastStretchingSTL<float>;