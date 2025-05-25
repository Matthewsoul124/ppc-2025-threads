#include "../include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace milovankin_m_histogram_stretching_stl {

bool TestTaskSTL::ValidationImpl() {
  return !task_data->inputs.empty() && !task_data->inputs_count.empty() && task_data->inputs_count[0] != 0 &&
         !task_data->outputs.empty() && !task_data->outputs_count.empty() &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool TestTaskSTL::PreProcessingImpl() {
  const uint8_t* input_data = task_data->inputs.front();
  const uint32_t input_size = task_data->inputs_count.front();

  img_.assign(input_data, input_data + input_size);
  return true;
}

struct MinMaxPair {
  uint8_t min_val;
  uint8_t max_val;

  MinMaxPair() : min_val(std::numeric_limits<uint8_t>::max()), max_val(0) {}

  MinMaxPair(uint8_t min, uint8_t max) : min_val(min), max_val(max) {}
};

bool TestTaskSTL::RunImpl() {
  if (img_.empty()) {
    return true;
  }

  auto num_threads = static_cast<std::size_t>(ppc::util::GetPPCNumThreads());
  if (num_threads == 0) {
    num_threads = 1;
  }
  num_threads = std::min<std::size_t>(num_threads, img_.size());

  auto [min_it, max_it] = std::ranges::minmax_element(img_.begin(), img_.end());
  const uint8_t min_val = *min_it;
  const uint8_t max_val = *max_it;

  if (min_val == max_val) {
    return true;
  }

  const int delta = static_cast<int>(max_val) - static_cast<int>(min_val);

  const std::size_t img_size = img_.size();
  const std::size_t base_chunk = img_size / num_threads;
  const std::size_t remainder = img_size % num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  std::size_t offset = 0;
  for (std::size_t i = 0; i < num_threads; ++i) {
    const std::size_t chunk = base_chunk + (i < remainder ? 1 : 0);
    const std::size_t start = offset;
    const std::size_t end = start + chunk;

    threads.emplace_back([this, start, end, min_val, delta]() {
      std::transform(std::next(img_.begin(), static_cast<std::ptrdiff_t>(start)),
                     std::next(img_.begin(), static_cast<std::ptrdiff_t>(end)),
                     std::next(img_.begin(), static_cast<std::ptrdiff_t>(start)),
                     [min_val, delta](uint8_t px) -> uint8_t {
                       int v = ((static_cast<int>(px) - static_cast<int>(min_val)) * 255 + delta / 2) / delta;
                       return static_cast<uint8_t>(v);
                     });
    });

    offset = end;
  }

  for (auto& t : threads) {
    t.join();
  }

  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  uint8_t* output_data = task_data->outputs[0];
  const uint32_t output_size = task_data->outputs_count[0];
  const uint32_t copy_size = std::min(output_size, static_cast<uint32_t>(img_.size()));

  std::copy_n(img_.cbegin(), copy_size, output_data);
  return true;
}

}  // namespace milovankin_m_histogram_stretching_stl
