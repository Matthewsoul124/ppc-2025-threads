#include "stl/khovansky_d_double_radix_batcher/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace khovansky_d_double_radix_batcher_stl {
namespace {
uint64_t EncodeDoubleToUint64(double value) {
  uint64_t bit_representation = 0;
  std::memcpy(&bit_representation, &value, sizeof(value));

  if ((bit_representation >> 63) != 0) {
    return ~bit_representation;
  }
  return bit_representation ^ (1ULL << 63);
}

double DecodeUint64ToDouble(uint64_t encoded) {
  if ((encoded >> 63) != 0) {
    encoded ^= (1ULL << 63);
  } else {
    encoded = ~encoded;
  }

  double result = 0.0;
  std::memcpy(&result, &encoded, sizeof(result));
  return result;
}

void RadixSort(std::vector<uint64_t>& array, int thread_count) {
  const int bits_in_byte = 8;
  const int total_bits = 64;
  const int bucket_count = 256;

  std::vector<uint64_t> buffer(array.size(), 0);
  std::vector<std::vector<int>> local_frequencies(thread_count, std::vector<int>(bucket_count, 0));

  for (int shift = 0; shift < total_bits; shift += bits_in_byte) {
    std::vector<std::thread> threads;
    size_t n = array.size();
    size_t block_size = (n + thread_count - 1) / thread_count;

    for (int t = 0; t < thread_count; ++t) {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);

      threads.emplace_back([&, begin, end, t]() {
        for (size_t i = begin; i < end; ++i) {
          auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
          local_frequencies[t][bucket]++;
        }
      });
    }
    for (auto& th : threads) {
      th.join();
    }

    std::vector<int> frequency(bucket_count, 0);
    for (int b = 0; b < bucket_count; ++b) {
      for (int t = 0; t < thread_count; ++t) {
        frequency[b] += local_frequencies[t][b];
        local_frequencies[t][b] = 0;
      }
    }

    for (int i = 1; i < bucket_count; i++) {
      frequency[i] += frequency[i - 1];
    }

    for (int i = static_cast<int>(array.size()) - 1; i >= 0; i--) {
      auto bucket = static_cast<uint8_t>((array[i] >> shift) & 0xFF);
      buffer[--frequency[bucket]] = array[i];
    }

    array.swap(buffer);
  }
}

void BatcherOddEvenMerge(std::vector<uint64_t>& array, int left, int right, int max_threads) {
  if (right - left <= 1) {
    return;
  }

  int middle = left + ((right - left) / 2);

  std::thread left_thread;
  std::thread right_thread;

  if (max_threads > 1) {
    left_thread = std::thread([&]() { BatcherOddEvenMerge(array, left, middle, max_threads / 2); });
    right_thread = std::thread([&]() { BatcherOddEvenMerge(array, middle, right, max_threads / 2); });
  } else {
    BatcherOddEvenMerge(array, left, middle, 1);
    BatcherOddEvenMerge(array, middle, right, 1);
  }

  if (left_thread.joinable()) {
    left_thread.join();
  }
  if (right_thread.joinable()) {
    right_thread.join();
  }

  for (int i = left; i + 1 < right; i += 2) {
    if (array[i] > array[i + 1]) {
      std::swap(array[i], array[i + 1]);
    }
  }
}

void RadixBatcherSort(std::vector<double>& data) {
  std::vector<uint64_t> transformed_data(data.size());
  size_t n = data.size();
  const int thread_count = std::max(1, std::min(static_cast<int>(n), ppc::util::GetPPCNumThreads()));
  size_t block_size = (n + thread_count - 1) / thread_count;
  std::vector<std::thread> threads(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&data, &transformed_data, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        transformed_data[i] = EncodeDoubleToUint64(data[i]);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }

  RadixSort(transformed_data, thread_count);
  BatcherOddEvenMerge(transformed_data, 0, static_cast<int>(n), thread_count);

  threads.clear();
  threads.resize(thread_count);

  for (int t = 0; t < thread_count; ++t) {
    threads[t] = std::thread([&data, &transformed_data, t, block_size, n]() {
      size_t begin = t * block_size;
      size_t end = std::min(begin + block_size, n);
      for (size_t i = begin; i < end; ++i) {
        data[i] = DecodeUint64ToDouble(transformed_data[i]);
      }
    });
  }
  for (auto& th : threads) {
    th.join();
  }
}
}  // namespace
}  // namespace khovansky_d_double_radix_batcher_stl

bool khovansky_d_double_radix_batcher_stl::RadixSTL::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);

  unsigned int input_size = task_data->inputs_count[0];
  unsigned int output_size = task_data->outputs_count[0];

  input_ = std::vector<double>(in_ptr, in_ptr + input_size);
  output_ = std::vector<double>(output_size, 0);

  return true;
}

bool khovansky_d_double_radix_batcher_stl::RadixSTL::ValidationImpl() {
  if (!task_data) {
    return false;
  }

  if (task_data->inputs[0] == nullptr && task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count[0] < 2) {
    return false;
  }

  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool khovansky_d_double_radix_batcher_stl::RadixSTL::RunImpl() {
  output_ = input_;
  khovansky_d_double_radix_batcher_stl::RadixBatcherSort(output_);
  return true;
}

bool khovansky_d_double_radix_batcher_stl::RadixSTL::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<double*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}
