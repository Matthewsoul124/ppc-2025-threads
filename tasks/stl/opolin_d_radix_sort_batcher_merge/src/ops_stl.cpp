#include "stl/opolin_d_radix_sort_batcher_merge/include/ops_stl.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + size_);
  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size);
  unsigned_data_.resize(size_);
  return true;
}

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::ValidationImpl() {
  // Check equality of counts elements
  size_ = static_cast<int>(task_data->inputs_count[0]);
  if (size_ <= 0 || task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }
  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::RunImpl() {
  int num_threads = ppc::util::GetPPCNumThreads();
  if (size_ <= 1) {
    output_ = input_;
    return true;
  }
  auto unum_threads = static_cast<size_t>(num_threads);
  ParallelProcessRange(static_cast<size_t>(size_), static_cast<unsigned int>(unum_threads),
                       [this](size_t start, size_t end) {
                         for (size_t i = start; i < end; ++i) {
                           unsigned_data_[i] = IntToUnsigned(input_[i]);
                         }
                       });

  size_t block_size = (static_cast<size_t>(size_) + unum_threads - 1) / unum_threads;
  size_t actual_num_blocks = (static_cast<size_t>(size_) + block_size - 1) / block_size;
  std::vector<std::function<void()>> sort_tasks;
  sort_tasks.reserve(actual_num_blocks);
  for (unsigned int i = 0; i < actual_num_blocks; ++i) {
    size_t start = i * block_size;
    size_t end = std::min(start + block_size, static_cast<size_t>(size_));
    if (start < end) {
      sort_tasks.emplace_back([this, start, end]() {
        RadixSortLSD(unsigned_data_.begin() + static_cast<ptrdiff_t>(start),
                     unsigned_data_.begin() + static_cast<ptrdiff_t>(end));
      });
    }
  }
  if (!sort_tasks.empty()) {
    ParallelRunTasks(sort_tasks);
  }
  IterativeOddEvenBlockMerge(unsigned_data_.begin(), unsigned_data_.end(), actual_num_blocks, block_size,
                             static_cast<unsigned int>(unum_threads));
  ParallelProcessRange(static_cast<size_t>(size_), static_cast<unsigned int>(unum_threads),
                       [this](size_t start, size_t end) {
                         for (size_t i = start; i < end; ++i) {
                           output_[i] = UnsignedToInt(unsigned_data_[i]);
                         }
                       });
  return true;
}

bool opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

uint32_t opolin_d_radix_batcher_sort_stl::IntToUnsigned(int value) { return static_cast<uint32_t>(value) ^ (1U << 31); }

int opolin_d_radix_batcher_sort_stl::UnsignedToInt(uint32_t value) { return static_cast<int>(value ^ (1U << 31)); }

void opolin_d_radix_batcher_sort_stl::ParallelProcessRange(size_t total_size, unsigned int num_threads,
                                                           const std::function<void(size_t start, size_t end)>& func) {
  if (total_size == 0) {
    return;
  }
  unsigned int actual_threads = std::min(num_threads, static_cast<unsigned int>(total_size));
  std::vector<std::thread> threads;
  threads.reserve(actual_threads);
  size_t block_size = (total_size + actual_threads - 1) / actual_threads;

  for (unsigned int i = 0; i < actual_threads; ++i) {
    size_t start = i * block_size;
    size_t end = std::min(start + block_size, total_size);
    if (start < end) {
      threads.emplace_back(func, start, end);
    }
  }
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void opolin_d_radix_batcher_sort_stl::ParallelRunTasks(const std::vector<std::function<void()>>& tasks) {
  if (tasks.empty()) {
    return;
  }
  unsigned int num_threads = ppc::util::GetPPCNumThreads();
  unsigned int actual_threads = std::min(num_threads, static_cast<unsigned int>(tasks.size()));
  std::vector<std::thread> threads;
  threads.reserve(actual_threads);
  std::atomic<size_t> task_idx(0);

  for (unsigned int i = 0; i < actual_threads; ++i) {
    threads.emplace_back([&]() {
      size_t current_task = 0;
      while ((current_task = task_idx.fetch_add(1, std::memory_order_relaxed)) < tasks.size()) {
        tasks[current_task]();
      }
    });
  }
  for (auto& t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }
}

void opolin_d_radix_batcher_sort_stl::RadixSortLSD(std::vector<uint32_t>::iterator begin,
                                                   std::vector<uint32_t>::iterator end) {
  size_t n = std::distance(begin, end);
  if (n <= 1) {
    return;
  }
  const int radix = 256;
  const int num_passes = sizeof(uint32_t);

  std::vector<uint32_t> buffer(n);
  std::vector<size_t> count(radix);

  for (int pass = 0; pass < num_passes; ++pass) {
    int shift = pass * 8;
    std::ranges::fill(count, 0);
    for (auto it = begin; it != end; ++it) {
      count[(*it >> shift) & (radix - 1)]++;
    }
    size_t cumulative_sum = 0;
    for (size_t i = 0; i < radix; ++i) {
      size_t current_count = count[i];
      count[i] = cumulative_sum;
      cumulative_sum += current_count;
    }
    for (auto it = begin; it != end; ++it) {
      buffer[count[(*it >> shift) & (radix - 1)]++] = *it;
    }
    std::copy(buffer.begin(), buffer.begin() + static_cast<ptrdiff_t>(n), begin);
  }
}

void opolin_d_radix_batcher_sort_stl::IterativeOddEvenBlockMerge(std::vector<uint32_t>::iterator data_begin,
                                                                 std::vector<uint32_t>::iterator data_end,
                                                                 size_t num_initial_blocks, size_t initial_block_size,
                                                                 unsigned int num_threads) {
  size_t n = std::distance(data_begin, data_end);
  if (num_initial_blocks <= 1 || n <= 1) {
    return;
  }
  size_t current_merge_block_size = initial_block_size;
  while (current_merge_block_size < n) {
    std::vector<std::function<void()>> merge_tasks;

    for (size_t i = 0; i < n; i += 2 * current_merge_block_size) {
      auto merge_begin = data_begin + static_cast<ptrdiff_t>(i);
      auto merge_mid = data_begin + static_cast<ptrdiff_t>(std::min(i + current_merge_block_size, n));
      auto merge_end = data_begin + static_cast<ptrdiff_t>(std::min(i + (2 * current_merge_block_size), n));
      if (merge_mid < merge_end) {
        merge_tasks.emplace_back(
            [merge_begin, merge_mid, merge_end]() { std::inplace_merge(merge_begin, merge_mid, merge_end); });
      }
    }
    if (!merge_tasks.empty()) {
      ParallelRunTasks(merge_tasks);
    }
    current_merge_block_size *= 2;
  }
}