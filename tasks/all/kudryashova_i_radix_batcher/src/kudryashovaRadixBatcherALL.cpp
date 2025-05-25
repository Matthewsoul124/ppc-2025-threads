#include "all/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherALL.hpp"

#include <oneapi/tbb/combinable.h>
#include <oneapi/tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/partitioner.h>

#include <algorithm>
#include <array>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>  // NOLINT(*-include-cleaner)
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <ranges>
#include <vector>

namespace mpi = boost::mpi;

void kudryashova_i_radix_batcher_all::ConvertDoublesToUint64(const std::vector<double>& data,
                                                             std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = 0;
          memcpy(&bits, &data[first + i], sizeof(double));
          converted[i] = ((bits & (1ULL << 63)) != 0) ? ~bits : bits ^ (1ULL << 63);
        }
      },
      tbb::auto_partitioner());
}

void kudryashova_i_radix_batcher_all::ConvertUint64ToDoubles(std::vector<double>& data,
                                                             const std::vector<uint64_t>& converted, size_t first) {
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, converted.size()),
      [&](const auto& range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          uint64_t bits = converted[i];
          bits = ((bits & (1ULL << 63)) != 0) ? (bits ^ (1ULL << 63)) : ~bits;
          memcpy(&data[first + i], &bits, sizeof(double));
        }
      },
      tbb::auto_partitioner());
}

void kudryashova_i_radix_batcher_all::RadixDoubleSort(std::vector<double>& data, size_t first, size_t last) {
  const size_t sort_size = last - first;
  std::vector<uint64_t> converted(sort_size);
  ConvertDoublesToUint64(data, converted, first);

  std::vector<uint64_t> buffer(sort_size);
  int bits_int_byte = 8;
  int max_byte_value = 255;
  size_t total_bits = sizeof(uint64_t) * CHAR_BIT;
  for (size_t shift = 0; shift < total_bits; shift += bits_int_byte) {
    tbb::combinable<std::array<size_t, 256>> local_counts;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, sort_size), [&](const auto& range) {
      auto& counts = local_counts.local();
      for (size_t i = range.begin(); i != range.end(); ++i) {
        ++counts[(converted[i] >> shift) & max_byte_value];
      }
    });

    std::array<size_t, 256> total_counts{};
    local_counts.combine_each([&](const auto& local_count) {
      for (size_t i = 0; i < 256; ++i) {
        total_counts[i] += local_count[i];
      }
    });
    size_t total = 0;
    for (auto& safe : total_counts) {
      size_t old = safe;
      safe = total;
      total += old;
    }

    tbb::parallel_for(tbb::blocked_range<size_t>(0, 256), [&](const auto& range) {
      for (size_t j = range.begin(); j != range.end(); ++j) {
        size_t count = total_counts[j];
        for (size_t i = 0; i < sort_size; ++i) {
          if (((converted[i] >> shift) & max_byte_value) == j) {
            buffer[count++] = converted[i];
          }
        }
      }
    });

    converted.swap(buffer);
  }
  ConvertUint64ToDoubles(data, converted, first);
}

void kudryashova_i_radix_batcher_all::BatcherMerge(std::vector<double>& target_array, size_t merge_start,
                                                   size_t mid_point, size_t merge_end) {
  const size_t total_elements = merge_end - merge_start;
  const size_t left_size = mid_point - merge_start;
  const size_t right_size = merge_end - mid_point;
  std::vector<double> left_array(target_array.begin() + static_cast<std::vector<double>::difference_type>(merge_start),
                                 target_array.begin() + static_cast<std::vector<double>::difference_type>(mid_point));
  std::vector<double> right_array(target_array.begin() + static_cast<std::vector<double>::difference_type>(mid_point),
                                  target_array.begin() + static_cast<std::vector<double>::difference_type>(merge_end));
  size_t left_ptr = 0;
  size_t right_ptr = 0;
  size_t merge_ptr = merge_start;
  for (size_t i = 0; i < total_elements; ++i) {
    if (i % 2 == 0) {
      if (left_ptr < left_size && (right_ptr >= right_size || left_array[left_ptr] <= right_array[right_ptr])) {
        target_array[merge_ptr++] = left_array[left_ptr++];
      } else {
        target_array[merge_ptr++] = right_array[right_ptr++];
      }
    } else {
      if (right_ptr < right_size && (left_ptr >= left_size || right_array[right_ptr] <= left_array[left_ptr])) {
        target_array[merge_ptr++] = right_array[right_ptr++];
      } else {
        target_array[merge_ptr++] = left_array[left_ptr++];
      }
    }
  }
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs[0] == nullptr || task_data->inputs_count[0] == 0) {
      return false;
    }
    input_data_.resize(task_data->inputs_count[0]);
    std::ranges::copy(std::views::counted(reinterpret_cast<double*>(task_data->inputs[0]), task_data->inputs_count[0]),
                      input_data_.begin());
  }
  return true;
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::RunImpl() {
  if (world_.rank() == 0) {
    n_ = input_data_.size();
    size_t base_chunk = n_ / world_.size();
    size_t remainder = n_ % world_.size();

    counts_.resize(world_.size());
    displs_.resize(world_.size());

    for (size_t i = 0; i < static_cast<size_t>(world_.size()); ++i) {
      size_t chunk_size = (i < remainder) ? base_chunk + 1 : base_chunk;
      counts_[i] = static_cast<int>(chunk_size);
      displs_[i] = (i == 0) ? 0 : displs_[i - 1] + counts_[i - 1];
    }
  }

  mpi::broadcast(world_, n_, 0);
  mpi::broadcast(world_, counts_, 0);
  mpi::broadcast(world_, displs_, 0);

  int local_size = counts_[world_.rank()];
  std::vector<double> local_data(local_size);
  boost::mpi::scatterv(world_, world_.rank() == 0 ? input_data_.data() : nullptr, counts_, displs_, local_data.data(),
                       local_size, 0);

  RadixDoubleSort(local_data, 0, local_data.size());

  std::vector<double> global_data;
  if (world_.rank() == 0) {
    global_data.resize(n_);
  }
  boost::mpi::gatherv(world_, local_data.data(), local_size, world_.rank() == 0 ? global_data.data() : nullptr, counts_,
                      displs_, 0);

  if (world_.rank() == 0) {
    input_data_ = global_data;
    size_t current_block_size = *std::ranges::max_element(counts_);
    while (current_block_size < n_) {
      size_t next_block_size = std::min(2 * current_block_size, n_);
      for (size_t start = 0; start < n_; start += next_block_size) {
        size_t mid = std::min(start + current_block_size, n_);
        size_t end = std::min(start + next_block_size, n_);
        if (mid < end) {
          BatcherMerge(input_data_, start, mid, end);
        }
      }
      current_block_size = next_block_size;
    }
    all_data_ = input_data_;
  }
  return true;
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::ValidationImpl() {
  if (world_.rank() == 0) {
    if (task_data->inputs.empty() || task_data->inputs[0] == nullptr || task_data->inputs_count.empty() ||
        task_data->inputs_count[0] == 0) {
      return false;
    }
    if (task_data->outputs.empty() || task_data->outputs[0] == nullptr || task_data->outputs_count.empty() ||
        task_data->outputs_count[0] != task_data->inputs_count[0]) {
      return false;
    }
  }
  return true;
}

bool kudryashova_i_radix_batcher_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    std::ranges::copy(all_data_, reinterpret_cast<double*>(task_data->outputs[0]));
  }
  return true;
}
