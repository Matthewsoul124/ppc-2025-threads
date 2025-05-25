
#include "all/tsatsyn_a_radix_sort_simple_merge/include/ops_all.hpp"

#include <algorithm>
#include <bit>
#include <boost/mpi/communicator.hpp>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace {
int CalculateBits(const std::vector<uint64_t> &data, bool is_pozitive) {
  if (data.empty()) {
    return 0;
  }
  uint64_t extreme_val = 0;
  int num_bits = 0;
  if (is_pozitive) {
    extreme_val = *std::ranges::max_element(data);
    num_bits = std::bit_width(extreme_val);
  } else {
    extreme_val = *std::ranges::min_element(data);
    num_bits = (extreme_val == 0) ? 0 : std::bit_width(extreme_val);
  }

  return num_bits;
}
inline void SendData(boost::mpi::communicator &world, std::vector<bool> &is_pozitive, std::vector<bool> &is_negative,
                     std::vector<double> &local_data, std::vector<double> &input_data) {
  if (world.size() > 1) {
    for (int proc = 1; proc < world.size(); proc++) {
      for (size_t j = proc; j < input_data.size(); j += world.size()) {
        input_data[j] < 0.0 ? is_negative[proc - 1] = true : is_pozitive[proc - 1] = true;
        local_data.push_back(input_data[j]);
      }
      world.send(proc, 0, local_data);
      local_data.clear();
    }
  }
  for (int j = 0; j < static_cast<int>(input_data.size()); j += world.size()) {
    local_data.push_back(input_data[j]);
  }
}
inline void ParallelParse(std::vector<uint64_t> &pozitive_copy, std::vector<uint64_t> &negative_copy,
                          std::vector<double> &local_data) {
#pragma omp parallel
  {
    std::vector<uint64_t> local_positive;
    std::vector<uint64_t> local_negative;
#pragma omp for nowait
    for (int i = 0; i < static_cast<int>(local_data.size()); ++i) {
      if (local_data[i] >= 0.0) {
        local_positive.push_back(*reinterpret_cast<const uint64_t *>(&local_data[i]));
      } else {
        local_negative.push_back(*reinterpret_cast<const uint64_t *>(&local_data[i]));
      }
    }
#pragma omp critical
    {
      pozitive_copy.insert(pozitive_copy.end(), local_positive.begin(), local_positive.end());
      negative_copy.insert(negative_copy.end(), local_negative.begin(), local_negative.end());
    }
  }
}
inline void RadixSort(std::vector<uint64_t> &data, bool is_pozitive) {
  int num_bits = CalculateBits(data, is_pozitive);
#pragma omp parallel for schedule(guided, 100)
  for (int bit = 0; bit < num_bits; bit++) {
    std::vector<uint64_t> group0;
    std::vector<uint64_t> group1;
    group0.reserve(data.size());
    group1.reserve(data.size());
    for (int i = 0; i < static_cast<int>(data.size()); i++) {
      if (((data[i] >> bit) & 1) != 0U) {
        group1.push_back(data[i]);
      } else {
        group0.push_back(data[i]);
      }
    }
    data = std::move(group0);
    data.insert(data.end(), group1.begin(), group1.end());
  }
}
inline double Uint64ToDouble(uint64_t value) {
  double result = NAN;
  static_assert(sizeof(double) == sizeof(uint64_t), "Size mismatch");
  std::memcpy(&result, &value, sizeof(double));
  return result;
}
inline void FinalParse(std::vector<uint64_t> &data, int code, boost::mpi::communicator &world,
                       std::vector<bool> indicator, bool is_pozitive) {
  if (world.rank() == 0) {
    std::vector<uint64_t> local_copy_for_recv;
    for (int proc = 1; proc < world.size(); proc++) {
      if (indicator[proc - 1]) {
        world.recv(proc, code, local_copy_for_recv);
        data.insert(data.end(), local_copy_for_recv.begin(), local_copy_for_recv.end());
        local_copy_for_recv.clear();
      }
    }
    if (!data.empty()) {
      RadixSort(data, is_pozitive);
    }

  } else {
    if (!data.empty()) {
      world.send(0, code, data);
    }
  }
}
inline void WriteNegativePart(const std::vector<uint64_t> &negative_copy, std::vector<double> &output) {
  const size_t size = negative_copy.size();
  for (int i = 0; i < static_cast<int>(size); i++) {
    const size_t output_idx = size - 1 - i;
    output[output_idx] = Uint64ToDouble(negative_copy[i]);
  }
}
inline void WritePositivePart(const std::vector<uint64_t> &positive_copy, const size_t offset,
                              std::vector<double> &output) {
  const size_t size = positive_copy.size();
  for (int i = 0; i < static_cast<int>(size); i++) {
    const size_t output_idx = offset + i;
    output[output_idx] = Uint64ToDouble(positive_copy[i]);
  }
}
inline void SafeDataWrite(const std::vector<uint64_t> &negative_copy, const std::vector<uint64_t> &pozitive_copy,
                          std::vector<double> &output) {
  if (!negative_copy.empty()) {
    WriteNegativePart(negative_copy, output);
  }
  if (!pozitive_copy.empty()) {
    WritePositivePart(pozitive_copy, negative_copy.size(), output);
  }
}
}  // namespace
bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::ValidationImpl() {
  // Check equality of counts elements
  if (world_.rank() == 0) {
    return (task_data->inputs_count[0] != 0) && (task_data->inputs_count[0] == task_data->outputs_count[0]);
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::PreProcessingImpl() {
  // Init value for input and output
  input_data_.clear();
  output_.clear();
  local_data_.clear();
  if (world_.rank() == 0) {
    auto *temp_ptr = reinterpret_cast<double *>(task_data->inputs[0]);
    input_data_ = std::vector<double>(temp_ptr, temp_ptr + task_data->inputs_count[0]);
    output_.resize(task_data->inputs_count[0]);
    // std::cout << std::endl << input_data_.size() << " V NACHALE" << std::endl;
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::RunImpl() {
  std::vector<bool> is_pozitive(world_.size() - 1, false);
  std::vector<bool> is_negative(world_.size() - 1, false);
  local_data_.clear();
  if (world_.rank() == 0) {
    SendData(world_, is_pozitive, is_negative, local_data_, input_data_);
  } else {
    world_.recv(0, 0, local_data_);
  }
  std::vector<uint64_t> pozitive_copy;
  std::vector<uint64_t> negative_copy;
  ParallelParse(pozitive_copy, negative_copy, local_data_);
  if (!pozitive_copy.empty()) {
    RadixSort(pozitive_copy, true);
  }
  if (!negative_copy.empty()) {
    RadixSort(negative_copy, false);
  }
  if (world_.size() > 1) {
    FinalParse(pozitive_copy, 1, world_, is_pozitive, true);
    FinalParse(negative_copy, 2, world_, is_negative, false);
  }
  if (world_.rank() == 0) {
    SafeDataWrite(negative_copy, pozitive_copy, output_);
  }
  return true;
}

bool tsatsyn_a_radix_sort_simple_merge_all::TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0) {
    assert(output_.size() == task_data->outputs_count[0]);
    for (size_t i = 0; i < output_.size(); i++) {
      reinterpret_cast<double *>(task_data->outputs[0])[i] = output_[i];
    }
  }
  input_data_.clear();
  output_.clear();
  local_data_.clear();
  return true;
}
