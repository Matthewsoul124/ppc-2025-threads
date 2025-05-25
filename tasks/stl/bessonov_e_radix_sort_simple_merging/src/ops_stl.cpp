#include "stl/bessonov_e_radix_sort_simple_merging/include/ops_stl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace bessonov_e_radix_sort_simple_merging_stl {

void TestTaskSTL::ConvertDoubleToBits(const std::vector<double>& input, std::vector<uint64_t>& bits, size_t start,
                                      size_t end) {
  for (size_t i = start; i < end; ++i) {
    uint64_t b = 0;
    std::memcpy(&b, &input[i], sizeof(double));
    b ^= (-static_cast<int64_t>(b >> 63) | (1ULL << 63));
    bits[i] = b;
  }
}

void TestTaskSTL::ConvertBitsToDouble(const std::vector<uint64_t>& bits, std::vector<double>& output, size_t start,
                                      size_t end) {
  for (size_t i = start; i < end; ++i) {
    uint64_t b = bits[i];
    b ^= (((b >> 63) - 1) | (1ULL << 63));
    double d = NAN;
    std::memcpy(&d, &b, sizeof(double));
    output[i] = d;
  }
}

void TestTaskSTL::RadixSortPass(std::vector<uint64_t>& bits, std::vector<uint64_t>& temp, int shift) {
  constexpr int kRadix = 256;
  const size_t n = bits.size();
  std::array<size_t, kRadix> count{};

  for (size_t i = 0; i < n; ++i) {
    count[(bits[i] >> shift) & 0xFF]++;
  }

  size_t total = 0;
  for (int i = 0; i < kRadix; ++i) {
    size_t old_count = count[i];
    count[i] = total;
    total += old_count;
  }

  for (size_t i = 0; i < n; ++i) {
    uint8_t digit = (bits[i] >> shift) & 0xFF;
    temp[count[digit]++] = bits[i];
  }

  bits.swap(temp);
}
}  // namespace bessonov_e_radix_sort_simple_merging_stl

bool bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL::PreProcessingImpl() {
  input_.resize(task_data->inputs_count[0]);
  output_.resize(task_data->outputs_count[0]);

  auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
  std::copy(in_ptr, in_ptr + task_data->inputs_count[0], input_.begin());

  return true;
}

bool bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL::ValidationImpl() {
  if (task_data->inputs.empty() || task_data->outputs.empty()) {
    return false;
  }

  if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
    return false;
  }

  if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
    return false;
  }

  if (task_data->inputs_count[0] == 0) {
    return false;
  }

  if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }

  if (task_data->inputs_count[0] > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }

  return true;
}

bool bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL::RunImpl() {
  const size_t n = input_.size();
  if (n == 0) {
    return true;
  }

  std::vector<uint64_t> bits(n);
  std::vector<uint64_t> temp(n);

  size_t num_threads = ppc::util::GetPPCNumThreads();
  num_threads = std::max<size_t>(1, num_threads);
  const size_t block_size = (n + num_threads - 1) / num_threads;
  {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
      size_t start = i * block_size;
      size_t end = std::min(start + block_size, n);
      if (start >= n) {
        break;
      }
      threads.emplace_back(ConvertDoubleToBits, std::cref(input_), std::ref(bits), start, end);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  constexpr int kPasses = sizeof(uint64_t);
  for (int pass = 0; pass < kPasses; ++pass) {
    RadixSortPass(bits, temp, pass * 8);
  }

  {
    std::vector<std::thread> threads;
    for (size_t i = 0; i < num_threads; ++i) {
      size_t start = i * block_size;
      size_t end = std::min(start + block_size, n);
      if (start >= n) {
        break;
      }
      threads.emplace_back(ConvertBitsToDouble, std::cref(bits), std::ref(output_), start, end);
    }
    for (auto& t : threads) {
      t.join();
    }
  }

  return true;
}

bool bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL::PostProcessingImpl() {
  auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
  std::ranges::copy(output_.begin(), output_.end(), out_ptr);
  return true;
}