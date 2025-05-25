#include "tbb/malyshev_v_radix_sort/include/ops_tbb.hpp"

#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace malyshev_v_radix_sort_tbb {

namespace {
union DoubleConverter {
  double d;
  uint64_t u;
};

uint64_t ConvertDoubleToUInt64(double d) {
  DoubleConverter converter;
  converter.d = d;
  return (converter.u & (1ULL << 63)) != 0U ? ~converter.u : (converter.u ^ (1ULL << 63));
}

void RadixSort(std::vector<double>& data, int exp) {
  const size_t data_size = data.size();
  std::vector<std::atomic<int>> count(256);
  for (auto& c : count) {
    c.store(0, std::memory_order_relaxed);
  }

  std::vector<double> output(data_size);

  tbb::parallel_for(tbb::blocked_range<size_t>(0, data_size), [&](const tbb::blocked_range<size_t>& r) {
    std::vector<int> local_count(256, 0);
    for (size_t i = r.begin(); i < r.end(); ++i) {
      uint64_t bits = ConvertDoubleToUInt64(data[i]);
      int digit = static_cast<int>((bits >> (exp * 8)) & 0xFF);
      local_count[digit]++;
    }
    for (int j = 0; j < 256; ++j) {
      count[j].fetch_add(local_count[j], std::memory_order_relaxed);
    }
  });

  int prev = count[0].load(std::memory_order_relaxed);
  for (int i = 1; i < 256; ++i) {
    int current = count[i].load(std::memory_order_relaxed);
    count[i].store(prev + current, std::memory_order_relaxed);
    prev += current;
  }

  for (size_t i = data_size; i-- > 0;) {
    uint64_t bits = ConvertDoubleToUInt64(data[i]);
    int digit = static_cast<int>((bits >> (exp * 8)) & 0xFF);
    output[--count[digit]] = data[i];
  }

  data = std::move(output);
}

void ProcessNumbers(std::vector<double>& numbers) {
  for (int exp = 0; exp < 8; ++exp) {
    RadixSort(numbers, exp);
  }
}

}  // namespace

bool SortTBB::PreProcessingImpl() {
  input_ = std::vector<double>(reinterpret_cast<double*>(task_data->inputs[0]),
                               reinterpret_cast<double*>(task_data->inputs[0]) + task_data->inputs_count[0]);
  output_.resize(task_data->outputs_count[0]);
  return true;
}

bool SortTBB::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool SortTBB::RunImpl() {
  if (input_.empty()) {
    output_ = input_;
    return true;
  }

  std::vector<double> negative;
  std::vector<double> positive;

  for (double num : input_) {
    if (num < 0) {
      negative.push_back(num);
    } else {
      positive.push_back(num);
    }
  }

  oneapi::tbb::task_arena arena;
  arena.execute([&] {
    tbb::task_group tg;
    tg.run([&] { ProcessNumbers(negative); });
    tg.run([&] { ProcessNumbers(positive); });
    tg.wait();
  });

  output_.clear();
  output_.insert(output_.end(), negative.begin(), negative.end());
  output_.insert(output_.end(), positive.begin(), positive.end());

  return true;
}

bool SortTBB::PostProcessingImpl() {
  std::ranges::copy(output_, reinterpret_cast<double*>(task_data->outputs[0]));
  return true;
}

}  // namespace malyshev_v_radix_sort_tbb