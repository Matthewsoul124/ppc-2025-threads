#include "all/bessonov_e_radix_sort_simple_merging/include/ops_all.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <functional>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace bessonov_e_radix_sort_simple_merging_all {

void TestTaskALL::ConvertDoubleToBits(const std::vector<double>& input, std::vector<uint64_t>& bits, size_t start,
                                      size_t end) {
  for (size_t i = start; i < end; ++i) {
    uint64_t b = 0;
    std::memcpy(&b, &input[i], sizeof(double));
    b ^= (-static_cast<int64_t>(b >> 63) | (1ULL << 63));
    bits[i] = b;
  }
}

void TestTaskALL::ConvertBitsToDouble(const std::vector<uint64_t>& bits, std::vector<double>& output, size_t start,
                                      size_t end) {
  for (size_t i = start; i < end; ++i) {
    uint64_t b = bits[i];
    b ^= (((b >> 63) - 1) | (1ULL << 63));
    double d = NAN;
    std::memcpy(&d, &b, sizeof(double));
    output[i] = d;
  }
}

void TestTaskALL::RadixSortPass(std::vector<uint64_t>& bits, std::vector<uint64_t>& temp, int shift) {
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

std::vector<double> TestTaskALL::Merge(const std::vector<double>& left, const std::vector<double>& right) {
  std::vector<double> result;
  result.reserve(left.size() + right.size());

  size_t i = 0;
  size_t j = 0;
  while (i < left.size() && j < right.size()) {
    if (left[i] < right[j]) {
      result.push_back(left[i++]);
    } else {
      result.push_back(right[j++]);
    }
  }

  while (i < left.size()) {
    result.push_back(left[i++]);
  }
  while (j < right.size()) {
    result.push_back(right[j++]);
  }

  return result;
}

void TestTaskALL::MergeChunks(std::deque<std::vector<double>>& chunks) {
  std::mutex merge_mutex;
  while (chunks.size() > 1) {
    std::vector<std::thread> merge_threads;
    std::deque<std::vector<double>> next;

    while (chunks.size() >= 2) {
      std::vector<double> a = std::move(chunks.front());
      chunks.pop_front();
      std::vector<double> b = std::move(chunks.front());
      chunks.pop_front();

      merge_threads.emplace_back([&next, &merge_mutex, a = std::move(a), b = std::move(b)]() {
        std::vector<double> merged = Merge(a, b);
        std::lock_guard<std::mutex> lock(merge_mutex);
        next.emplace_back(std::move(merged));
      });
    }

    if (!chunks.empty()) {
      next.push_back(std::move(chunks.front()));
      chunks.pop_front();
    }

    for (auto& t : merge_threads) {
      t.join();
    }
    chunks = std::move(next);
  }
}

void TestTaskALL::HandleSingleProcess() {
  const size_t n = input_.size();
  const size_t threads = std::max<size_t>(1, ppc::util::GetPPCNumThreads());
  const size_t block = (n + threads - 1) / threads;

  std::vector<uint64_t> bits(n);
  std::vector<uint64_t> temp(n);

  {
    std::vector<std::thread> th;
    for (size_t i = 0; i < threads; ++i) {
      size_t start = i * block;
      size_t end = std::min(start + block, n);
      if (start < end) {
        th.emplace_back(ConvertDoubleToBits, std::cref(input_), std::ref(bits), start, end);
      }
    }
    for (auto& t : th) {
      t.join();
    }
  }

  for (int pass = 0; pass < static_cast<int>(sizeof(uint64_t)); ++pass) {
    RadixSortPass(bits, temp, pass * 8);
  }

  output_.resize(n);
  {
    std::vector<std::thread> th;
    for (size_t i = 0; i < threads; ++i) {
      size_t start = i * block;
      size_t end = std::min(start + block, n);
      if (start < end) {
        th.emplace_back(ConvertBitsToDouble, std::cref(bits), std::ref(output_), start, end);
      }
    }
    for (auto& t : th) {
      t.join();
    }
  }
}

void TestTaskALL::HandleParallelProcess() {
  const int rank = world_.rank();
  const int size = world_.size();
  size_t n = input_.size();

  world_.barrier();

  std::vector<int> sendcounts(size);
  std::vector<int> displs(size);
  for (int i = 0; i < size; ++i) {
    sendcounts[i] = static_cast<int>(n / size) + (i < static_cast<int>(n % size) ? 1 : 0);
    displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
  }

  std::vector<double> local_input(sendcounts[rank]);
  MPI_Scatterv(input_.data(), sendcounts.data(), displs.data(), MPI_DOUBLE, local_input.data(), sendcounts[rank],
               MPI_DOUBLE, 0, MPI_COMM_WORLD);

  size_t local_n = local_input.size();
  const size_t threads = std::max<size_t>(1, ppc::util::GetPPCNumThreads());
  const size_t block = (local_n + threads - 1) / threads;

  std::vector<uint64_t> bits(local_n);
  std::vector<uint64_t> temp(local_n);

  {
    std::vector<std::thread> th;
    for (size_t i = 0; i < threads; ++i) {
      size_t start = i * block;
      size_t end = std::min(start + block, local_n);
      if (start < end) {
        th.emplace_back(ConvertDoubleToBits, std::cref(local_input), std::ref(bits), start, end);
      }
    }
    for (auto& t : th) {
      t.join();
    }
  }

  for (int pass = 0; pass < static_cast<int>(sizeof(uint64_t)); ++pass) {
    RadixSortPass(bits, temp, pass * 8);
  }

  std::vector<double> local_sorted(local_n);
  {
    std::vector<std::thread> th;
    for (size_t i = 0; i < threads; ++i) {
      size_t start = i * block;
      size_t end = std::min(start + block, local_n);
      if (start < end) {
        th.emplace_back(ConvertBitsToDouble, std::cref(bits), std::ref(local_sorted), start, end);
      }
    }
    for (auto& t : th) {
      t.join();
    }
  }

  world_.barrier();

  if (rank == 0) {
    output_.resize(n);
  }

  MPI_Gatherv(local_sorted.data(), static_cast<int>(local_n), MPI_DOUBLE, output_.data(), sendcounts.data(),
              displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0 && size > 1) {
    std::deque<std::vector<double>> chunks;
    for (int i = 0; i < size; ++i) {
      ptrdiff_t start = displs[i];
      ptrdiff_t count = sendcounts[i];
      chunks.emplace_back(output_.begin() + start, output_.begin() + start + count);
    }
    MergeChunks(chunks);
    output_ = std::move(chunks.front());
  }
}

bool TestTaskALL::PreProcessingImpl() {
  if (world_.rank() == 0) {
    input_.resize(task_data->inputs_count[0]);
    output_.resize(task_data->outputs_count[0]);

    auto* in_ptr = reinterpret_cast<double*>(task_data->inputs[0]);
    std::copy(in_ptr, in_ptr + task_data->inputs_count[0], input_.begin());
  }
  size_t input_size = input_.size();
  boost::mpi::broadcast(world_, input_size, 0);
  if (world_.rank() != 0) {
    input_.resize(input_size);
  }
  return true;
}

bool TestTaskALL::ValidationImpl() {
  bool local_valid = true;

  if (world_.rank() == 0) {
    local_valid = !(task_data->inputs.empty() || task_data->outputs.empty());
    local_valid &= task_data->inputs[0] != nullptr && task_data->outputs[0] != nullptr;

    if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
      local_valid = false;
    } else {
      local_valid &= task_data->inputs_count[0] > 0;
      local_valid &= task_data->inputs_count[0] == task_data->outputs_count[0];
      local_valid &= task_data->inputs_count[0] <= static_cast<size_t>(std::numeric_limits<int>::max());
    }
  }

  boost::mpi::broadcast(world_, local_valid, 0);
  return local_valid;
}

bool TestTaskALL::RunImpl() {
  const int rank = world_.rank();
  const int size = world_.size();

  size_t n = 0;
  if (rank == 0) {
    n = input_.size();
  }
  world_.barrier();
  boost::mpi::broadcast(world_, n, 0);

  if (n == 0) {
    return true;
  }

  if (size == 1) {
    HandleSingleProcess();
  } else {
    HandleParallelProcess();
  }

  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  if (world_.rank() == 0 && !output_.empty()) {
    auto* out_ptr = reinterpret_cast<double*>(task_data->outputs[0]);
    std::ranges::copy(output_, out_ptr);
  }
  return true;
}
}  // namespace bessonov_e_radix_sort_simple_merging_all