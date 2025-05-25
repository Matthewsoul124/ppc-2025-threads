#include "all/sotskov_a_shell_sorting_with_simple_merging/include/ops_all.hpp"

#include <algorithm>
#include <boost/mpi/collectives/gatherv.hpp>
#include <boost/mpi/collectives/scatterv.hpp>
#include <cstddef>
#include <thread>
#include <vector>

#include "boost/mpi/collectives/broadcast.hpp"
#include "core/util/include/util.hpp"

void sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::CalculateDistribution(
    int total, std::vector<int>& counts, std::vector<int>& displs) const {
  counts.resize(size_);
  displs.resize(size_);
  int base_size = total / size_;
  int remainder = total % size_;
  for (int i = 0; i < size_; ++i) {
    counts[i] = (i < remainder) ? base_size + 1 : base_size;
    displs[i] = (i == 0) ? 0 : displs[i - 1] + counts[i - 1];
  }
}

void sotskov_a_shell_sorting_with_simple_merging_all::ShellSort(std::vector<int>& arr, size_t left, size_t right) {
  size_t array_size = right - left + 1;
  size_t gap = 1;
  while (gap < array_size / 3) {
    gap = gap * 3 + 1;
  }

  while (gap > 0) {
    for (size_t i = left + gap; i <= right; ++i) {
      int current_element = arr[i];
      size_t j = i;
      while (j >= left + gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
    gap /= 3;
  }
}

void sotskov_a_shell_sorting_with_simple_merging_all::ParallelMerge(std::vector<int>& arr, size_t left, size_t mid,
                                                                    size_t right) {
  std::vector<int> temp(right - left + 1);
  size_t i = left;
  size_t j = mid + 1;
  size_t k = 0;

  while (i <= mid && j <= right) {
    temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
  }
  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= right) {
    temp[k++] = arr[j++];
  }
  std::ranges::copy(temp, arr.begin() + static_cast<std::ptrdiff_t>(left));
}

void sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(std::vector<int>& arr) {
  if (arr.empty() || arr.size() == 1) {
    return;
  }

  int array_size = static_cast<int>(arr.size());
  int num_threads = std::min(ppc::util::GetPPCNumThreads(), static_cast<int>(std::thread::hardware_concurrency()));
  int chunk_size = std::max(1, (array_size + num_threads - 1) / num_threads);

  std::vector<std::thread> workers;
  for (int i = 0; i < num_threads; ++i) {
    const int left = i * chunk_size;
    const int right = std::min(left + chunk_size - 1, array_size - 1);
    if (left < right) {
      workers.emplace_back([&arr, left, right]() { ShellSort(arr, left, right); });
    }
  }

  for (auto& worker : workers) {
    worker.join();
  }

  for (int size = chunk_size; size < array_size; size *= 2) {
    for (int i = 0; i < array_size; i += 2 * size) {
      const int left = i;
      const int mid = std::min(i + size - 1, array_size - 1);
      const int right = std::min(i + (2 * size) - 1, array_size - 1);
      if (mid < right) {
        ParallelMerge(arr, left, mid, right);
      }
    }
  }
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::PreProcessingImpl() {
  int total = 0;
  if (rank_ == 0) {
    total = static_cast<int>(task_data->inputs_count[0]);
  }
  boost::mpi::broadcast(world_, total, 0);

  if (total == 0) {
    return true;
  }

  std::vector<int> global_data;
  if (rank_ == 0) {
    global_data.resize(total);
    auto* src = reinterpret_cast<int*>(task_data->inputs[0]);
    std::copy(src, src + total, global_data.begin());
  }

  std::vector<int> counts;
  std::vector<int> displs;
  CalculateDistribution(total, counts, displs);
  input_.resize(counts[rank_]);

  boost::mpi::scatterv(world_, global_data.data(), counts, displs, input_.data(), counts[rank_], 0);

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::PostProcessingImpl() {
  int total = 0;
  if (rank_ == 0) {
    total = static_cast<int>(task_data->inputs_count[0]);
  }
  boost::mpi::broadcast(world_, total, 0);

  if (total == 0) {
    return true;
  }

  std::vector<int> counts;
  std::vector<int> displs;
  CalculateDistribution(total, counts, displs);

  std::vector<int> result;
  if (rank_ == 0) {
    result.resize(total);
  }

  boost::mpi::gatherv(world_, input_.data(), static_cast<int>(input_.size()), (rank_ == 0) ? result.data() : nullptr,
                      counts, displs, 0);

  if (rank_ == 0) {
    for (int step = 1; step < size_; step *= 2) {
      for (int i = 0; i < size_ - step; i += 2 * step) {
        int left = displs[i];
        int mid = displs[i + step];
        int right = (i + (2 * step) < size_) ? displs[i + (2 * step)] : total;
        std::inplace_merge(result.begin() + left, result.begin() + mid, result.begin() + right);
      }
    }

    if (task_data->outputs[0] != nullptr) {
      auto* dst = reinterpret_cast<int*>(task_data->outputs[0]);
      std::ranges::copy(result.begin(), result.end(), dst);
    }
  }

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::ValidationImpl() {
  if (rank_ != 0) {
    return true;
  }

  if (task_data->inputs_count.empty() || task_data->outputs_count.empty() ||
      task_data->inputs_count[0] != task_data->outputs_count[0]) {
    return false;
  }

  auto* output = reinterpret_cast<int*>(task_data->outputs[0]);
  size_t size = task_data->outputs_count[0];
  for (size_t i = 1; i < size; i++) {
    if (output[i] < output[i - 1]) {
      return false;
    }
  }

  return true;
}

bool sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL::RunImpl() {
  ShellSortWithSimpleMerging(input_);
  return true;
}