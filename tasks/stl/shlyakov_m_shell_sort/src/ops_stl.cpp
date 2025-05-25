#include "stl/shlyakov_m_shell_sort/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

namespace shlyakov_m_shell_sort_stl {

bool TestTaskSTL::PreProcessingImpl() {
  const std::size_t sz = task_data->inputs_count[0];
  auto* ptr = reinterpret_cast<int*>(task_data->inputs[0]);

  input_.assign(ptr, ptr + sz);
  output_ = input_;
  return true;
}

bool TestTaskSTL::ValidationImpl() { return task_data->inputs_count[0] == task_data->outputs_count[0]; }

bool TestTaskSTL::RunImpl() {
  int array_size = static_cast<int>(input_.size());
  if (array_size < 2) {
    return true;
  }

  unsigned int hardware_threads = ppc::util::GetPPCNumThreads();
  int num_threads = (hardware_threads > 0) ? static_cast<int>(hardware_threads) : 1;

  int sub_arr_size = (array_size + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;

  for (int i = 0; i < num_threads; ++i) {
    int left = i * sub_arr_size;
    int right = std::min(left + sub_arr_size - 1, array_size - 1);

    if (left < right) {
      threads.emplace_back([this, left, right]() { ShellSort(left, right, input_); });
    }
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  std::vector<int> buffer(input_.size());
  while (num_threads > 1) {
    int new_num_threads = (num_threads + 1) / 2;
    threads.clear();

    for (int i = 0; i < new_num_threads; ++i) {
      int left = i * 2 * sub_arr_size;
      int mid = std::min(left + sub_arr_size - 1, array_size - 1);
      int right = std::min(left + (2 * sub_arr_size) - 1, array_size - 1);

      if (mid < right) {
        threads.emplace_back([this, left, mid, right]() {
          std::vector<int> local_buffer(input_.size());
          Merge(left, mid, right, input_, local_buffer);
        });
      }
    }

    for (auto& thread : threads) {
      if (thread.joinable()) {
        thread.join();
      }
    }

    sub_arr_size *= 2;
    num_threads = new_num_threads;
  }

  output_ = input_;
  return true;
}

void Merge(int left, int mid, int right, std::vector<int>& arr, std::vector<int>& buffer) {
  int i = left;
  int j = mid + 1;
  int k = 0;
  const int merge_size = right - left + 1;

  if (buffer.size() < static_cast<std::size_t>(merge_size)) {
    buffer.resize(static_cast<std::size_t>(merge_size));
  }

  for (; i <= mid || j <= right;) {
    if (i > mid) {
      buffer[k++] = arr[j++];
    } else if (j > right) {
      buffer[k++] = arr[i++];
    } else {
      buffer[k++] = (arr[i] <= arr[j]) ? arr[i++] : arr[j++];
    }
  }

  for (size_t idx = 0; idx < static_cast<size_t>(k); ++idx) {
    arr[left + idx] = buffer[idx];
  }
}

void ShellSort(int left, int right, std::vector<int>& arr) {
  int sub_array_size = right - left + 1;
  int gap = 1;

  for (; gap <= sub_array_size / 3;) {
    gap = gap * 3 + 1;
  }

  for (; gap > 0; gap /= 3) {
    for (int k = left + gap; k <= right; ++k) {
      int current_element = arr[k];
      int j = k;

      while (j >= left + gap && arr[j - gap] > current_element) {
        arr[j] = arr[j - gap];
        j -= gap;
      }
      arr[j] = current_element;
    }
  }
}

bool TestTaskSTL::PostProcessingImpl() {
  for (std::size_t idx = 0; idx < output_.size(); ++idx) {
    reinterpret_cast<int*>(task_data->outputs[0])[idx] = output_[idx];
  }
  return true;
}

}  // namespace shlyakov_m_shell_sort_stl
