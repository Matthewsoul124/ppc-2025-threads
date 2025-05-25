#pragma once

#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace pikarychev_i_hoare_sort_simple_merge {

template <class T>
class HoareSeq : public ppc::core::Task {
 public:
  explicit HoareSeq(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool ValidationImpl() override {
    const auto input_size = task_data->inputs_count[0];
    const auto output_size = task_data->outputs_count[0];
    return input_size == output_size;
  }

  bool PreProcessingImpl() override {
    input_.assign(reinterpret_cast<T*>(task_data->inputs[0]),
                  reinterpret_cast<T*>(task_data->inputs[0]) + task_data->inputs_count[0]);
    res_.resize(input_.size());
    reverse_ = *reinterpret_cast<bool*>(task_data->inputs[1]);
    return true;
  }

  bool RunImpl() override {
    const auto size = input_.size();
    std::ranges::copy_n(input_.begin(), size, res_.begin());

    DoSort(res_, 0, size - 1,
           [reverse = reverse_](const auto& a, const auto& b) { return reverse ? (a < b) : (a > b); });

    return true;
  }

  bool PostProcessingImpl() override {
    std::ranges::copy_n(res_.begin(), res_.size(), reinterpret_cast<T*>(task_data->outputs[0]));
    return true;
  }

 private:
  static int Partition(std::vector<T>& block, int low, int high, const std::function<bool(const T&, const T&)>& comp) {
    int e = low - 1;
    for (int j = low; j <= high - 1; j++) {
      if (!comp(block[j], block[high])) {
        continue;
      }

      e++;
      std::swap(block[e], block[j]);
    }
    std::swap(block[e + 1], block[high]);
    return ++e;
  };
  static void DoSort(std::vector<T>& arr, int low, int high, const std::function<bool(const T&, const T&)>& comp) {
    if (low >= high) {
      return;
    }
    const auto border = Partition(arr, low, high, comp);
    DoSort(arr, low, border - 1, comp);
    DoSort(arr, border + 1, high, comp);
  }

  bool reverse_;
  std::vector<T> input_;
  std::vector<T> res_;
};

}  // namespace pikarychev_i_hoare_sort_simple_merge