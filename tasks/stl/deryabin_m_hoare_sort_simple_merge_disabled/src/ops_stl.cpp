#include "stl/deryabin_m_hoare_sort_simple_merge/include/ops_stl.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

void deryabin_m_hoare_sort_simple_merge_stl::HoareSort(std::vector<double>& a, size_t first, size_t last) {
  if (first >= last) {
    return;
  }
  const double x = a[(first + last) >> 1];
  double* pi = &a[first];
  double* pj = &a[last];
  do {
    while (*pi < x) {
      pi++;
    }
    while (*pj > x) {
      pj--;
    }
    const double tmp = *pi;
    *pi = *pj;
    *pj = tmp;
  } while (pi < pj);
  const size_t j = pj - a.data();
  const size_t i = pi - a.data();
  HoareSort(a, first, j);
  HoareSort(a, i + 1, last);
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::RunImpl() {
  size_t count = 0;
  while (count != chunk_count_) {
    HoareSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
    count++;
  }
  size_t chunk_count = chunk_count_;
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_) - 1); i++) {
    for (size_t j = 0; j < chunk_count; j++) {
      std::inplace_merge(input_array_A_.begin() + static_cast<long>(j * min_chunk_size_ << (i + 1)),
                         input_array_A_.begin() + static_cast<long>(((j << 1 | 1) * (min_chunk_size_ << i))),
                         input_array_A_.begin() + static_cast<long>((j + 1) * min_chunk_size_ << (i + 1)));
      chunk_count--;
    }
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSequential::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::PreProcessingImpl() {
  input_array_A_ = reinterpret_cast<std::vector<double>*>(task_data->inputs[0])[0];
  dimension_ = task_data->inputs_count[0];
  chunk_count_ = task_data->inputs_count[1];
  min_chunk_size_ = dimension_ / chunk_count_;
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::ValidationImpl() {
  return static_cast<unsigned short>(task_data->inputs_count[0]) > 2 &&
         static_cast<unsigned short>(task_data->inputs_count[1]) >= 2 &&
         task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::RunImpl() {
  const size_t num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> workers;
  workers.reserve(num_threads);
  if (chunk_count_ < num_threads) {
    // Увеличиваем число кусочков до ближайшей степени двойки >= num_threads,
    // чтобы эффективно загрузить все доступные потоки
    chunk_count_ = 1ULL << std::bit_width(num_threads - 1);
    min_chunk_size_ = dimension_ / chunk_count_;
  }
  auto parallel_for = [&](size_t start, size_t end, auto&& func) {
    const size_t num_chunk_per_thread = (end - start) / num_threads;
    for (size_t i = 0; i < num_threads - 1; ++i) {
      workers.emplace_back([=, &func] {
        for (size_t j = start + (i * num_chunk_per_thread); j < start + (i + 1) * num_chunk_per_thread; ++j) {
          func(j);
        }
      });
    }
    workers.emplace_back([=, &func] {
      for (size_t j = start + ((num_threads - 1) * num_chunk_per_thread); j < end; ++j) {
        func(j);
      }
    });
    for (auto& worker : workers) {
      worker.join();
    }
    workers.clear();
  };
  parallel_for(0, chunk_count_, [this](size_t count) {
    HoareSort(input_array_A_, count * min_chunk_size_, ((count + 1) * min_chunk_size_) - 1);
  });
  for (size_t i = 0; i < static_cast<size_t>(std::bit_width(chunk_count_) -
                                             1);  // Вычисялем сколько уровней слияния потребуется как логарифм по
                                                  // основанию 2 от числа частей chunk_count_
       ++i) {  // На каждом уровне сливаются пары соседних блоков размером min_chunk_size_ × 2^i
    parallel_for(
        0, chunk_count_ >> (i + 1), [this, i](size_t j) {  // Распределение слияний между потоками на каждом уровне
          std::inplace_merge(
              input_array_A_.begin() +
                  static_cast<long>(j * min_chunk_size_ << (i + 1)),  // Вызов std::inplace_merge для слияния двух
                                                                      // подмассивов, отсортированных функцией HoareSort
              input_array_A_.begin() + static_cast<long>(((j << 1 | 1) * (min_chunk_size_ << i))),
              input_array_A_.begin() + static_cast<long>((j + 1) * min_chunk_size_ << (i + 1)));
        });
  }
  return true;
}

bool deryabin_m_hoare_sort_simple_merge_stl::HoareSortTaskSTL::PostProcessingImpl() {
  reinterpret_cast<std::vector<double>*>(task_data->outputs[0])[0] = input_array_A_;
  return true;
}
