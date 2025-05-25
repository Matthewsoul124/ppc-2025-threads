#pragma once

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

namespace ermolaev_v_graham_scan_stl {

constexpr size_t kMinInputPoints = 3;
constexpr size_t kMinStackPoints = 2;

class Point {
 public:
  int x;
  int y;

  Point(int x_value, int y_value) : x(x_value), y(y_value) {}
  Point() : x(0), y(0) {}
  bool operator==(const Point& rhs) const { return y == rhs.y && x == rhs.x; }
  bool operator!=(const Point& rhs) const { return !(*this == rhs); }
  auto operator<=>(const Point& rhs) const {
    if (auto cmp = y <=> rhs.y; cmp != 0) {
      return cmp;
    }
    return x <=> rhs.x;
  }
};

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<Point> input_, output_;

  static inline int CrossProduct(const Point& p1, const Point& p2, const Point& p3);
  bool CheckGrahamNecessaryConditions();
  void GrahamScan();
  bool IsAllCollinear();
  bool IsAllSame();

  template <typename Iterator, typename Comparator>
  void ParallelSort(Iterator begin, Iterator end, Comparator comp);
};

template <typename Iterator, typename Comparator>
void ermolaev_v_graham_scan_stl::TestTaskSTL::ParallelSort(Iterator begin, Iterator end, Comparator comp) {
  const size_t n = std::distance(begin, end);

  int num_threads = ppc::util::GetPPCNumThreads();
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::vector<size_t> chunk_boundaries(num_threads + 1);
  for (size_t i = 0; i <= static_cast<size_t>(num_threads); i++) {
    auto boundary = std::max((i * n) / num_threads, i);
    if (boundary > n) {
      boundary = n;
    }

    chunk_boundaries[i] = boundary;
  }

  for (int i = 0; i < num_threads; i++) {
    threads.emplace_back([=]() { std::sort(begin + chunk_boundaries[i], begin + chunk_boundaries[i + 1], comp); });
  }
  for (auto& t : threads) {
    t.join();
  }
  threads.clear();

  std::vector<typename std::iterator_traits<Iterator>::value_type> buffer(n);

  Iterator current_src_it = begin;
  Iterator current_dst_it = buffer.begin();
  bool result_in_original_array = true;

  std::vector<size_t> active_boundaries = chunk_boundaries;

  while (active_boundaries.size() - 1 > 1) {
    std::vector<size_t> next_boundaries;
    next_boundaries.push_back(0);
    size_t num_current_segments = active_boundaries.size() - 1;
    threads.reserve((num_current_segments + 1) / 2);

    for (size_t i = 0; i < num_current_segments / 2; ++i) {
      Iterator s1_begin = current_src_it + active_boundaries[2 * i];
      Iterator s1_end = current_src_it + active_boundaries[2 * i + 1];
      Iterator s2_begin = current_src_it + active_boundaries[2 * i + 1];
      Iterator s2_end = current_src_it + active_boundaries[2 * i + 2];
      Iterator d_begin = current_dst_it + active_boundaries[2 * i];

      threads.emplace_back([=]() { std::merge(s1_begin, s1_end, s2_begin, s2_end, d_begin, comp); });
      next_boundaries.push_back(active_boundaries[2 * i + 2]);
    }

    if (num_current_segments % 2 != 0) {
      size_t last_segment_original_idx = num_current_segments - 1;
      Iterator s_begin = current_src_it + active_boundaries[last_segment_original_idx];
      Iterator s_end = current_src_it + active_boundaries[last_segment_original_idx + 1];
      Iterator d_begin = current_dst_it + active_boundaries[last_segment_original_idx];

      threads.emplace_back([=]() { std::copy(s_begin, s_end, d_begin); });
      next_boundaries.push_back(active_boundaries[last_segment_original_idx + 1]);
    }

    for (auto& t : threads) {
      t.join();
    }
    threads.clear();

    active_boundaries = next_boundaries;
    std::swap(current_src_it, current_dst_it);
    result_in_original_array = !result_in_original_array;
  }

  if (!result_in_original_array) {
    std::copy(current_src_it, current_src_it + n, begin);
  }
}

}  // namespace ermolaev_v_graham_scan_stl
