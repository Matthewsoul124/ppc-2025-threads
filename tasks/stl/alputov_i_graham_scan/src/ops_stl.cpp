#include "stl/alputov_i_graham_scan/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <ranges>
#include <thread>
#include <tuple>
#include <vector>

#include "core/util/include/util.hpp"

namespace alputov_i_graham_scan_stl {

Point::Point(double x, double y) : x(x), y(y) {}

bool Point::operator<(const Point& other) const { return std::tie(y, x) < std::tie(other.y, other.x); }

bool Point::operator==(const Point& other) const { return std::tie(x, y) == std::tie(other.x, other.y); }

bool Point::operator!=(const Point& other) const { return !(*this == other); }

bool TestTaskSTL::PreProcessingImpl() {
  auto* input_ptr = reinterpret_cast<Point*>(task_data->inputs[0]);
  input_points_ = std::vector<Point>(input_ptr, input_ptr + task_data->inputs_count[0]);
  return true;
}

bool TestTaskSTL::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs_count[0] <= task_data->outputs_count[0]);
}

double TestTaskSTL::Cross(const Point& o, const Point& a, const Point& b) {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

Point TestTaskSTL::FindPivot() const {
  auto comparator = [](const Point& a, const Point& b) { return a < b; };
  return *std::ranges::min_element(input_points_, comparator);
}

void TestTaskSTL::RemoveDuplicates(std::vector<Point>& points) {
  auto result = std::ranges::unique(points);
  points.erase(result.end(), points.end());
}

bool TestTaskSTL::CompareAngles(const Point& first_point, const Point& second_point, const Point& pivot_point) {
  const auto first_dx = first_point.x - pivot_point.x;
  const auto first_dy = first_point.y - pivot_point.y;
  const auto second_dx = second_point.x - pivot_point.x;
  const auto second_dy = second_point.y - pivot_point.y;

  const double cross_product = (first_dx * second_dy) - (first_dy * second_dx);
  constexpr double kEpsilon = 1e-10;

  if (std::abs(cross_product) < kEpsilon) {
    const auto first_distance_squared = (first_dx * first_dx) + (first_dy * first_dy);
    const auto second_distance_squared = (second_dx * second_dx) + (second_dy * second_dy);
    return first_distance_squared < second_distance_squared;
  }

  return cross_product > 0;
}

std::vector<Point> TestTaskSTL::SortPoints(const Point& pivot) {
  std::vector<Point> points;
  points.reserve(input_points_.size());
  for (const auto& p : input_points_) {
    if (!(p == pivot)) {
      points.push_back(p);
    }
  }

  ParallelSort(points, [&](const Point& a, const Point& b) { return TestTaskSTL::CompareAngles(a, b, pivot); });
  TestTaskSTL::RemoveDuplicates(points);
  return points;
}

std::vector<Point> TestTaskSTL::BuildHull(const std::vector<Point>& sorted_points, const Point& pivot) {
  std::vector<Point> hull;
  hull.reserve(sorted_points.size() + 1);
  hull.push_back(pivot);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && TestTaskSTL::Cross(hull[hull.size() - 2], hull.back(), p) < 1e-10) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  while (hull.size() >= 3 && TestTaskSTL::Cross(hull[hull.size() - 2], hull.back(), hull[0]) < 1e-10) {
    hull.pop_back();
  }

  return hull;
}

bool TestTaskSTL::RunImpl() {
  const Point pivot = FindPivot();
  auto sorted_points = SortPoints(pivot);

  if (sorted_points.empty()) {
    convex_hull_ = {pivot};
    return true;
  }

  convex_hull_ = TestTaskSTL::BuildHull(sorted_points, pivot);
  return true;
}

bool TestTaskSTL::PostProcessingImpl() {
  auto* output_ptr = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(convex_hull_, output_ptr);
  return true;
}

const std::vector<Point>& TestTaskSTL::GetConvexHull() const { return convex_hull_; }

void TestTaskSTL::ParallelSort(std::vector<Point>& points, auto comp) {
  const size_t n = points.size();
  const size_t num_threads = std::min(n, static_cast<size_t>(ppc::util::GetPPCNumThreads()));

  if (num_threads <= 1) {
    std::sort(points.begin(), points.end(), comp);
    return;
  }

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::vector<size_t> chunk_offsets(num_threads + 1);

  for (size_t i = 0; i <= num_threads; ++i) {
    chunk_offsets[i] = (i * n) / num_threads;
  }

  using DifferenceType = typename std::vector<Point>::difference_type;

  for (size_t i = 0; i < num_threads; ++i) {
    threads.emplace_back([&points, &comp, start_offset = chunk_offsets[i], end_offset = chunk_offsets[i + 1]]() {
      auto start_it = std::next(points.begin(), static_cast<DifferenceType>(start_offset));
      auto end_it = std::next(points.begin(), static_cast<DifferenceType>(end_offset));
      std::sort(start_it, end_it, comp);
    });
  }

  for (auto& t : threads) {
    t.join();
  }

  for (size_t i = 1; i < num_threads; ++i) {
    auto mid_it = std::next(points.begin(), static_cast<DifferenceType>(chunk_offsets[i]));
    auto end_it = std::next(points.begin(), static_cast<DifferenceType>(chunk_offsets[i + 1]));
    std::inplace_merge(points.begin(), mid_it, end_it, comp);
  }
}

}  // namespace alputov_i_graham_scan_stl