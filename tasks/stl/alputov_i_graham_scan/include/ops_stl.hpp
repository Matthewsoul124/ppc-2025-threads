#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace alputov_i_graham_scan_stl {

struct Point {
  double x, y;
  Point(double x = 0, double y = 0);
  bool operator<(const Point& other) const;
  bool operator==(const Point& other) const;
  bool operator!=(const Point& other) const;
};

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  [[nodiscard]] const std::vector<Point>& GetConvexHull() const;

 private:
  std::vector<Point> input_points_;
  std::vector<Point> convex_hull_;

  [[nodiscard]] Point FindPivot() const;
  [[nodiscard]] std::vector<Point> SortPoints(const Point& pivot);
  static std::vector<Point> BuildHull(const std::vector<Point>& sorted_points, const Point& pivot);
  static void RemoveDuplicates(std::vector<Point>& points);
  [[nodiscard]] static bool CompareAngles(const Point& first_point, const Point& second_point,
                                          const Point& pivot_point);

  static double Cross(const Point& o, const Point& a, const Point& b);

  void ParallelSort(std::vector<Point>& points, auto comp);
};

}  // namespace alputov_i_graham_scan_stl