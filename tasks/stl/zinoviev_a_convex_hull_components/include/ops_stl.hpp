#pragma once

#include <vector>

#include "core/task/include/task.hpp"

namespace zinoviev_a_convex_hull_components_stl {

struct Point {
  int x, y;

  bool operator<(const Point& other) const { return x < other.x || (x == other.x && y < other.y); }
  bool operator==(const Point& other) const = default;
};

class ConvexHullSTL : public ppc::core::Task {
 public:
  explicit ConvexHullSTL(ppc::core::TaskDataPtr task_data);
  bool PreProcessingImpl() noexcept override;
  bool ValidationImpl() noexcept override;
  bool RunImpl() noexcept override;
  bool PostProcessingImpl() noexcept override;

 private:
  std::vector<std::vector<Point>> components_;
  std::vector<std::vector<Point>> hulls_;

  static std::vector<Point> FindConvexHull(const std::vector<Point>& points) noexcept;
  static int Cross(const Point& o, const Point& a, const Point& b) noexcept;
  static void BFS(const int* input_data, int width, int height, int start_x, int start_y, std::vector<bool>& visited,
                  std::vector<Point>& component) noexcept;
};

}  // namespace zinoviev_a_convex_hull_components_stl