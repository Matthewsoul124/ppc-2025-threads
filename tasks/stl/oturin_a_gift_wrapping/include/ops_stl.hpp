#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace oturin_a_gift_wrapping_stl {

struct Coord {
  int x, y;
  bool operator==(const Coord o) const { return (x == o.x && y == o.y); }
  bool operator!=(const Coord o) const { return x != o.x || y != o.y; }
};

double Distance(Coord a, Coord b);

// Angle Between Three Points
double ABTP(Coord a, Coord b, Coord c);

// Angle Between Three Points for leftmost point
double ABTP(Coord a, Coord c);

class TestTaskSTL : public ppc::core::Task {
 public:
  explicit TestTaskSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

#ifdef __linux__
  void SearchThreadP(std::size_t start, std::size_t end, std::pair<int, double> &thread_result);
#endif
  std::pair<int, double> SearchThread(std::size_t start, std::size_t end);

 private:
  std::vector<Coord> input_, output_;
  int n_;

  int FindMostLeft();
  void FindSecondPoint(int start_index, int &search_index);
  void PointSearch(double t, double &line_angle, std::size_t &search_index, std::size_t i);
};

}  // namespace oturin_a_gift_wrapping_stl
