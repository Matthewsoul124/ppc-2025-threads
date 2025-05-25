#include "stl/ermolaev_v_graham_scan/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <vector>

#include "core/util/include/util.hpp"

int ermolaev_v_graham_scan_stl::TestTaskSTL::CrossProduct(const Point &p1, const Point &p2, const Point &p3) {
  return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p3.x - p1.x) * (p2.y - p1.y));
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::IsAllCollinear() {
  if (input_.size() <= 2) {
    return true;
  }

  Point p1 = input_[0];
  Point p2;
  bool found_second_distinct = false;

  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i] != p1) {
      p2 = input_[i];
      found_second_distinct = true;
      break;
    }
  }

  if (!found_second_distinct) {
    return true;
  }

  return std::all_of(input_.begin(), input_.end(),
                     [&](const auto &p_check) { return CrossProduct(p1, p2, p_check) == 0; });
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::IsAllSame() {
  if (input_.size() <= 1) {
    return true;
  }

  const Point &first = input_[0];
  return !any_of(input_.begin() + 1, input_.end(), [&](const Point &p) { return p != first; });
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::CheckGrahamNecessaryConditions() {
  if (input_.size() < kMinInputPoints) {
    return false;
  }

  return !IsAllSame() && !IsAllCollinear();
}

void ermolaev_v_graham_scan_stl::TestTaskSTL::GrahamScan() {
  output_.clear();
  output_.emplace_back(input_[0]);
  output_.emplace_back(input_[1]);

  Point p1;
  Point p2;
  Point p3;
  for (size_t i = kMinStackPoints; i < input_.size(); i++) {
    while (output_.size() >= kMinStackPoints) {
      p1 = output_[output_.size() - 2];
      p2 = output_[output_.size() - 1];
      p3 = input_[i];

      int cross = CrossProduct(p1, p2, p3);

      if (cross > 0) {
        break;
      }
      output_.pop_back();
    }
    output_.emplace_back(input_[i]);
  }
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::PreProcessingImpl() {
  auto *in_ptr = reinterpret_cast<Point *>(task_data->inputs[0]);
  input_ = std::vector<Point>(in_ptr, in_ptr + task_data->inputs_count[0]);
  output_ = std::vector<Point>();
  return true;
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::ValidationImpl() {
  return task_data->inputs_count[0] >= kMinInputPoints && task_data->inputs_count[0] <= task_data->outputs_count[0];
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::RunImpl() {
  if (!CheckGrahamNecessaryConditions()) {
    return false;
  }

  auto base_it = std::ranges::min_element(input_);
  std::iter_swap(input_.begin(), base_it);

  ParallelSort(input_.begin() + 1, input_.end(), [&](const Point &a, const Point &b) {
    auto squared_dist = [](const Point &p1, const Point &p2) -> int {
      int dx = p1.x - p2.x;
      int dy = p1.y - p2.y;
      return ((dx * dx) + (dy * dy));
    };

    int cross = CrossProduct(input_[0], a, b);
    if (cross == 0) {
      return squared_dist(a, input_[0]) < squared_dist(b, input_[0]);
    }

    return cross > 0;
  });

  GrahamScan();

  return true;
}

bool ermolaev_v_graham_scan_stl::TestTaskSTL::PostProcessingImpl() {
  task_data->outputs_count.clear();
  task_data->outputs_count.push_back(output_.size());
  std::ranges::copy(output_, reinterpret_cast<Point *>(task_data->outputs[0]));
  return true;
}