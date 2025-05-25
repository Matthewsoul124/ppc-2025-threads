#include "stl/zinoviev_a_convex_hull_components/include/ops_stl.hpp"

#include <algorithm>
#include <cstddef>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"

using namespace zinoviev_a_convex_hull_components_stl;

ConvexHullSTL::ConvexHullSTL(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}

bool ConvexHullSTL::PreProcessingImpl() noexcept {
  if (!task_data || task_data->inputs.empty() || task_data->inputs_count.size() < 2) {
    return false;
  }

  const auto* input_data = reinterpret_cast<int*>(task_data->inputs[0]);
  const int width = static_cast<int>(task_data->inputs_count[0]);
  const int height = static_cast<int>(task_data->inputs_count[1]);

  components_.clear();

  std::vector<bool> visited(static_cast<size_t>(width) * height, false);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const size_t idx = (static_cast<size_t>(y) * width) + x;
      if (!visited[idx] && input_data[idx] != 0) {
        std::vector<Point> component;
        BFS(input_data, width, height, x, y, visited, component);
        components_.push_back(component);
      }
    }
  }

  return true;
}

void ConvexHullSTL::BFS(const int* input_data, int width, int height, int start_x, int start_y,
                        std::vector<bool>& visited, std::vector<Point>& component) noexcept {
  std::queue<Point> queue;
  queue.push({start_x, start_y});
  const size_t start_idx = (static_cast<size_t>(start_y) * width) + start_x;
  visited[start_idx] = true;

  constexpr int kDx[] = {-1, 1, 0, 0};
  constexpr int kDy[] = {0, 0, -1, 1};

  while (!queue.empty()) {
    Point p = queue.front();
    queue.pop();
    component.push_back(p);

    for (int i = 0; i < 4; ++i) {
      int nx = p.x + kDx[i];
      int ny = p.y + kDy[i];
      if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
        const size_t nidx = (static_cast<size_t>(ny) * width) + nx;
        if (!visited[nidx] && input_data[nidx] != 0) {
          visited[nidx] = true;
          queue.push({nx, ny});
        }
      }
    }
  }
}

bool ConvexHullSTL::ValidationImpl() noexcept {
  return task_data->inputs_count.size() == 2 && task_data->outputs_count.size() == 1 &&
         task_data->inputs_count[0] > 0 && task_data->inputs_count[1] > 0;
}

int ConvexHullSTL::Cross(const Point& o, const Point& a, const Point& b) noexcept {
  return ((a.x - o.x) * (b.y - o.y)) - ((a.y - o.y) * (b.x - o.x));
}

std::vector<Point> ConvexHullSTL::FindConvexHull(const std::vector<Point>& points) noexcept {
  if (points.size() < 3) {
    return points;
  }

  std::vector<Point> sorted_points(points);
  std::ranges::sort(sorted_points,
                    [](const Point& a, const Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); });

  std::vector<Point> hull;
  hull.reserve(sorted_points.size() * 2);

  for (const auto& p : sorted_points) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), p) <= 0) {
      hull.pop_back();
    }
    hull.push_back(p);
  }

  hull.pop_back();
  for (auto it = sorted_points.rbegin(); it != sorted_points.rend(); ++it) {
    while (hull.size() >= 2 && Cross(hull[hull.size() - 2], hull.back(), *it) <= 0) {
      hull.pop_back();
    }
    hull.push_back(*it);
  }

  if (!hull.empty()) {
    hull.pop_back();
  }

  return hull;
}

bool ConvexHullSTL::RunImpl() noexcept {
  const size_t num_components = components_.size();
  hulls_.resize(num_components);

  unsigned int num_threads = ppc::util::GetPPCNumThreads();

  if (num_threads == 0) {
    num_threads = 1;
  }

  if (num_components == 0) {
    return true;
  }

  size_t actual_num_threads = std::min(static_cast<size_t>(num_threads), num_components);
  if (actual_num_threads == 0 && num_components > 0) {
    actual_num_threads = 1;
  } else if (num_components == 0) {
    actual_num_threads = 0;
  }

  if (actual_num_threads > 0) {
    std::vector<std::thread> threads(actual_num_threads);
    const size_t components_per_thread = (num_components + actual_num_threads - 1) / actual_num_threads;

    auto process_components = [this](size_t start_idx, size_t end_idx) {
      for (size_t i = start_idx; i < end_idx; ++i) {
        if (i < components_.size()) {
          hulls_[i] = FindConvexHull(components_[i]);
        }
      }
    };

    for (size_t t = 0; t < actual_num_threads; ++t) {
      const size_t start = t * components_per_thread;
      const size_t end = std::min(start + components_per_thread, num_components);
      if (start < end) {
        threads[t] = std::thread(process_components, start, end);
      }
    }

    for (size_t t = 0; t < actual_num_threads; ++t) {
      if (threads[t].joinable()) {
        threads[t].join();
      }
    }
  }

  return true;
}

bool ConvexHullSTL::PostProcessingImpl() noexcept {
  if (task_data->outputs.empty()) {
    return false;
  }

  size_t total_points = 0;
  for (const auto& hull : hulls_) {
    total_points += hull.size();
  }

  if (task_data->outputs_count.empty() || static_cast<size_t>(task_data->outputs_count[0]) < total_points) {
    if (total_points > 0) {
      return false;
    }
  }

  auto* output = reinterpret_cast<Point*>(task_data->outputs[0]);
  size_t offset = 0;
  for (const auto& hull : hulls_) {
    if (!hull.empty()) {
      if (offset + hull.size() <= static_cast<size_t>(task_data->outputs_count[0])) {
        std::ranges::copy(hull, output + offset);
        offset += hull.size();
      } else {
        return false;
      }
    }
  }

  if (!task_data->outputs_count.empty()) {
    task_data->outputs_count[0] = static_cast<int>(total_points);
  } else if (total_points > 0) {
    return false;
  }

  return true;
}