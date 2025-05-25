#include "stl/kapustin_i_jarv_alg/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/util/include/util.hpp"

namespace kapustin_i_jarv_alg_stl {
struct PairHash {
  size_t operator()(const std::pair<int, int>& p) const {
    return std::hash<int>()(p.first) ^ std::hash<int>()(p.second);
  }
};

struct PairEqual {
  bool operator()(const std::pair<int, int>& lhs, const std::pair<int, int>& rhs) const {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }
};
}  // namespace kapustin_i_jarv_alg_stl

bool kapustin_i_jarv_alg_stl::TestTaskSTL::ValidationImpl() {
  return !task_data->inputs.empty() && task_data->inputs[0] != nullptr;
}

void kapustin_i_jarv_alg_stl::TestTaskSTL::FindBestPointMultithreaded(size_t current_index,
                                                                      std::vector<size_t>& local_best) {
  const size_t total_points = input_.size();
  const size_t num_threads = local_best.size();
  const size_t chunk_size = (total_points + num_threads - 1) / num_threads;

  std::vector<std::thread> threads;
  threads.reserve(num_threads);

  for (size_t thread_id = 0; thread_id < num_threads; ++thread_id) {
    const size_t start = thread_id * chunk_size;
    const size_t end = std::min(start + chunk_size, total_points);

    threads.emplace_back([this, current_index, &local_best, thread_id, start, end]() {
      const auto& current_point = input_[current_index];
      size_t best = (current_index + 1) % input_.size();
      for (size_t j = start; j < end; ++j) {
        if (j == current_index) {
          continue;
        }
        const auto& candidate = input_[j];
        const auto& best_point = input_[best];

        const int orient = ((best_point.second - current_point.second) * (candidate.first - best_point.first)) -
                           ((best_point.first - current_point.first) * (candidate.second - best_point.second));

        if (orient > 0) {
          best = j;
        } else if (orient == 0) {
          const int dx1 = candidate.first - current_point.first;
          const int dy1 = candidate.second - current_point.second;
          const int dx2 = best_point.first - current_point.first;
          const int dy2 = best_point.second - current_point.second;
          const int dist1 = (dx1 * dx1) + (dy1 * dy1);
          const int dist2 = (dx2 * dx2) + (dy2 * dy2);
          if (dist1 > dist2) {
            best = j;
          }
        }
      }
      local_best[thread_id] = best;
    });
  }

  for (auto& t : threads) {
    t.join();
  }
}

bool kapustin_i_jarv_alg_stl::TestTaskSTL::PreProcessingImpl() {
  std::vector<std::pair<int, int>> points;
  for (size_t i = 0; i < task_data->inputs.size(); ++i) {
    auto* data = reinterpret_cast<std::pair<int, int>*>(task_data->inputs[i]);
    points.insert(points.end(), data, data + task_data->inputs_count[i]);
  }
  input_ = std::move(points);

  leftmost_index_ = 0;
  for (size_t i = 1; i < input_.size(); ++i) {
    if (input_[i].first < input_[leftmost_index_].first) {
      leftmost_index_ = i;
    }
  }
  current_point_ = input_[leftmost_index_];
  return true;
}

bool kapustin_i_jarv_alg_stl::TestTaskSTL::RunImpl() {
  std::unordered_set<std::pair<int, int>, PairHash, PairEqual> unique_points;
  const auto start_point = current_point_;
  size_t current_index = leftmost_index_;
  output_.clear();
  output_.reserve(input_.size());

  output_.push_back(start_point);
  unique_points.insert(start_point);
  const auto num_threads = std::min(static_cast<size_t>(ppc::util::GetPPCNumThreads()), input_.size());

  do {
    std::vector<size_t> local_best(num_threads, (current_index + 1) % input_.size());
    FindBestPointMultithreaded(current_index, local_best);

    size_t best_index = local_best[0];
    for (size_t i = 1; i < local_best.size(); ++i) {
      const int orient = Orientation(input_[current_index], input_[best_index], input_[local_best[i]]);
      if (orient > 0 || (orient == 0 && CalculateDistance(input_[local_best[i]], input_[current_index]) >
                                            CalculateDistance(input_[best_index], input_[current_index]))) {
        best_index = local_best[i];
      }
    }

    if (input_[best_index] == start_point) {
      break;
    }
    current_point_ = input_[best_index];
    output_.push_back(current_point_);
    unique_points.insert(current_point_);
    current_index = best_index;
  } while (current_point_ != start_point);

  return true;
}

bool kapustin_i_jarv_alg_stl::TestTaskSTL::PostProcessingImpl() {
  auto* result_ptr = reinterpret_cast<std::pair<int, int>*>(task_data->outputs[0]);
  std::ranges::copy(output_, result_ptr);
  return true;
}

int kapustin_i_jarv_alg_stl::TestTaskSTL::CalculateDistance(const std::pair<int, int>& p1,
                                                            const std::pair<int, int>& p2) {
  const int dx = p1.first - p2.first;
  const int dy = p1.second - p2.second;
  return (dx * dx) + (dy * dy);
}

int kapustin_i_jarv_alg_stl::TestTaskSTL::Orientation(const std::pair<int, int>& p, const std::pair<int, int>& q,
                                                      const std::pair<int, int>& r) {
  const int val = ((q.second - p.second) * (r.first - q.first)) - ((q.first - p.first) * (r.second - q.second));
  if (val == 0) {
    return 0;
  }
  return val > 0 ? 1 : -1;
}
