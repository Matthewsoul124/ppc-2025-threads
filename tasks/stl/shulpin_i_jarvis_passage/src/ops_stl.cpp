#include "stl/shulpin_i_jarvis_passage/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_set>
#include <vector>

#include "core/util/include/util.hpp"

namespace {
int Orientation(const shulpin_i_jarvis_stl::Point& p, const shulpin_i_jarvis_stl::Point& q,
                const shulpin_i_jarvis_stl::Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}

}  // namespace

void shulpin_i_jarvis_stl::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_stl::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_stl::Point>& output_jar) {
  size_t total = input_jar.size();
  output_jar.clear();

  size_t start = 0;
  for (size_t i = 1; i < total; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  size_t active = start;
  do {
    output_jar.emplace_back(input_jar[active]);
    size_t candidate = (active + 1) % total;

    for (size_t index = 0; index < total; ++index) {
      if (Orientation(input_jar[active], input_jar[index], input_jar[candidate]) == 2) {
        candidate = index;
      }
    }

    active = candidate;
  } while (active != start);
}

bool shulpin_i_jarvis_stl::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_stl::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_stl::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_stl::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_stl::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_stl::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

#ifdef __linux__
// this whole nolint block is for NOLINT(readability-function-cognitive-complexity). using it as end-of-line comment
// doesn't work. all other linter warnings have been resolved
// NOLINTBEGIN
void shulpin_i_jarvis_stl::JarvisSTLParallel::MakeJarvisPassageSTL(
    std::vector<shulpin_i_jarvis_stl::Point>& input_jar, std::vector<shulpin_i_jarvis_stl::Point>& output_jar) {
  output_jar.clear();

  std::unordered_set<Point, PointHash, PointEqual> unique_points;

  size_t most_left = 0;
  for (size_t i = 1; i < input_jar.size(); ++i) {
    if (input_jar[i].x < input_jar[most_left].x ||
        (input_jar[i].x == input_jar[most_left].x && input_jar[i].y < input_jar[most_left].y)) {
      most_left = i;
    }
  }

  const Point& min_point = input_jar[most_left];
  std::vector<Point> convex_hull = {min_point};
  Point prev_point = min_point;
  Point next_point;

  int num_threads = ppc::util::GetPPCNumThreads();
  int chunk_size = static_cast<int>(input_jar.size() / num_threads);

  std::vector<std::thread> threads;
  std::vector<Point> candidates(num_threads, input_jar[0]);
  std::vector<bool> thread_ready(num_threads, false);
  std::vector<bool> thread_done(num_threads, false);
  std::mutex mtx;
  std::condition_variable cv;
  bool stop = false;

  auto findNextPointThread = [&](int tid) {
    while (true) {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] { return thread_ready[tid] || stop; });

      if (stop) {
        return;
      }

      int start = tid * chunk_size;
      int end = (tid == num_threads - 1) ? static_cast<int>(input_jar.size()) : (tid + 1) * chunk_size;
      Point candidate = input_jar[start];

      for (int i = start; i < end; ++i) {
        const auto& point = input_jar[i];
        if (point == prev_point) {
          continue;
        }

        double cross_product = ((point.y - prev_point.y) * (candidate.x - prev_point.x)) -
                               ((point.x - prev_point.x) * (candidate.y - prev_point.y));
        double dist1 = std::pow(point.x - prev_point.x, 2) + std::pow(point.y - prev_point.y, 2);
        double dist2 = std::pow(candidate.x - prev_point.x, 2) + std::pow(candidate.y - prev_point.y, 2);

        if (cross_product > 0 || (cross_product == 0 && dist1 > dist2)) {
          candidate = point;
        }
      }

      candidates[tid] = candidate;
      thread_ready[tid] = false;
      thread_done[tid] = true;
      cv.notify_all();
    }
  };

  for (int i = 0; i < num_threads; ++i) {
    threads.emplace_back(findNextPointThread, i);
  }

  do {
    next_point = input_jar[0];

    {
      std::unique_lock<std::mutex> lock(mtx);
      for (int i = 0; i < num_threads; ++i) {
        thread_ready[i] = true;
        thread_done[i] = false;
      }
    }
    cv.notify_all();

    {
      std::unique_lock<std::mutex> lock(mtx);
      cv.wait(lock, [&] {
        return std::ranges::all_of(thread_done.begin(), thread_done.end(), [](bool done) { return done; });
      });
    }

    for (const auto& candidate : candidates) {
      double cross_product = ((candidate.y - prev_point.y) * (next_point.x - prev_point.x)) -
                             ((candidate.x - prev_point.x) * (next_point.y - prev_point.y));
      double dist1 = std::pow(candidate.x - prev_point.x, 2) + std::pow(candidate.y - prev_point.y, 2);
      double dist2 = std::pow(next_point.x - prev_point.x, 2) + std::pow(next_point.y - prev_point.y, 2);
      if (cross_product > 0 || (cross_product == 0 && dist1 > dist2)) {
        next_point = candidate;
      }
    }

    if (unique_points.find(next_point) == unique_points.end()) {
      output_jar.push_back(next_point);
      unique_points.insert(next_point);
    }

    prev_point = next_point;

  } while (next_point != min_point);

  {
    std::unique_lock<std::mutex> lock(mtx);
    stop = true;
    cv.notify_all();
  }

  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
}
// NOLINTEND
#else
void shulpin_i_jarvis_stl::JarvisSTLParallel::MakeJarvisPassageSTL(
    std::vector<shulpin_i_jarvis_stl::Point>& input_jar, std::vector<shulpin_i_jarvis_stl::Point>& output_jar) {
  output_jar.clear();

  std::unordered_set<shulpin_i_jarvis_stl::Point, shulpin_i_jarvis_stl::PointHash, shulpin_i_jarvis_stl::PointEqual>
      unique_points;

  size_t most_left = 0;
  for (size_t i = 1; i < input_jar.size(); ++i) {
    if (input_jar[i].x < input_jar[most_left].x ||
        (input_jar[i].x == input_jar[most_left].x && input_jar[i].y < input_jar[most_left].y)) {
      most_left = i;
    }
  }

  const Point& min_point = input_jar[most_left];
  std::vector<Point> convex_hull = {min_point};
  Point prev_point = min_point;
  Point next_point;

  auto findNextPoint = [](const Point& current_point, const std::vector<Point>& points, int start, int end,
                          Point& candidate) {
    for (int i = start; i < end; ++i) {
      const auto& point = points[i];
      if (point == current_point) {
        continue;
      }
      double cross_product = ((point.y - current_point.y) * (candidate.x - current_point.x)) -
                             ((point.x - current_point.x) * (candidate.y - current_point.y));
      double dist_current_point = std::pow(point.x - current_point.x, 2) + std::pow(point.y - current_point.y, 2);
      double dist_candidate = std::pow(candidate.x - current_point.x, 2) + std::pow(candidate.y - current_point.y, 2);
      if (cross_product > 0 || (cross_product == 0 && dist_current_point > dist_candidate)) {
        candidate = point;
      }
    }
  };

  do {
    next_point = input_jar[0];
    int num_threads = ppc::util::GetPPCNumThreads();
    int chunk_size = input_jar.size() / num_threads;
    std::vector<std::thread> threads;
    std::vector<Point> candidates(num_threads, next_point);

    for (int i = 0; i < num_threads; ++i) {
      int start = i * chunk_size;
      int end = (i == num_threads - 1) ? input_jar.size() : (i + 1) * chunk_size;
      threads.emplace_back(findNextPoint, std::ref(prev_point), std::cref(input_jar), start, end,
                           std::ref(candidates[i]));
    }

    for (auto& thread : threads) {
      if (thread.joinable()) thread.join();
    }

    for (const auto& candidate : candidates) {
      double cross_product = ((candidate.y - prev_point.y) * (next_point.x - prev_point.x)) -
                             ((candidate.x - prev_point.x) * (next_point.y - prev_point.y));
      double dist_prev_point = std::pow(candidate.x - prev_point.x, 2) + std::pow(candidate.y - prev_point.y, 2);
      double dist_next_point = std::pow(next_point.x - prev_point.x, 2) + std::pow(next_point.y - prev_point.y, 2);
      if (cross_product > 0 || (cross_product == 0 && dist_prev_point > dist_next_point)) {
        next_point = candidate;
      }
    }

    if (unique_points.find(next_point) == unique_points.end()) {
      output_jar.push_back(next_point);
      unique_points.insert(next_point);
    }

    prev_point = next_point;

  } while (next_point != min_point);
}
#endif

bool shulpin_i_jarvis_stl::JarvisSTLParallel::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_stl::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_stl::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_stl_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_stl_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_stl::JarvisSTLParallel::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_stl::JarvisSTLParallel::RunImpl() {
  MakeJarvisPassageSTL(input_stl_, output_stl_);
  return true;
}

bool shulpin_i_jarvis_stl::JarvisSTLParallel::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_stl_.begin(), output_stl_.end(), result);
  return true;
}