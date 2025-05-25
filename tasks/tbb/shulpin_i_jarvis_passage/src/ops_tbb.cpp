#include "tbb/shulpin_i_jarvis_passage/include/ops_tbb.hpp"

#include <oneapi/tbb/concurrent_unordered_set.h>
#include <oneapi/tbb/concurrent_vector.h>
#include <oneapi/tbb/parallel_reduce.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace {
int Orientation(const shulpin_i_jarvis_tbb::Point& p, const shulpin_i_jarvis_tbb::Point& q,
                const shulpin_i_jarvis_tbb::Point& r) {
  double val = ((q.y - p.y) * (r.x - q.x)) - ((q.x - p.x) * (r.y - q.y));
  if (std::fabs(val) < 1e-9) {
    return 0;
  }
  return (val > 0) ? 1 : 2;
}
}  // namespace

void shulpin_i_jarvis_tbb::JarvisSequential::MakeJarvisPassage(std::vector<shulpin_i_jarvis_tbb::Point>& input_jar,
                                                               std::vector<shulpin_i_jarvis_tbb::Point>& output_jar) {
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

bool shulpin_i_jarvis_tbb::JarvisSequential::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_tbb::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_tbb::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_seq_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_seq_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_tbb::JarvisSequential::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_tbb::JarvisSequential::RunImpl() {
  MakeJarvisPassage(input_seq_, output_seq_);
  return true;
}

bool shulpin_i_jarvis_tbb::JarvisSequential::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_seq_.begin(), output_seq_.end(), result);
  return true;
}

void shulpin_i_jarvis_tbb::JarvisTBBParallel::MakeJarvisPassageTBB(
    std::vector<shulpin_i_jarvis_tbb::Point>& input_jar, std::vector<shulpin_i_jarvis_tbb::Point>& output_jar) {
  int n = static_cast<int>(input_jar.size());

  int start = 0;
  for (int i = 1; i < n; ++i) {
    if (input_jar[i].x < input_jar[start].x ||
        (input_jar[i].x == input_jar[start].x && input_jar[i].y < input_jar[start].y)) {
      start = i;
    }
  }

  std::vector<Point> hull;
  hull.reserve(n);
  tbb::concurrent_unordered_set<Point, PointHash, PointEqual> visited;

  int current = start;
  do {
    const Point& cur_pt = input_jar[current];
    if (visited.insert(cur_pt).second) {
      hull.push_back(cur_pt);
    }

    struct BlockCandidate {
      int index;
      Point pt;
    };
    tbb::concurrent_vector<BlockCandidate> local_cands;
    int initial = (current + 1) % n;

    tbb::parallel_for(tbb::blocked_range<int>(0, n), [&](const tbb::blocked_range<int>& r) {
      int best = initial;
      for (int i = r.begin(); i < r.end(); ++i) {
        if (i == current) {
          continue;
        }
        int orient = Orientation(cur_pt, input_jar[i], input_jar[best]);
        if (orient == 2) {
          best = i;
        }
      }
      local_cands.push_back({best, input_jar[best]});
    });

    int candidate = local_cands[0].index;
    for (size_t i = 1; i < local_cands.size(); ++i) {
      int idx = local_cands[i].index;
      int orient = Orientation(cur_pt, input_jar[idx], input_jar[candidate]);
      if (orient == 2) {
        candidate = idx;
      }
    }

    current = candidate;
  } while (current != start);

  output_jar = std::move(hull);
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::PreProcessingImpl() {
  std::vector<shulpin_i_jarvis_tbb::Point> tmp_input;

  auto* tmp_data = reinterpret_cast<shulpin_i_jarvis_tbb::Point*>(task_data->inputs[0]);
  size_t tmp_size = task_data->inputs_count[0];
  tmp_input.assign(tmp_data, tmp_data + tmp_size);

  input_tbb_ = tmp_input;

  size_t output_size = task_data->outputs_count[0];
  output_tbb_.resize(output_size);

  return true;
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::ValidationImpl() {
  return (task_data->inputs_count[0] >= 3) && (task_data->inputs[0] != nullptr);
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::RunImpl() {
  MakeJarvisPassageTBB(input_tbb_, output_tbb_);
  return true;
}

bool shulpin_i_jarvis_tbb::JarvisTBBParallel::PostProcessingImpl() {
  auto* result = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::ranges::copy(output_tbb_.begin(), output_tbb_.end(), result);
  return true;
}
