#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <unordered_set>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/trubin_a_algorithm_dijkstra/include/ops_stl.hpp"

namespace {
std::vector<int> GenerateRandomDenseGraph(size_t num_vertices, size_t max_edges_per_vertex) {
  std::vector<int> graph_data;
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> edge_count_dist(1, max_edges_per_vertex);
  std::uniform_int_distribution<int> weight_dist(100, 1000);

  for (size_t i = 0; i < num_vertices; ++i) {
    size_t num_edges = edge_count_dist(rng);
    std::unordered_set<size_t> used;
    while (used.size() < num_edges) {
      size_t to = rng() % num_vertices;
      if (to == i || used.contains(to)) {
        continue;
      }
      used.insert(to);
      graph_data.push_back(static_cast<int>(to));
      graph_data.push_back(weight_dist(rng));
    }
    graph_data.push_back(-1);
  }
  return graph_data;
}

void RunDijkstraPerfTest(size_t num_vertices, size_t max_edges_per_vertex, bool is_pipeline) {
  auto graph_data = GenerateRandomDenseGraph(num_vertices, max_edges_per_vertex);
  int start_vertex = 0;
  std::vector<int> distances(num_vertices, -42);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.push_back(reinterpret_cast<uint8_t*>(graph_data.data()));
  task_data_stl->inputs_count.push_back(static_cast<uint32_t>(graph_data.size()));
  task_data_stl->inputs.push_back(reinterpret_cast<uint8_t*>(&start_vertex));
  task_data_stl->outputs.push_back(reinterpret_cast<uint8_t*>(distances.data()));
  task_data_stl->outputs_count.push_back(static_cast<uint32_t>(num_vertices));

  auto task = std::make_shared<trubin_a_algorithm_dijkstra_stl::TestTaskSTL>(task_data_stl);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);

  if (is_pipeline) {
    perf_analyzer->PipelineRun(perf_attr, perf_results);
  } else {
    perf_analyzer->TaskRun(perf_attr, perf_results);
  }
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  ASSERT_EQ(distances[start_vertex], 0);
  for (size_t i = 0; i < num_vertices; ++i) {
    EXPECT_TRUE(distances[i] >= -1);
  }
}
}  // namespace

TEST(trubin_a_algorithm_dijkstra_stl, test_pipeline_run) {
  constexpr size_t kNumVertices = 200000;
  constexpr size_t kMaxEdgesPerVertex = 100;
  RunDijkstraPerfTest(kNumVertices, kMaxEdgesPerVertex, true);
}

TEST(trubin_a_algorithm_dijkstra_stl, test_task_run) {
  constexpr size_t kNumVertices = 200000;
  constexpr size_t kMaxEdgesPerVertex = 100;
  RunDijkstraPerfTest(kNumVertices, kMaxEdgesPerVertex, false);
}
