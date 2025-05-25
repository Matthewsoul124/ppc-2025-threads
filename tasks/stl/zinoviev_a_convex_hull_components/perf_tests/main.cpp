#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/zinoviev_a_convex_hull_components/include/ops_stl.hpp"

using namespace zinoviev_a_convex_hull_components_stl;

namespace {

void VerifyResult(const std::vector<Point>& actual, const std::vector<Point>& expected) {
  auto sorted_actual = actual;
  auto sorted_expected = expected;

  auto point_comparator = [](const Point& a, const Point& b) { return a.x < b.x || (a.x == b.x && a.y < b.y); };

  std::ranges::sort(sorted_actual, point_comparator);
  std::ranges::sort(sorted_expected, point_comparator);

  ASSERT_EQ(sorted_actual.size(), sorted_expected.size());
  for (size_t i = 0; i < sorted_actual.size(); ++i) {
    ASSERT_EQ(sorted_actual[i].x, sorted_expected[i].x);
    ASSERT_EQ(sorted_actual[i].y, sorted_expected[i].y);
  }
}

}  // namespace

TEST(zinoviev_a_convex_hull_components_stl, test_pipeline_run) {
  const int size = 5000;
  std::vector<int> input(size * size, 0);
  for (int i = 0; i < size; ++i) {
    input[i] = 1;
    input[((size - 1) * size) + i] = 1;
    input[i * size] = 1;
    input[(i * size) + (size - 1)] = 1;
  }

  const std::vector<Point> expected = {
      {.x = 0, .y = 0}, {.x = size - 1, .y = 0}, {.x = 0, .y = size - 1}, {.x = size - 2, .y = size - 1}};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(size);
  task_data->inputs_count.emplace_back(size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new Point[4]));
  task_data->outputs_count.emplace_back(4);

  auto task = std::make_shared<ConvexHullSTL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 200;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto* output = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::vector<Point> actual(output, output + 4);
  VerifyResult(actual, expected);

  delete[] reinterpret_cast<Point*>(task_data->outputs[0]);
}

TEST(zinoviev_a_convex_hull_components_stl, test_task_run) {
  const int size = 5000;
  std::vector<int> input(size * size, 0);
  for (int i = 0; i < size; ++i) {
    input[i] = 1;
    input[((size - 1) * size) + i] = 1;
    input[i * size] = 1;
    input[(i * size) + (size - 1)] = 1;
  }

  const std::vector<Point> expected = {
      {.x = 0, .y = 0}, {.x = size - 1, .y = 0}, {.x = 0, .y = size - 1}, {.x = size - 2, .y = size - 1}};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(size);
  task_data->inputs_count.emplace_back(size);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(new Point[4]));
  task_data->outputs_count.emplace_back(4);

  auto task = std::make_shared<ConvexHullSTL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 200;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  auto* output = reinterpret_cast<Point*>(task_data->outputs[0]);
  std::vector<Point> actual(output, output + 4);
  VerifyResult(actual, expected);

  delete[] reinterpret_cast<Point*>(task_data->outputs[0]);
}