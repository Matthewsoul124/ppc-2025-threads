#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "stl/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_stl.hpp"

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl, test_pipeline_run) {
  std::vector<double> lower = {0.1, 0.1, 0.1};
  std::vector<double> upper = {2.0, 2.0, 2.0};
  std::vector<int> steps = {200, 200, 200};
  double result = 0.0;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_stl->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_stl->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_stl->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_stl->outputs_count.emplace_back(sizeof(double));

  auto test_task_stl =
      std::make_shared<prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl::TestTaskSTL>(task_data_stl);

  test_task_stl->SetFunction([](const std::vector<double>& point) {
    return (std::exp((point[0] * point[1])) * std::sin(point[2])) + std::sqrt((point[0] + point[1] + point[2])) +
           std::log((1.0 + (point[0] * point[1] * point[2])));
  });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl, test_task_run) {
  std::vector<double> lower = {0.1, 0.1};
  std::vector<double> upper = {3.0, 3.0};
  std::vector<int> steps = {800, 800};
  double result = 0.0;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_stl->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_stl->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_stl->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_stl->outputs_count.emplace_back(sizeof(double));

  auto test_task_stl =
      std::make_shared<prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl::TestTaskSTL>(task_data_stl);

  test_task_stl->SetFunction([](const std::vector<double>& point) {
    return (std::pow(point[0], 2.0) * std::cos(point[1])) +
           (std::exp((point[0] + point[1])) * std::log((1.0 + (point[0] * point[1]))));
  });

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
}