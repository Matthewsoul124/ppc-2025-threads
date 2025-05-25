#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/kolokolova_d_integral_simpson_method/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(kolokolova_d_integral_simpson_method_all, test_pipeline_run) {
  auto func = [](std::vector<double> vec) {
    return (vec[2] * vec[2] * vec[2] * vec[1] * vec[1] / 10) + (4 * vec[0] * vec[0]) - (10 * vec[2]);
  };
  std::vector<int> step = {130, 130, 130};
  std::vector<int> bord = {1, 11, 2, 10, 0, 10};
  double func_result = 0.0;

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data->inputs_count.emplace_back(step.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data->inputs_count.emplace_back(bord.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_all = std::make_shared<kolokolova_d_integral_simpson_method_all::TestTaskALL>(task_data, func);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  double ans = 927638.04;
  double error = 1.0;
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NEAR(func_result, ans, error);
  }
}

TEST(kolokolova_d_integral_simpson_method_all, test_task_run) {
  auto func = [](std::vector<double> vec) {
    return (vec[2] * vec[2] * vec[2] * vec[1] * vec[1] / 10) + (4 * vec[0] * vec[0]) - (10 * vec[2]);
  };
  std::vector<int> step = {130, 130, 130};
  std::vector<int> bord = {1, 11, 2, 10, 0, 10};
  double func_result = 0.0;

  // Create task_data
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(step.data()));
  task_data->inputs_count.emplace_back(step.size());

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(bord.data()));
  task_data->inputs_count.emplace_back(bord.size());

  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(&func_result));
  task_data->outputs_count.emplace_back(1);

  // Create Task
  auto test_task_all = std::make_shared<kolokolova_d_integral_simpson_method_all::TestTaskALL>(task_data, func);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);

  double ans = 927638.04;
  double error = 1.0;
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_NEAR(func_result, ans, error);
  }
}
