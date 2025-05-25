#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/bessonov_e_radix_sort_simple_merging/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(bessonov_e_radix_sort_simple_merging_all, test_pipeline_run) {
  const int n = 5000000;
  std::vector<double> input_vector;
  std::vector<double> output_vector;
  std::vector<double> result_vector;

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    input_vector.resize(n);
    for (int i = 0; i < n; i++) {
      input_vector[i] = static_cast<double>(n - i);
    }
    output_vector.resize(n, 0.0);
    result_vector = input_vector;
    std::ranges::sort(result_vector);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  auto test_task = std::make_shared<bessonov_e_radix_sort_simple_merging_all::TestTaskALL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  if (world.rank() == 0) {
    ASSERT_EQ(output_vector, result_vector);
  }
}

TEST(bessonov_e_radix_sort_simple_merging_all, test_task_run) {
  const int n = 5000000;
  std::vector<double> input_vector;
  std::vector<double> output_vector;
  std::vector<double> result_vector;

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    input_vector.resize(n);
    for (int i = 0; i < n; i++) {
      input_vector[i] = static_cast<double>(n - i);
    }
    output_vector.resize(n, 0.0);
    result_vector = input_vector;
    std::ranges::sort(result_vector);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
    task_data->inputs_count.emplace_back(input_vector.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
    task_data->outputs_count.emplace_back(output_vector.size());
  }

  auto test_task = std::make_shared<bessonov_e_radix_sort_simple_merging_all::TestTaskALL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0]() {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
  }

  if (world.rank() == 0) {
    ASSERT_EQ(output_vector, result_vector);
  }
}