#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/shulpin_i_jarvis_passage/include/ops_all.hpp"
#include "all/shulpin_i_jarvis_passage/include/test_modules.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"
#include "mpi.h"

constexpr int k_ = 100;
constexpr int kBound = 1000;

TEST(shulpin_i_jarvis_all, test_pipeline_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t num_points = 5000000;

  std::vector<shulpin_i_jarvis_all::Point> hull = {};
  std::vector<shulpin_i_jarvis_all::Point> input = {};

  if (rank == 0) {
    hull = {{-kBound, -kBound}, {-kBound, kBound}, {kBound, kBound}, {kBound, -kBound}};
    input = shulpin_all_test_module::PerfRandomGenerator(num_points, -kBound + k_, kBound - k_);
    input.insert(input.end(), hull.begin(), hull.end());
  }

  std::vector<shulpin_i_jarvis_all::Point> out(input.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }
  auto test_task_all = std::make_shared<shulpin_i_jarvis_all::JarvisALLParallel>(task_data_par);

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

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (size_t i = 0; i < hull.size(); ++i) {
      EXPECT_EQ(hull[i].x, out[i].x);
      EXPECT_EQ(hull[i].y, out[i].y);
    }
  }
}

TEST(shulpin_i_jarvis_all, test_task_run) {
  int rank{};
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  size_t num_points = 5000000;

  std::vector<shulpin_i_jarvis_all::Point> hull = {};
  std::vector<shulpin_i_jarvis_all::Point> input = {};

  if (rank == 0) {
    hull = {{-kBound, -kBound}, {-kBound, kBound}, {kBound, kBound}, {kBound, -kBound}};
    input = shulpin_all_test_module::PerfRandomGenerator(num_points, -kBound + k_, kBound - k_);
    input.insert(input.end(), hull.begin(), hull.end());
  }

  std::vector<shulpin_i_jarvis_all::Point> out(input.size());

  auto task_data_par = std::make_shared<ppc::core::TaskData>();

  if (rank == 0) {
    task_data_par->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
    task_data_par->inputs_count.emplace_back(input.size());
    task_data_par->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_par->outputs_count.emplace_back(out.size());
  }
  auto test_task_all = std::make_shared<shulpin_i_jarvis_all::JarvisALLParallel>(task_data_par);

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
  perf_analyzer->TaskRun(perf_attr, perf_results);

  if (rank == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (size_t i = 0; i < hull.size(); ++i) {
      EXPECT_EQ(hull[i].x, out[i].x);
      EXPECT_EQ(hull[i].y, out[i].y);
    }
  }
}