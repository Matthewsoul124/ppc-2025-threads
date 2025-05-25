#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/filatev_v_foks/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<double> GeneratMatrix(filatev_v_foks_all::MatrixSize size) {
  std::vector<double> matrix(size.n * size.m);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(-100.0, 100.0);

  for (auto &el : matrix) {
    el = dist(gen);
  }

  return matrix;
}

std::vector<double> IdentityMatrix(size_t size) {
  std::vector<double> matrix(size * size, 0);

  for (size_t i = 0; i < size; i++) {
    matrix[(i * size) + i] = 1;
  }

  return matrix;
}

}  // namespace

TEST(filatev_v_foks_all, test_pipeline_run) {
  boost::mpi::communicator world;
  constexpr int kCount = 800;

  filatev_v_foks_all::MatrixSize size_a(kCount, kCount);
  filatev_v_foks_all::MatrixSize size_b(kCount, kCount);
  filatev_v_foks_all::MatrixSize size_c(kCount, kCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  size_t size_block = 40;
  std::vector<double> matrix_a;
  std::vector<double> matrix_c;
  std::vector<double> matrix_b;

  if (world.rank() == 0) {
    matrix_a = GeneratMatrix(size_a);
    matrix_c.resize(size_c.n * size_c.m, 0.0);
    matrix_b = IdentityMatrix(size_b.n);

    task_data->inputs_count.emplace_back(size_a.n);
    task_data->inputs_count.emplace_back(size_a.m);
    task_data->inputs_count.emplace_back(size_b.n);
    task_data->inputs_count.emplace_back(size_b.m);
    task_data->inputs_count.emplace_back(size_block);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

    task_data->outputs_count.emplace_back(size_c.n);
    task_data->outputs_count.emplace_back(size_c.m);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  }

  // Create Task
  auto task = std::make_shared<filatev_v_foks_all::Focks>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    ASSERT_EQ(matrix_a, matrix_c);
  }
}

TEST(filatev_v_foks_all, test_task_run) {
  boost::mpi::communicator world;
  constexpr int kCount = 800;

  filatev_v_foks_all::MatrixSize size_a(kCount, kCount);
  filatev_v_foks_all::MatrixSize size_b(kCount, kCount);
  filatev_v_foks_all::MatrixSize size_c(kCount, kCount);

  auto task_data = std::make_shared<ppc::core::TaskData>();

  size_t size_block = 40;
  std::vector<double> matrix_a;
  std::vector<double> matrix_c;
  std::vector<double> matrix_b;

  if (world.rank() == 0) {
    matrix_a = GeneratMatrix(size_a);
    matrix_c.resize(size_c.n * size_c.m, 0.0);
    matrix_b = IdentityMatrix(size_b.n);

    task_data->inputs_count.emplace_back(size_a.n);
    task_data->inputs_count.emplace_back(size_a.m);
    task_data->inputs_count.emplace_back(size_b.n);
    task_data->inputs_count.emplace_back(size_b.m);
    task_data->inputs_count.emplace_back(size_block);

    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));

    task_data->outputs_count.emplace_back(size_c.n);
    task_data->outputs_count.emplace_back(size_c.m);

    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_c.data()));
  }

  // Create Task
  auto task = std::make_shared<filatev_v_foks_all::Focks>(task_data);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    ASSERT_EQ(matrix_a, matrix_c);
  }
}
