#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/odintsov_m_multmatrix_cannon/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(odintsov_m_multmatrix_cannon_all, test_pipeline_run) {
  boost::mpi::communicator com;
  // Create data
  std::vector<double> matrix_a(90000, 1);
  std::vector<double> matrix_b(90000, 1);
  std::vector<double> out(90000, 0);
  std::vector<double> matrix_c(90000, 300);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    // Create task_data

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  }

  // Create Task
  auto test_task_stl = std::make_shared<odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL>(task_data_all);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (com.rank() == 0) {
    EXPECT_EQ(out, matrix_c);
  }
}

TEST(odintsov_m_multmatrix_cannon_all, test_task_run) {
  boost::mpi::communicator com;
  // Create data
  std::vector<double> matrix_a(90000, 1);
  std::vector<double> matrix_b(90000, 1);
  std::vector<double> out(90000, 0);
  std::vector<double> matrix_c(90000, 300);
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (com.rank() == 0) {
    // Create task_data

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_a.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(matrix_b.data()));
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->inputs_count.emplace_back(matrix_a.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  }

  // Create Task
  auto test_task_stl = std::make_shared<odintsov_m_mulmatrix_cannon_all::MulMatrixCannonALL>(task_data_all);

  // Create Perf attributes
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 50;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [t0] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  // Create and init perf results
  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  // Create Perf analyzer
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_stl);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (com.rank() == 0) {
    EXPECT_EQ(out, matrix_c);
  }
}
