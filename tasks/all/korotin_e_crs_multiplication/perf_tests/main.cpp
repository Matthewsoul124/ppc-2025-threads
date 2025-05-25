#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/korotin_e_crs_multiplication/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(korotin_e_crs_multiplication_all, test_pipeline_run) {
  boost::mpi::communicator world;
  const unsigned int n = 900;
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  std::vector<double> a_val(n * n, 1);
  std::vector<double> b_val(n * n, 1);
  std::vector<unsigned int> a_ri(n + 1, 0);
  std::vector<unsigned int> a_col(n * n);
  std::vector<unsigned int> b_ri(n + 1, 0);
  std::vector<unsigned int> b_col(n * n);
  std::vector<unsigned int> out_ri(n + 1, 0);
  std::vector<unsigned int> out_col(n * n);
  std::vector<double> out_val(n * n);
  std::vector<unsigned int> c_ri(n + 1, 0);
  std::vector<unsigned int> c_col(n * n);
  std::vector<double> c_val(n * n, n);

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < n; j++) {
        a_col[(i * n) + j] = j;
        b_col[(i * n) + j] = j;
        c_col[(i * n) + j] = j;
      }
      a_ri[i + 1] = n * (i + 1);
      b_ri[i + 1] = n * (i + 1);
      c_ri[i + 1] = n * (i + 1);
    }

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_ri.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
    task_data_all->inputs_count.emplace_back(a_ri.size());
    task_data_all->inputs_count.emplace_back(a_col.size());
    task_data_all->inputs_count.emplace_back(a_val.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_ri.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
    task_data_all->inputs_count.emplace_back(b_ri.size());
    task_data_all->inputs_count.emplace_back(b_col.size());
    task_data_all->inputs_count.emplace_back(b_val.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ri.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
    task_data_all->outputs_count.emplace_back(out_ri.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<korotin_e_crs_multiplication_all::CrsMultiplicationALL>(task_data_all);
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    ASSERT_EQ(c_ri, out_ri);
    ASSERT_EQ(c_col, out_col);
    ASSERT_EQ(c_val, out_val);
  }
}

TEST(korotin_e_crs_multiplication_all, test_task_run) {
  boost::mpi::communicator world;
  const unsigned int n = 900;
  auto task_data_all = std::make_shared<ppc::core::TaskData>();

  std::vector<double> a_val(n * n, 1);
  std::vector<double> b_val(n * n, 1);
  std::vector<unsigned int> a_ri(n + 1, 0);
  std::vector<unsigned int> a_col(n * n);
  std::vector<unsigned int> b_ri(n + 1, 0);
  std::vector<unsigned int> b_col(n * n);
  std::vector<unsigned int> out_ri(n + 1, 0);
  std::vector<unsigned int> out_col(n * n);
  std::vector<double> out_val(n * n);
  std::vector<unsigned int> c_ri(n + 1, 0);
  std::vector<unsigned int> c_col(n * n);
  std::vector<double> c_val(n * n, n);

  if (world.rank() == 0) {
    for (unsigned int i = 0; i < n; i++) {
      for (unsigned int j = 0; j < n; j++) {
        a_col[(i * n) + j] = j;
        b_col[(i * n) + j] = j;
        c_col[(i * n) + j] = j;
      }
      a_ri[i + 1] = n * (i + 1);
      b_ri[i + 1] = n * (i + 1);
      c_ri[i + 1] = n * (i + 1);
    }

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_ri.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_col.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(a_val.data()));
    task_data_all->inputs_count.emplace_back(a_ri.size());
    task_data_all->inputs_count.emplace_back(a_col.size());
    task_data_all->inputs_count.emplace_back(a_val.size());

    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_ri.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_col.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(b_val.data()));
    task_data_all->inputs_count.emplace_back(b_ri.size());
    task_data_all->inputs_count.emplace_back(b_col.size());
    task_data_all->inputs_count.emplace_back(b_val.size());

    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_ri.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_col.data()));
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out_val.data()));
    task_data_all->outputs_count.emplace_back(out_ri.size());
  }

  // Create Task
  auto test_task_all = std::make_shared<korotin_e_crs_multiplication_all::CrsMultiplicationALL>(task_data_all);
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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    ASSERT_EQ(c_ri, out_ri);
    ASSERT_EQ(c_col, out_col);
    ASSERT_EQ(c_val, out_val);
  }
}