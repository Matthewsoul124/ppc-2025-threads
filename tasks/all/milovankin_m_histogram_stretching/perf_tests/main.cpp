#include <gtest/gtest.h>

#include <boost/mpi/collectives.hpp>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "../include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {

milovankin_m_histogram_stretching_all::TestTaskAll CreateParallelTask(std::vector<uint8_t>& data_in,
                                                                      std::vector<uint8_t>& data_out) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(data_in.data());
  task_data->inputs_count.emplace_back(static_cast<uint32_t>(data_in.size()));

  task_data->outputs.emplace_back(data_out.data());
  task_data->outputs_count.emplace_back(static_cast<uint32_t>(data_out.size()));

  return milovankin_m_histogram_stretching_all::TestTaskAll(task_data);
}

}  // namespace

TEST(milovankin_m_histogram_stretching_all, test_pipeline_run) {
  boost::mpi::communicator world;

  std::vector<uint8_t> data_in;
  std::vector<uint8_t> data_out;

  // Only rank 0 initializes the input data
  if (world.rank() == 0) {
    data_in.resize(123456789, 123);
    data_out.resize(data_in.size());
    data_in.front() = 5;
    data_in.back() = 155;
  } else {
    data_in.resize(123456789);
    data_out.resize(123456789);
  }

  boost::mpi::broadcast(world, data_in.data(), static_cast<int>(data_in.size()), 0);

  auto task =
      std::make_shared<milovankin_m_histogram_stretching_all::TestTaskAll>(CreateParallelTask(data_in, data_out));

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

  // Only rank 0 prints results and verifies output
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    // Verify output correctness
    std::vector<uint8_t> data_expected(data_in.size(), 201);  // ((123 - 5) * 255 + 150 / 2) / 150 = 201
    data_expected.front() = 0;
    data_expected.back() = 255;

    EXPECT_EQ(data_out, data_expected);
  }
}

TEST(milovankin_m_histogram_stretching_all, test_task_run) {
  boost::mpi::communicator world;

  std::vector<uint8_t> data_in;
  std::vector<uint8_t> data_out;

  // Only rank 0 initializes the input data
  if (world.rank() == 0) {
    data_in.resize(123456789, 123);
    data_out.resize(data_in.size());
    data_in.front() = 5;
    data_in.back() = 155;
  } else {
    data_in.resize(123456789);
    data_out.resize(123456789);
  }

  // Broadcast data to all processes
  boost::mpi::broadcast(world, data_in.data(), static_cast<int>(data_in.size()), 0);

  auto task =
      std::make_shared<milovankin_m_histogram_stretching_all::TestTaskAll>(CreateParallelTask(data_in, data_out));

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

  // Only rank 0 prints results and verifies output
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    // Verify output correctness
    std::vector<uint8_t> data_expected(data_in.size(), 201);  // ((123 - 5) * 255 + 150 / 2) / 150 = 201
    data_expected.front() = 0;
    data_expected.back() = 255;

    EXPECT_EQ(data_out, data_expected);
  }
}