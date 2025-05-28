#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/collectives/broadcast.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "all/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_mpi.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi, test_pipeline_run) {
  constexpr int kCount = 520000;

  auto get_random_vector = [](int sz, int a, int b) -> std::vector<int> {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> dis(a, b);
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; i++) {
      vec[i] = dis(gen);
    }
    return vec;
  };

  boost::mpi::communicator world;
  std::vector<int> input;
  if (world.rank() == 0) {
    input = get_random_vector(kCount, 0, 999);
  }
  boost::mpi::broadcast(world, input, 0);
  if (input.size() != kCount) input.resize(kCount);
  std::vector<int> output(kCount, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto mpi_task = std::make_shared<fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi::TestTaskMPI>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    std::copy(mpi_task->get_internal_output().begin(), mpi_task->get_internal_output().end(), output.begin());
  }

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    std::vector<int> expected = input;
    std::ranges::sort(expected);
    EXPECT_EQ(output, expected);
  }

  for (int r = 0; r < world.size(); ++r) {
    if (world.rank() == r) {
      std::cout << "rank " << r << " input ptr: " << static_cast<void*>(input.data()) << ", size: " << input.size()
                << std::endl;
      std::cout << "rank " << r << " output ptr: " << static_cast<void*>(output.data()) << ", size: " << output.size()
                << std::endl;
    }
    world.barrier();
  }
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi, test_task_run) {
  constexpr int kCount = 520000;

  auto get_random_vector = [](int sz, int a, int b) -> std::vector<int> {
    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_int_distribution<> dis(a, b);
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; i++) {
      vec[i] = dis(gen);
    }
    return vec;
  };

  boost::mpi::communicator world;
  std::vector<int> input;
  if (world.rank() == 0) {
    input = get_random_vector(kCount, 0, 999);
  }
  boost::mpi::broadcast(world, input, 0);
  if (input.size() != kCount) input.resize(kCount);
  std::vector<int> output(kCount, 0);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  auto mpi_task = std::make_shared<fyodorov_m_shell_sort_with_even_odd_batcher_merge_mpi::TestTaskMPI>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 10;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(mpi_task);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    std::copy(mpi_task->get_internal_output().begin(), mpi_task->get_internal_output().end(), output.begin());
  }

  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);

    std::vector<int> expected = input;
    std::ranges::sort(expected);
    EXPECT_EQ(output, expected);
  }

  for (int r = 0; r < world.size(); ++r) {
    if (world.rank() == r) {
      std::cout << "rank " << r << " input ptr: " << static_cast<void*>(input.data()) << ", size: " << input.size()
                << std::endl;
      std::cout << "rank " << r << " output ptr: " << static_cast<void*>(output.data()) << ", size: " << output.size()
                << std::endl;
    }
    world.barrier();
  }
}