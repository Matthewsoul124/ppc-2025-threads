#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <random>
#include <vector>

#include "all/kudryashova_i_radix_batcher/include/kudryashovaRadixBatcherALL.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

std::vector<double> kudryashova_i_radix_batcher_all::GetRandomDoubleVector(int size) {
  std::vector<double> vector(size);
  std::mt19937 generator(static_cast<unsigned>(std::time(nullptr)));
  std::uniform_real_distribution<double> distribution(-1000.0, 1000.0);
  for (int i = 0; i < size; ++i) {
    vector[i] = distribution(generator);
  }
  return vector;
}

TEST(kudryashova_i_radix_batcher_all, test_pipeline_run) {
  boost::mpi::communicator world;
  int global_vector_size = 3000000;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_all::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());

  auto test_task_all = std::make_shared<kudryashova_i_radix_batcher_all::TestTaskALL>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (std::vector<double>::size_type i = 1; i < result.size(); i++) {
      ASSERT_LE(result[i - 1], result[i]);
    }
  }
}

TEST(kudryashova_i_radix_batcher_all, test_task_run) {
  boost::mpi::communicator world;
  int global_vector_size = 3000000;
  std::vector<double> global_vector = kudryashova_i_radix_batcher_all::GetRandomDoubleVector(global_vector_size);
  std::vector<double> result(global_vector_size);
  std::shared_ptr<ppc::core::TaskData> task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(global_vector.data()));
  task_data->inputs_count.emplace_back(global_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
  task_data->outputs_count.emplace_back(result.size());

  auto test_task_all = std::make_shared<kudryashova_i_radix_batcher_all::TestTaskALL>(task_data);
  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 3;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto current_time_point = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
    return static_cast<double>(duration) * 1e-9;
  };
  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    for (std::vector<double>::size_type i = 1; i < result.size(); i++) {
      ASSERT_LE(result[i - 1], result[i]);
    }
  }
}
