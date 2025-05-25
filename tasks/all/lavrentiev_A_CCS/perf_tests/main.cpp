#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include "all/lavrentiev_A_CCS/include/ops_stl.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<double> GenerateRandomMatrix(int size, int sparse_size) {
  std::vector<double> data(size);
  std::random_device device;
  std::mt19937 generator(device());
  std::uniform_int_distribution<> random_element(-500, 500);
  size = size / sparse_size;
  for (int i = 0; i < size; ++i) {
    data[i] = static_cast<double>(random_element(generator));
  }
  std::ranges::shuffle(data, generator);
  return data;
}

std::vector<double> GenerateSingleMatrix(int size) {
  std::vector<double> test_data(size, 0.0);
  int sqrt = static_cast<int>(std::sqrt(size));
  for (int i = 0; i < sqrt; ++i) {
    for (int j = 0; j < sqrt; ++j) {
      if (i == j) {
        test_data[(sqrt * i) + j] = 1.0;
      }
    }
  }
  return test_data;
}
constexpr auto kEpsilon = 0.000001;
struct TestData {
  std::vector<double> random_data;
  std::vector<double> single_matrix;
  std::vector<double> result;
  std::shared_ptr<ppc::core::TaskData> task_data_all;
  TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size, int sparse_size,
           boost::mpi::communicator &world);
};

TestData::TestData(std::pair<int, int> matrix1_size, std::pair<int, int> matrix2_size, int sparse_size,
                   boost::mpi::communicator &world) {
  random_data = GenerateRandomMatrix(matrix1_size.first * matrix1_size.second, sparse_size);
  single_matrix = GenerateSingleMatrix(matrix2_size.first * matrix2_size.second);
  result.resize(matrix1_size.first * matrix2_size.second);
  task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(random_data.data()));
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(single_matrix.data()));
    task_data_all->inputs_count.emplace_back(matrix1_size.first);
    task_data_all->inputs_count.emplace_back(matrix1_size.second);
    task_data_all->inputs_count.emplace_back(matrix2_size.first);
    task_data_all->inputs_count.emplace_back(matrix2_size.second);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(result.data()));
    task_data_all->outputs_count.emplace_back(result.size());
  }
}
}  // namespace

TEST(lavrentiev_a_ccs_all, test_pipeline_run) {
  boost::mpi::communicator world;
  constexpr double kSize = 700;
  auto task = TestData({kSize, kSize}, {kSize, kSize}, 6, world);
  auto test_task_all = std::make_shared<lavrentiev_a_ccs_all::CCSALL>(task.task_data_all);
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
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    for (size_t i = 0; i < task.result.size(); ++i) {
      EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
    }
  }
}

TEST(lavrentiev_a_ccs_all, test_task_run) {
  boost::mpi::communicator world;
  constexpr double kSize = 700;
  auto task = TestData({kSize, kSize}, {kSize, kSize}, 6, world);
  auto test_task_all = std::make_shared<lavrentiev_a_ccs_all::CCSALL>(task.task_data_all);
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
  ppc::core::Perf::PrintPerfStatistic(perf_results);
  if (world.rank() == 0) {
    for (size_t i = 0; i < task.result.size(); ++i) {
      EXPECT_NEAR(task.result[i], task.random_data[i], kEpsilon);
    }
  }
}
