#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "all/kozlova_e_contrast_enhancement/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
std::vector<uint8_t> GenerateVector(size_t length);

std::vector<uint8_t> GenerateVector(size_t length) {
  std::vector<uint8_t> vec;
  vec.reserve(length);
  for (size_t i = 0; i < length; ++i) {
    vec.push_back(rand() % 256);
  }
  return vec;
}
}  // namespace

TEST(kozlova_e_contrast_enhancement_all, test_pipeline_run) {
  constexpr size_t kSize = 19875000;
  size_t width = 7500;
  size_t height = 2650;
  boost::mpi::communicator world;
  // Create data
  std::vector<uint8_t> in;
  std::vector<uint8_t> out(kSize, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateVector(kSize);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_alluential = std::make_shared<kozlova_e_contrast_enhancement_all::TestTaskAll>(task_data_all);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    uint8_t min_value = *std::ranges::min_element(in);
    uint8_t max_value = *std::ranges::max_element(in);

    for (size_t i = 0; i < in.size(); ++i) {
      auto expected = static_cast<uint8_t>(((in[i] - min_value) / (double)(max_value - min_value)) * 255);
      expected = std::clamp((int)expected, 0, 255);
      EXPECT_EQ(out[i], expected);
    }
  }
}

TEST(kozlova_e_contrast_enhancement_all, test_task_run) {
  constexpr size_t kSize = 19875000;
  size_t width = 7500;
  size_t height = 2650;
  boost::mpi::communicator world;
  // Create data
  std::vector<uint8_t> in;
  std::vector<uint8_t> out(kSize, 0);

  // Create task_data
  auto task_data_all = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    in = GenerateVector(kSize);
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->inputs_count.emplace_back(width);
    task_data_all->inputs_count.emplace_back(height);
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  // Create Task
  auto test_task_alluential = std::make_shared<kozlova_e_contrast_enhancement_all::TestTaskAll>(task_data_all);

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
  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_alluential);
  perf_analyzer->TaskRun(perf_attr, perf_results);
  ppc::core::Perf::PrintPerfStatistic(perf_results);

  if (world.rank() == 0) {
    uint8_t min_value = *std::ranges::min_element(in);
    uint8_t max_value = *std::ranges::max_element(in);

    for (size_t i = 0; i < in.size(); ++i) {
      auto expected = static_cast<uint8_t>(((in[i] - min_value) / (double)(max_value - min_value)) * 255);
      expected = std::clamp((int)expected, 0, 255);
      EXPECT_EQ(out[i], expected);
    }
  }
}