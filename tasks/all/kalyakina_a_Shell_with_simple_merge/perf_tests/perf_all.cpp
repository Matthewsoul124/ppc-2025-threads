#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "all/kalyakina_a_Shell_with_simple_merge/include/ops_all.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<int> CreateReverseSortedVector(unsigned int size, const int left) {
  std::vector<int> result;
  while (size-- != 0) {
    result.push_back(left + static_cast<int>(size));
  }
  return result;
}
}  // namespace

TEST(kalyakina_a_shell_with_simple_merge_all, test_pipeline_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> in;
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = CreateReverseSortedVector(3000000, -1500000);
    out = std::vector<int>(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<kalyakina_a_shell_with_simple_merge_all::ShellSortALL>(task_data_all);

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

  auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_all);
  perf_analyzer->PipelineRun(perf_attr, perf_results);
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(std::ranges::is_sorted(out.begin(), out.end()));
  }
}

TEST(kalyakina_a_shell_with_simple_merge_all, test_task_run) {
  boost::mpi::communicator world;

  // Create data
  std::vector<int> in;
  std::vector<int> out;

  // Create TaskData
  std::shared_ptr<ppc::core::TaskData> task_data_all = std::make_shared<ppc::core::TaskData>();

  if (world.rank() == 0) {
    in = CreateReverseSortedVector(3000000, -1500000);
    out = std::vector<int>(in.size());
    task_data_all->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
    task_data_all->inputs_count.emplace_back(in.size());
    task_data_all->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data_all->outputs_count.emplace_back(out.size());
  }

  auto test_task_all = std::make_shared<kalyakina_a_shell_with_simple_merge_all::ShellSortALL>(task_data_all);

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
  // Create Perf analyzer
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    ASSERT_TRUE(std::ranges::is_sorted(out.begin(), out.end()));
  }
}
