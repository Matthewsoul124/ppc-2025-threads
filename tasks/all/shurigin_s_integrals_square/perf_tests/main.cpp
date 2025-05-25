#include <gtest/gtest.h>

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wcast-align"
#endif
#include <mpi.h>
#ifdef __clang__
#pragma clang diagnostic pop
#endif

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "all/shurigin_s_integrals_square/include/ops_mpi.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {
auto test_function_2d = [](const std::vector<double>& point) {
  if (point.size() < 2) {
    return 0.0;
  }
  double x = point[0];
  double y = point[1];
  return std::cos((x * x) + (y * y)) * (1 + (x * x) + (y * y));
};
}  // namespace

namespace shurigin_s_integrals_square_mpi_test {

const double kExpectedResult = 4.35751;
const double kEpsilon = 1e-3;

TEST(shurigin_s_integrals_square_mpi, test_pipeline_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double down_limit_x = -1.0;
  double up_limit_x = 1.0;
  double down_limit_y = -1.0;
  double up_limit_y = 1.0;
  int count_x = 5000;
  int count_y = 5000;
  int dimensions = 2;

  std::vector<double> inputs_data;
  inputs_data.push_back(down_limit_x);
  inputs_data.push_back(down_limit_y);
  inputs_data.push_back(up_limit_x);
  inputs_data.push_back(up_limit_y);
  inputs_data.push_back(static_cast<double>(count_x));
  inputs_data.push_back(static_cast<double>(count_y));

  if (inputs_data.size() != static_cast<size_t>(3) * dimensions) {
    if (rank == 0) {
      std::cerr << "Error in test setup: Prepared inputs_data size mismatch!\n";
      std::cerr << "Expected size: " << (static_cast<size_t>(3) * dimensions) << ", Actual size: " << inputs_data.size()
                << "\n";
    }
    FAIL() << "Test setup failed: Prepared inputs_data size mismatch.";
  }

  std::vector<double> result_vec(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data = nullptr;

  task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs_data.data()));
    task_data->inputs_count.emplace_back(inputs_data.size() * sizeof(double));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vec.data()));
    task_data->outputs_count.emplace_back(result_vec.size() * sizeof(double));
  } else {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vec.data()));
    task_data->outputs_count.emplace_back(result_vec.size() * sizeof(double));
  }

  auto test_task_mpi = std::make_shared<shurigin_s_integrals_square_mpi::Integral>(task_data);

  test_task_mpi->SetFunction(test_function_2d, dimensions);

  if (rank == 0) {
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;

    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&, t0] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);

    perf_analyzer->PipelineRun(perf_attr, perf_results);

    ppc::core::Perf::PrintPerfStatistic(perf_results);

    ASSERT_NEAR(result_vec[0], kExpectedResult, kEpsilon);
  }
}

TEST(shurigin_s_integrals_square_mpi, test_task_run) {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  double down_limit_x = -1.0;
  double up_limit_x = 1.0;
  double down_limit_y = -1.0;
  double up_limit_y = 1.0;
  int count_x = 5000;
  int count_y = 5000;
  int dimensions = 2;

  std::vector<double> inputs_data;
  inputs_data.push_back(down_limit_x);
  inputs_data.push_back(down_limit_y);
  inputs_data.push_back(up_limit_x);
  inputs_data.push_back(up_limit_y);
  inputs_data.push_back(static_cast<double>(count_x));
  inputs_data.push_back(static_cast<double>(count_y));

  if (inputs_data.size() != static_cast<size_t>(3) * dimensions) {
    if (rank == 0) {
      std::cerr << "Error in test setup: Prepared inputs_data size mismatch!\n";
      std::cerr << "Expected size: " << (static_cast<size_t>(3) * dimensions) << ", Actual size: " << inputs_data.size()
                << "\n";
    }
    FAIL() << "Test setup failed: Prepared inputs_data size mismatch.";
  }

  std::vector<double> result_vec(1, 0.0);

  std::shared_ptr<ppc::core::TaskData> task_data = nullptr;

  task_data = std::make_shared<ppc::core::TaskData>();
  if (rank == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(inputs_data.data()));
    task_data->inputs_count.emplace_back(inputs_data.size() * sizeof(double));
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vec.data()));
    task_data->outputs_count.emplace_back(result_vec.size() * sizeof(double));
  } else {
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(result_vec.data()));
    task_data->outputs_count.emplace_back(result_vec.size() * sizeof(double));
  }

  auto test_task_mpi = std::make_shared<shurigin_s_integrals_square_mpi::Integral>(task_data);

  test_task_mpi->SetFunction(test_function_2d, dimensions);

  if (rank == 0) {
    auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
    perf_attr->num_running = 10;

    const auto t0 = std::chrono::high_resolution_clock::now();
    perf_attr->current_timer = [&, t0] {
      auto current_time_point = std::chrono::high_resolution_clock::now();
      auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(current_time_point - t0).count();
      return static_cast<double>(duration) * 1e-9;
    };

    auto perf_results = std::make_shared<ppc::core::PerfResults>();
    auto perf_analyzer = std::make_shared<ppc::core::Perf>(test_task_mpi);

    perf_analyzer->TaskRun(perf_attr, perf_results);

    ppc::core::Perf::PrintPerfStatistic(perf_results);

    ASSERT_NEAR(result_vec[0], kExpectedResult, kEpsilon);
  }
}

}  // namespace shurigin_s_integrals_square_mpi_test