#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "all/borisov_s_strassen/include/ops_all.hpp"
#include "core/perf/include/perf.hpp"
#include "core/task/include/task.hpp"

namespace {

std::vector<double> MultiplyNaive(const std::vector<double>& a, const std::vector<double>& b, int rows_a, int cols_a,
                                  int cols_b) {
  std::vector<double> c(rows_a * cols_b, 0.0);
  for (int i = 0; i < rows_a; ++i) {
    for (int j = 0; j < cols_b; ++j) {
      double sum = 0.0;
      for (int k = 0; k < cols_a; ++k) {
        sum += a[(i * cols_a) + k] * b[(k * cols_b) + j];
      }
      c[(i * cols_b) + j] = sum;
    }
  }
  return c;
}

void GenerateRandomMatrix(int rows, int cols, std::vector<double>& matrix) {
  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_real_distribution<double> dist(-50.0, 50.0);
  matrix.resize(rows * cols);
  for (auto& value : matrix) {
    value = dist(rng);
  }
}

}  // namespace

TEST(borisov_s_strassen_all, test_pipeline_run) {
  boost::mpi::communicator world;

  constexpr int kRowsA = 512;
  constexpr int kColsA = 512;
  constexpr int kRowsB = 512;
  constexpr int kColsB = 512;

  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> in_buf;
  std::vector<double> out_buf;
  if (world.rank() == 0) {
    GenerateRandomMatrix(kRowsA, kColsA, a);
    GenerateRandomMatrix(kRowsB, kColsB, b);
    in_buf = {static_cast<double>(kRowsA), static_cast<double>(kColsA), static_cast<double>(kRowsB),
              static_cast<double>(kColsB)};
    in_buf.insert(in_buf.end(), a.begin(), a.end());
    in_buf.insert(in_buf.end(), b.begin(), b.end());
    out_buf.assign(2 + (kRowsA * kColsB), 0.0);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_buf.data()));
    task_data->inputs_count.push_back(in_buf.size());
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_buf.data()));
    task_data->outputs_count.push_back(out_buf.size());
  }

  auto task = std::make_shared<borisov_s_strassen_all::ParallelStrassenMpiStl>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - t0).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->PipelineRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    auto expected = MultiplyNaive(a, b, kRowsA, kColsA, kColsB);
    std::vector<double> got(out_buf.begin() + 2, out_buf.end());
    ASSERT_EQ(expected.size(), got.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(expected[i], got[i], 1e-8);
    }
  }
}

TEST(borisov_s_strassen_all, test_task_run) {
  boost::mpi::communicator world;

  constexpr int kRowsA = 512;
  constexpr int kColsA = 512;
  constexpr int kRowsB = 512;
  constexpr int kColsB = 512;

  std::vector<double> a;
  std::vector<double> b;
  std::vector<double> in_buf;
  std::vector<double> out_buf;
  if (world.rank() == 0) {
    GenerateRandomMatrix(kRowsA, kColsA, a);
    GenerateRandomMatrix(kRowsB, kColsB, b);
    in_buf = {static_cast<double>(kRowsA), static_cast<double>(kColsA), static_cast<double>(kRowsB),
              static_cast<double>(kColsB)};
    in_buf.insert(in_buf.end(), a.begin(), a.end());
    in_buf.insert(in_buf.end(), b.begin(), b.end());
    out_buf.assign(2 + (kRowsA * kColsB), 0.0);
  }

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.push_back(reinterpret_cast<uint8_t*>(in_buf.data()));
    task_data->inputs_count.push_back(in_buf.size());
    task_data->outputs.push_back(reinterpret_cast<uint8_t*>(out_buf.data()));
    task_data->outputs_count.push_back(out_buf.size());
  }

  auto task = std::make_shared<borisov_s_strassen_all::ParallelStrassenMpiStl>(task_data);

  auto perf_attr = std::make_shared<ppc::core::PerfAttr>();
  perf_attr->num_running = 5;
  const auto t0 = std::chrono::high_resolution_clock::now();
  perf_attr->current_timer = [&] {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now - t0).count();
  };

  auto perf_results = std::make_shared<ppc::core::PerfResults>();
  auto perf = std::make_shared<ppc::core::Perf>(task);

  perf->TaskRun(perf_attr, perf_results);
  if (world.rank() == 0) {
    ppc::core::Perf::PrintPerfStatistic(perf_results);
    auto expected = MultiplyNaive(a, b, kRowsA, kColsA, kColsB);
    std::vector<double> got(out_buf.begin() + 2, out_buf.end());
    ASSERT_EQ(expected.size(), got.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_NEAR(expected[i], got[i], 1e-8);
    }
  }
}
