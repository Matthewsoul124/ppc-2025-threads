#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "../include/integrate_mpi.hpp"
#include "boost/mpi/communicator.hpp"
#include "core/task/include/task.hpp"

using namespace khasanyanov_k_trapezoid_method_all;

TEST(khasanyanov_k_trapezoid_method_all, test_integrate_1) {
  boost::mpi::communicator world;
  constexpr double kPrecision = 0.01;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return (5 * x[0]) + (2 * x[1]) - (3 * x[2]); };

  IntegrationBounds bounds = {{-3, 1.0}, {0.0, 2.0}, {0.5, 1.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodALL::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodALL task(task_data_seq, f);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(-21.0, result, kPrecision);
  }
}

TEST(khasanyanov_k_trapezoid_method_all, test_integrate_2) {
  boost::mpi::communicator world;
  constexpr double kPrecision = 0.01;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return (x[0] * x[0]) + (2 * x[1]) - (6.5 * x[2]); };

  IntegrationBounds bounds = {{-3, -2.0}, {0.0, 2.0}, {0.5, 1.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodALL::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodALL task(task_data_seq, f);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(3.4583, result, kPrecision);
  }
}

TEST(khasanyanov_k_trapezoid_method_all, test_integrate_3) {
  boost::mpi::communicator world;
  constexpr double kPrecision = 0.01;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return sin(x[0]) - x[1]; };

  IntegrationBounds bounds = {{0.0, 1.0}, {0.0, 2.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodALL::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodALL task(task_data_seq, f);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(-1.08060, result, kPrecision);
  }
}

TEST(khasanyanov_k_trapezoid_method_all, test_integrate_4) {
  boost::mpi::communicator world;
  constexpr double kPrecision = 0.01;
  double result{};
  auto f = [](const std::vector<double>& x) -> double { return (7.4 * x[0]) - (x[1] * x[1]); };

  IntegrationBounds bounds = {{-50.0, -47.0}, {-2.0, -1.0}};

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodALL::CreateTaskData(task_data_seq, context, &result);
  TrapezoidalMethodALL task(task_data_seq, f);

  ASSERT_TRUE(task.Validation());

  task.PreProcessing();
  task.Run();
  task.PostProcessing();
  if (world.rank() == 0) {
    ASSERT_NEAR(-1083.7, result, kPrecision);
  }
}

TEST(khasanyanov_k_trapezoid_method_all, test_invalid_input) {
  boost::mpi::communicator world;
  constexpr double kPrecision = 0.01;
  auto f = [](const std::vector<double>& x) -> double { return sin(x[0]) - x[1]; };

  IntegrationBounds bounds;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  TaskContext context{.function = f, .bounds = bounds, .precision = kPrecision};
  TrapezoidalMethodALL::CreateTaskData(task_data_seq, context, nullptr);
  TrapezoidalMethodALL task(task_data_seq, f);

  if (world.rank() == 0) {
    ASSERT_FALSE(task.Validation());
  }
}
