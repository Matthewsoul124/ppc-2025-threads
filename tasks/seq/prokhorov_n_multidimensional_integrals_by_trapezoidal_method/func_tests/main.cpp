#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "seq/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_seq.hpp"

constexpr double kPi = std::numbers::pi;

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq, test_integral_1d_quadratic) {
  std::vector<double> lower = {0.0};
  std::vector<double> upper = {1.0};
  std::vector<int> steps = {10000};
  double expected = 1.0 / 3.0;
  double result = 0.0;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_seq->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_seq->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_seq->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_seq->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq::TestTaskSequential test_task_sequential(
      task_data_seq);

  test_task_sequential.SetFunction([](const std::vector<double>& point) { return point[0] * point[0]; });

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-4);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq, test_integral_1d_sin) {
  std::vector<double> lower = {0.0};
  std::vector<double> upper = {kPi / 2};
  std::vector<int> steps = {10000};
  double expected = 1.0;
  double result = 0.0;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_seq->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_seq->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_seq->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_seq->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq::TestTaskSequential test_task_sequential(
      task_data_seq);

  test_task_sequential.SetFunction([](const std::vector<double>& point) { return std::sin(point[0]); });

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-5);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq, test_integral_2d_circle_area) {
  std::vector<double> lower = {-1.0, -1.0};
  std::vector<double> upper = {1.0, 1.0};
  std::vector<int> steps = {500, 500};
  double expected = kPi;
  double result = 0.0;

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_seq->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_seq->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_seq->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_seq->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_seq::TestTaskSequential test_task_sequential(
      task_data_seq);

  test_task_sequential.SetFunction(
      [](const std::vector<double>& point) { return (point[0] * point[0] + point[1] * point[1] <= 1.0) ? 1.0 : 0.0; });

  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-2);
}