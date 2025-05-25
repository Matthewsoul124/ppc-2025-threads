#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/prokhorov_n_multidimensional_integrals_by_trapezoidal_method/include/ops_stl.hpp"

constexpr double kPi = std::numbers::pi;

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl, test_integral_1d_quadratic) {
  std::vector<double> lower = {0.0};
  std::vector<double> upper = {1.0};
  std::vector<int> steps = {10000};
  double expected = 1.0 / 3.0;
  double result = 0.0;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_stl->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_stl->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_stl->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_stl->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl::TestTaskSTL test_task_stl(task_data_stl);

  test_task_stl.SetFunction([](const std::vector<double>& point) { return point[0] * point[0]; });

  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-4);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl, test_integral_1d_sin) {
  std::vector<double> lower = {0.0};
  std::vector<double> upper = {kPi / 2};
  std::vector<int> steps = {10000};
  double expected = 1.0;
  double result = 0.0;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_stl->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_stl->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_stl->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_stl->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl::TestTaskSTL test_task_stl(task_data_stl);

  test_task_stl.SetFunction([](const std::vector<double>& point) { return std::sin(point[0]); });

  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-5);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl, test_integral_2d_circle_area) {
  std::vector<double> lower = {-1.0, -1.0};
  std::vector<double> upper = {1.0, 1.0};
  std::vector<int> steps = {500, 500};
  double expected = kPi;
  double result = 0.0;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_stl->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_stl->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_stl->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_stl->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl::TestTaskSTL test_task_stl(task_data_stl);

  test_task_stl.SetFunction(
      [](const std::vector<double>& point) { return (point[0] * point[0] + point[1] * point[1] <= 1.0) ? 1.0 : 0.0; });

  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-2);
}

TEST(prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl, test_integral_3d_sphere_volume) {
  std::vector<double> lower = {-1.0, -1.0, -1.0};
  std::vector<double> upper = {1.0, 1.0, 1.0};
  std::vector<int> steps = {100, 100, 100};
  double expected = 4.0 * kPi / 3.0;
  double result = 0.0;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(lower.data()));
  task_data_stl->inputs_count.emplace_back(lower.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(upper.data()));
  task_data_stl->inputs_count.emplace_back(upper.size() * sizeof(double));
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t*>(steps.data()));
  task_data_stl->inputs_count.emplace_back(steps.size() * sizeof(int));
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t*>(&result));
  task_data_stl->outputs_count.emplace_back(sizeof(double));

  prokhorov_n_multidimensional_integrals_by_trapezoidal_method_stl::TestTaskSTL test_task_stl(task_data_stl);

  test_task_stl.SetFunction([](const std::vector<double>& point) {
    return (point[0] * point[0] + point[1] * point[1] + point[2] * point[2] <= 1.0) ? 1.0 : 0.0;
  });

  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_NEAR(result, expected, 1e-1);
}