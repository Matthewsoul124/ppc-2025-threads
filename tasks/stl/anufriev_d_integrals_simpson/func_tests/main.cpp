#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/anufriev_d_integrals_simpson/include/ops_stl.hpp"

namespace {
const double kPi = std::numbers::pi;

std::shared_ptr<ppc::core::TaskData> MakeTaskData(const std::vector<double>& elements,
                                                  std::vector<double>& out_buffer) {
  auto task_data = std::make_shared<ppc::core::TaskData>();
  auto* input_ptr = reinterpret_cast<uint8_t*>(const_cast<double*>(elements.data()));
  auto* output_ptr = reinterpret_cast<uint8_t*>(out_buffer.data());
  task_data->inputs.push_back(input_ptr);
  task_data->inputs_count.push_back(static_cast<uint32_t>(elements.size() * sizeof(double)));
  task_data->outputs.push_back(output_ptr);
  task_data->outputs_count.push_back(static_cast<uint32_t>(out_buffer.size() * sizeof(double)));
  return task_data;
}
}  // namespace

using namespace anufriev_d_integrals_simpson_stl;

TEST(anufriev_d_integrals_simpson_stl, test_1D_sin) {
  std::vector<double> in = {1, 0.0, kPi / 2.0, 100, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();
  double result = out_buffer[0];
  EXPECT_NEAR(result, 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_stl, test_2D_sum_of_squares) {
  std::vector<double> in = {2, 0.0, 1.0, 100, 0.0, 1.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();
  double result = out_buffer[0];
  EXPECT_NEAR(result, 2.0 / 3.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_stl, test_2D_sin_cos) {
  std::vector<double> in = {2, 0.0, kPi / 2.0, 100, 0.0, kPi / 2.0, 100, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();
  double result = out_buffer[0];
  EXPECT_NEAR(result, 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_stl, test_unknown_func) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  task.Run();
  task.PostProcessing();
  double result = out_buffer[0];
  EXPECT_DOUBLE_EQ(result, 0.0);
}

TEST(anufriev_d_integrals_simpson_stl, test_invalid_empty_input) {
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(nullptr);
  td->inputs_count.push_back(0);
  td->outputs.push_back(nullptr);
  td->outputs_count.push_back(0);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_stl, test_invalid_dimension_zero) {
  std::vector<double> in = {0, 0.0, 1.0, 2, 999};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_stl, test_invalid_not_enough_data) {
  std::vector<double> in = {2, 0.0, 1.0, 2.0, 999.0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_stl, test_invalid_odd_n) {
  std::vector<double> in = {1, 0.0, 1.0, 3, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_stl, test_invalid_negative_n) {
  std::vector<double> in = {1, 0.0, 1.0, -2, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}

TEST(anufriev_d_integrals_simpson_stl, test_no_output_buffer) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<std::uint32_t>(in.size()));
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.Validation());
}

TEST(anufriev_d_integrals_simpson_stl, MinValidIterationsN0_is_2_1D) {
  std::vector<double> in = {1, 0.0, 3.0, 2, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 9.0, 1e-5);
}

TEST(anufriev_d_integrals_simpson_stl, MinValidIterationsN0_is_2_2D) {
  std::vector<double> in = {2, 0.0, 1.0, 2, 0.0, 1.0, 2, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 2.0 / 3.0, 1e-5);
}

TEST(anufriev_d_integrals_simpson_stl, FewIterationsN0_is_4_1D) {
  std::vector<double> in = {1, 0.0, kPi, 4, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 2.0, 5e-3);
}

TEST(anufriev_d_integrals_simpson_stl, ManyIterationsN0_1D) {
  std::vector<double> in = {1, 0.0, 1.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 1.0 / 3.0, 1e-4);
}

TEST(anufriev_d_integrals_simpson_stl, MultiDim_N_Mismatch) {
  std::vector<double> in = {2, 0.0, 1.0, 100, 0.0, 1.0, 2, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 2.0 / 3.0, 1e-4);
}

TEST(anufriev_d_integrals_simpson_stl, MultiDim_N_Mismatch_Reversed) {
  std::vector<double> in = {2, 0.0, 1.0, 2, 0.0, 1.0, 100, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 2.0 / 3.0, 1e-4);
}

TEST(anufriev_d_integrals_simpson_stl, Preprocessing_InvalidN_Zero) {
  std::vector<double> in = {1, 0.0, 1.0, 0, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_FALSE(task.PreProcessing());
}

TEST(anufriev_d_integrals_simpson_stl, Preprocessing_InvalidN_Odd) {
  std::vector<double> in = {1, 0.0, 1.0, 3, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_FALSE(task.PreProcessing());
}

TEST(anufriev_d_integrals_simpson_stl, ManyIterations_PossibleSingleThreadPath) {
  std::vector<double> in = {1, 0.0, 2.0, 200, 0};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 8.0 / 3.0, 1e-4);
}

TEST(anufriev_d_integrals_simpson_stl, TestWithFuncCode1_2D) {
  std::vector<double> in = {2, 0.0, kPi / 2.0, 50, 0.0, kPi / 2.0, 50, 1};
  std::vector<double> out_buffer(1, 0.0);
  auto td = MakeTaskData(in, out_buffer);
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out_buffer[0], 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_stl, Run_Without_PreProcessing) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out = {0.0};
  auto td = MakeTaskData(in, out);
  IntegralsSimpsonSTL task(td);
  EXPECT_ANY_THROW(task.Run());
}

TEST(anufriev_d_integrals_simpson_stl, PostProcessing_Only) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out = {0.0};
  auto td = MakeTaskData(in, out);
  IntegralsSimpsonSTL task(td);
  EXPECT_ANY_THROW(task.PostProcessing());
}

TEST(anufriev_d_integrals_simpson_stl, Validation_EmptyOutputsCount) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  std::vector<double> out(1, 0.0);
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<uint32_t>(in.size() * sizeof(double)));
  td->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.Validation());
}

TEST(anufriev_d_integrals_simpson_stl, 3D_SumOfSquares) {
  std::vector<double> in = {3, 0.0, 1.0, 10, 0.0, 2.0, 10, 0.0, 3.0, 10, 0};
  std::vector<double> out(1, 0.0);
  auto td = MakeTaskData(in, out);
  IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_GT(out[0], 0.0);
}

TEST(anufriev_d_integrals_simpson_stl, 1D_SinFunction) {
  std::vector<double> in = {1, 0.0, kPi / 2, 100, 1};
  std::vector<double> out(1, 0.0);
  auto td = MakeTaskData(in, out);
  IntegralsSimpsonSTL task(td);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  EXPECT_NEAR(out[0], 1.0, 1e-3);
}

TEST(anufriev_d_integrals_simpson_stl, test_ValidationImpl_no_outputs) {
  std::vector<double> in = {1, 0.0, 1.0, 2, 0};
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(const_cast<double*>(in.data())));
  td->inputs_count.push_back(static_cast<uint32_t>(in.size() * sizeof(double)));
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.ValidationImpl());
}

TEST(anufriev_d_integrals_simpson_stl, test_PreProcessingImpl_small_input_size) {
  double dummy = 0.0;
  auto td = std::make_shared<ppc::core::TaskData>();
  td->inputs.push_back(reinterpret_cast<uint8_t*>(&dummy));
  td->inputs_count.push_back(static_cast<uint32_t>(sizeof(double) / 2));
  std::vector<double> out(1);
  td->outputs.push_back(reinterpret_cast<uint8_t*>(out.data()));
  td->outputs_count.push_back(static_cast<uint32_t>(sizeof(double)));
  anufriev_d_integrals_simpson_stl::IntegralsSimpsonSTL task(td);
  EXPECT_FALSE(task.PreProcessingImpl());
}  // namespace anufriev_d_integrals_simpson_stl