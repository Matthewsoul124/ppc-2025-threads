#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/bessonov_e_radix_sort_simple_merging/include/ops_stl.hpp"

namespace {
std::vector<double> GenerateVector(std::size_t n, double first, double last) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dist(first, last);

  std::vector<double> result(n);
  for (std::size_t i = 0; i < n; ++i) {
    result[i] = dist(gen);
  }
  return result;
}
}  // namespace

TEST(bessonov_e_radix_sort_simple_merging_stl, FirstTest) {
  std::vector<double> input_vector = {3.4, 1.2, 0.5, 7.8, 2.3, 4.5, 6.7, 8.9, 1.0, 0.2, 5.6, 4.3, 9.1, 1.5, 3.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, SingleElementTest) {
  std::vector<double> input_vector = {42.0};
  std::vector<double> output_vector(1, 0.0);
  std::vector<double> result_vector = {42.0};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, NegativeAndPositiveTest) {
  std::vector<double> input_vector = {-3.2, 1.1, -7.5, 0.0, 4.4, -2.2, 3.3};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-7.5, -3.2, -2.2, 0.0, 1.1, 3.3, 4.4};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, RandomVectorTest) {
  const std::size_t n = 1000;
  std::vector<double> input_vector = GenerateVector(n, -1000.0, 1000.0);
  std::vector<double> output_vector(n, 0.0);

  std::vector<double> result_vector = input_vector;
  std::ranges::sort(result_vector);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, AllSameElementsTest) {
  std::vector<double> input_vector = {3.14, 3.14, 3.14, 3.14};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {3.14, 3.14, 3.14, 3.14};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, ExtremeValuesTest) {
  std::vector<double> input_vector = {std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest(), 0.0,
                                      -42.5, 100.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {std::numeric_limits<double>::lowest(), -42.5, 0.0, 100.0,
                                       std::numeric_limits<double>::max()};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, TinyNumbersTest) {
  std::vector<double> input_vector = {1e-10, -1e-10, 1e-20, -1e-20};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-1e-10, -1e-20, 1e-20, 1e-10};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, DenormalNumbersTest) {
  std::vector<double> input_vector = {1e-310, -1e-310, 0.0};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {-1e-310, 0.0, 1e-310};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, ReverseOrderTest) {
  std::vector<double> input_vector = {9.1, 8.9, 7.8, 6.7, 5.6, 4.5, 4.3, 3.4, 3.0, 2.3, 1.5, 1.2, 1.0, 0.5, 0.2};
  std::vector<double> output_vector(input_vector.size(), 0.0);
  std::vector<double> result_vector = {0.2, 0.5, 1.0, 1.2, 1.5, 2.3, 3.0, 3.4, 4.3, 4.5, 5.6, 6.7, 7.8, 8.9, 9.1};

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input_vector.data()));
  task_data->inputs_count.emplace_back(input_vector.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output_vector.data()));
  task_data->outputs_count.emplace_back(output_vector.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL test_task(task_data);
  ASSERT_TRUE(test_task.Validation());
  test_task.PreProcessing();
  test_task.Run();
  test_task.PostProcessing();

  ASSERT_EQ(output_vector, result_vector);
}

TEST(bessonov_e_radix_sort_simple_merging_stl, Validation_NullInput) {
  std::vector<double> output(100);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(nullptr);
  task_data->inputs_count.emplace_back(100);
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_stl, Validation_NullOutput) {
  std::vector<double> input(100);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(nullptr);
  task_data->outputs_count.emplace_back(100);

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_stl, Validation_SizeMismatch) {
  std::vector<double> input(100);
  std::vector<double> output(50);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_stl, Validation_ZeroSize) {
  std::vector<double> input(0);
  std::vector<double> output(0);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_stl, Validation_SizeOverflow) {
  auto huge_size = static_cast<size_t>(INT_MAX) + 1;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(nullptr);
  task_data->inputs_count.emplace_back(huge_size);
  task_data->outputs.emplace_back(nullptr);
  task_data->outputs_count.emplace_back(huge_size);

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_stl, Validation_EmptyCounts) {
  std::vector<double> input(100);
  std::vector<double> output(100);
  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_FALSE(task.Validation());
}

TEST(bessonov_e_radix_sort_simple_merging_stl, Validation_MaxSize) {
  auto max_size = static_cast<size_t>(INT_MAX);

  auto* dummy_input = new uint8_t[1];
  auto* dummy_output = new uint8_t[1];

  auto task_data = std::make_shared<ppc::core::TaskData>();
  task_data->inputs.emplace_back(dummy_input);
  task_data->inputs_count.emplace_back(max_size);
  task_data->outputs.emplace_back(dummy_output);
  task_data->outputs_count.emplace_back(max_size);

  bessonov_e_radix_sort_simple_merging_stl::TestTaskSTL task(task_data);
  ASSERT_TRUE(task.Validation());

  delete[] dummy_input;
  delete[] dummy_output;
}