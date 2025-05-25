#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/fyodorov_m_shell_sort_with_even_odd_batcher_merge/include/ops_stl.hpp"

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_small_array) {
  std::vector<int> input = {35, 33, 42, 10, 14, 19, 27, 44};
  std::vector<int> expected_output = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_small_array_with_negative_values) {
  std::vector<int> input = {35, 33, 42, -10, -14, 19, 27, 44};
  std::vector<int> expected_output = {-14, -10, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_random_sequence) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> distrib(-100, 100);

  const size_t size = 10;
  std::vector<int> input(size);
  for (auto &num : input) {
    num = distrib(gen);
  }

  std::vector<int> expected_output = input;
  std::ranges::sort(expected_output);

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_sorted_array) {
  std::vector<int> input = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> expected_output = {10, 14, 19, 27, 33, 35, 42, 44};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_empty_array) {
  std::vector<int> input;
  std::vector<int> expected_output;
  std::vector<int> output;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, reverse_pozitive_array) {
  int arrsize = 12;
  std::vector<int> in = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> out(arrsize, 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, reverse_pozitive_double) {
  int arrsize = 10;
  std::vector<int> in = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
  std::vector<int> out(arrsize, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());
  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();
  std::ranges::sort(in);
  EXPECT_EQ(in, out);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_validation_failure) {
  std::vector<int> input = {35, 33, 42, 10};
  std::vector<int> output;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);

  ASSERT_EQ(test_task_stl.Validation(), false);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_empty_array_validation) {
  std::vector<int> input;
  std::vector<int> output;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);

  ASSERT_EQ(test_task_stl.Validation(), true);

  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, input);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_single_element_array) {
  std::vector<int> input = {42};
  std::vector<int> expected_output = {42};
  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);

  ASSERT_EQ(test_task_stl.Validation(), true);

  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_two_element) {
  std::vector<int> input = {5, 2};
  std::vector<int> expected_output = {2, 5};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_reverse_negative_array) {
  std::vector<int> input = {-1, -2, -3};
  std::vector<int> expected_output = {-3, -2, -1};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_reverse_negative_duplicate_sequence) {
  std::vector<int> input = {3, 3, 2, 2, 1, 1};
  std::vector<int> expected_output = {1, 1, 2, 2, 3, 3};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_partially_sorted_array) {
  std::vector<int> input = {1, 3, 2, 4, 6, 5};
  std::vector<int> expected_output = {1, 2, 3, 4, 5, 6};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_mixed_even_odd) {
  std::vector<int> input = {7, 2, 5, 4, 3, 6};
  std::vector<int> expected_output = {2, 3, 4, 5, 6, 7};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}

TEST(fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl, test_alternating_positive_negative) {
  std::vector<int> input = {1, -2, 3, -4, 5, -6};
  std::vector<int> expected_output = {-6, -4, -2, 1, 3, 5};

  std::vector<int> output(input.size(), 0);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(input.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(output.data()));
  task_data_stl->outputs_count.emplace_back(output.size());

  fyodorov_m_shell_sort_with_even_odd_batcher_merge_stl::TestTaskSTL test_task_stl(task_data_stl);
  ASSERT_EQ(test_task_stl.Validation(), true);
  test_task_stl.PreProcessing();
  test_task_stl.Run();
  test_task_stl.PostProcessing();

  EXPECT_EQ(output, expected_output);
}