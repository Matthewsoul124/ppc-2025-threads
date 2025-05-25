#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/shlyakov_m_shell_sort/include/ops_stl.hpp"

namespace {
std::vector<int> GenerateRandomArray(size_t size) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::uniform_int_distribution<int> distribution_range(-1000, 1000);
  int min_val = distribution_range(generator);
  int max_val = distribution_range(generator);

  if (min_val > max_val) {
    std::swap(min_val, max_val);
  }

  std::uniform_int_distribution<int> distribution(min_val, max_val);

  std::vector<int> arr(size);
  for (size_t i = 0; i < size; ++i) {
    arr[i] = distribution(generator);
  }
  return arr;
}

bool IsSorted(const std::vector<int> &arr) {
  if (arr.empty()) {
    return true;
  }
  for (size_t i = 1; i < arr.size(); ++i) {
    if (arr[i - 1] > arr[i]) {
      return false;
    }
  }
  return true;
}
}  // namespace

TEST(shlyakov_m_shell_sort_stl, Test_Empty_Array) {
  std::vector<int> in;
  std::vector<int> out;

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
}

TEST(shlyakov_m_shell_sort_stl, Test_Already_Sorted_Array) {
  std::vector<int> in = {1, 2, 3, 4, 5};
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  EXPECT_EQ(in, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Reverse_Sorted_Array) {
  std::vector<int> in = {5, 4, 3, 2, 1};
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = {1, 2, 3, 4, 5};
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_Small) {
  std::vector<int> in = GenerateRandomArray(10);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_Large) {
  size_t array_size = 200;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_Simple_Size) {
  size_t array_size = 241;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_500) {
  size_t array_size = 500;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_501) {
  size_t array_size = 501;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_1000) {
  size_t array_size = 1000;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_1001) {
  size_t array_size = 1001;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_999) {
  size_t array_size = 999;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_10000) {
  size_t array_size = 10000;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_10001) {
  size_t array_size = 10001;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_9999) {
  size_t array_size = 9999;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_15000) {
  size_t array_size = 15000;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_15001) {
  size_t array_size = 15001;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Random_Array_With_14999) {
  size_t array_size = 14999;
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_With_Eq_Numbers) {
  size_t array_size = 100;
  std::vector<int> in(array_size, 3);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_With_False_Validation) {
  size_t array_size = 100;
  std::vector<int> in(array_size, 3);
  std::vector<int> out(in.size() - 1);

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_FALSE(test_task_tbb.Validation());
}

TEST(shlyakov_m_shell_sort_stl, Test_Mersenne_3) {
  size_t array_size = 3;  // M_2
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Mersenne_7) {
  size_t array_size = 7;  // M_3
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Mersenne_31) {
  size_t array_size = 31;  // M_5
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Mersenne_127) {
  size_t array_size = 127;  // M_7
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}

TEST(shlyakov_m_shell_sort_stl, Test_Mersenne_8191) {
  size_t array_size = 8191;  // M_13
  std::vector<int> in = GenerateRandomArray(array_size);
  std::vector<int> out(in.size());

  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_stl->inputs_count.emplace_back(in.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  shlyakov_m_shell_sort_stl::TestTaskSTL test_task_tbb(task_data_stl);
  ASSERT_TRUE(test_task_tbb.Validation());
  ASSERT_TRUE(test_task_tbb.PreProcessing());
  ASSERT_TRUE(test_task_tbb.Run());
  ASSERT_TRUE(test_task_tbb.PostProcessing());

  EXPECT_TRUE(IsSorted(out));
  std::vector<int> expected = in;
  std::ranges::sort(expected);
  EXPECT_EQ(expected, out);
}
