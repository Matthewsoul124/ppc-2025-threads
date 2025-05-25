#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "core/task/include/task.hpp"
#include "stl/opolin_d_radix_sort_batcher_merge/include/ops_stl.hpp"

namespace opolin_d_radix_batcher_sort_stl {
namespace {
void GenDataRadixSort(size_t size, std::vector<int> &vec, std::vector<int> &expected) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(-1000, 1000);
  vec.clear();
  expected.clear();
  vec.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    vec.push_back(dis(gen));
  }
  expected = vec;
  std::ranges::sort(expected);
}
}  // namespace
}  // namespace opolin_d_radix_batcher_sort_stl

TEST(opolin_d_radix_batcher_sort_stl, test_size_3) {
  int size = 3;
  std::vector<int> expected;
  std::vector<int> input;
  input = {2, 1, 10};
  expected = {1, 2, 10};

  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_size_6) {
  int size = 6;
  std::vector<int> expected;
  std::vector<int> input;
  input = {3, 1, 7, 0, 12, 2};
  expected = {0, 1, 2, 3, 7, 12};
  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_empty) {
  int size = 0;
  std::vector<int> expected;
  std::vector<int> input;
  std::vector<int> out;
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(size);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(size);

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), false);
}

TEST(opolin_d_radix_batcher_sort_stl, test_one_element) {
  int size = 1;
  std::vector<int> expected;
  std::vector<int> input;
  input = {31};
  expected = {31};
  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_negative_values) {
  int size = 5;
  std::vector<int> expected;
  std::vector<int> input;
  input = {-12, -4, -7, -2, -34};
  expected = {-34, -12, -7, -4, -2};
  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_sorted) {
  int size = 5;
  std::vector<int> expected;
  std::vector<int> input;
  input = {0, 1, 2, 6, 7};
  expected = {0, 1, 2, 6, 7};

  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_equal_values) {
  int size = 3;
  std::vector<int> expected;
  std::vector<int> input;
  input = {2, 2, 2};
  expected = {2, 2, 2};
  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_reversed) {
  int size = 7;
  std::vector<int> expected;
  std::vector<int> input;
  input = {6, 3, 2, 0, -4, -6, -10};
  expected = {-10, -6, -4, 0, 2, 3, 6};
  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_varying_digit_counts) {
  int size = 6;
  std::vector<int> expected;
  std::vector<int> input;
  input = {123456, 12, 123, 1, 12345, 1234};
  expected = {1, 12, 123, 1234, 12345, 123456};
  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_negative_size) {
  int size = -1;
  std::vector<int> expected;
  std::vector<int> input;
  std::vector<int> out;
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(size);
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(size);

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), false);
}

TEST(opolin_d_radix_batcher_sort_stl, test_size_prime_7) {
  int size = 7;
  std::vector<int> expected;
  std::vector<int> input;
  opolin_d_radix_batcher_sort_stl::GenDataRadixSort(size, input, expected);

  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}

TEST(opolin_d_radix_batcher_sort_stl, test_double_reversed_order) {
  int size = 6;
  std::vector<int> expected;
  std::vector<int> input;
  input = {3, 2, 1, 6, 5, 4};
  expected = {1, 2, 3, 4, 5, 6};

  std::vector<int> out(size, 0);
  auto task_data_stl = std::make_shared<ppc::core::TaskData>();
  task_data_stl->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_stl->inputs_count.emplace_back(out.size());
  task_data_stl->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_stl->outputs_count.emplace_back(out.size());

  auto test_task_stl = std::make_shared<opolin_d_radix_batcher_sort_stl::RadixBatcherSortTaskStl>(task_data_stl);
  ASSERT_EQ(test_task_stl->Validation(), true);
  test_task_stl->PreProcessing();
  test_task_stl->Run();
  test_task_stl->PostProcessing();
  ASSERT_EQ(out, expected);
}