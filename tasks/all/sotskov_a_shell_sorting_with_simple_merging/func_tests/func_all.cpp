#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "all/sotskov_a_shell_sorting_with_simple_merging/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace sotskov_a_shell_sorting_with_simple_merging_all {
namespace {
struct RandomVectorParams {
  int size;
  int min_value;
  int max_value;
};

struct SortingTestParams {
  std::vector<int> expected;
  std::vector<int> input;
};

std::vector<int> GenerateRandomVector(const RandomVectorParams &params) {
  std::random_device random_device;
  std::mt19937 generator(random_device());
  std::uniform_int_distribution<int> distribution(params.min_value, params.max_value);

  std::vector<int> random_vector(params.size);
  for (int &element : random_vector) {
    element = distribution(generator);
  }
  return random_vector;
}

void RunSortingTest(SortingTestParams &params) {
  boost::mpi::communicator world;
  std::vector<int> out(params.input.size());
  std::vector<int> expected_sorted = params.expected;
  ShellSortWithSimpleMerging(expected_sorted);

  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t *>(params.input.data()));
    task_data->inputs_count.emplace_back(params.input.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
    task_data->outputs_count.emplace_back(out.size());
  }

  sotskov_a_shell_sorting_with_simple_merging_all::TestTaskALL task(task_data);
  ASSERT_TRUE(task.ValidationImpl());
  ASSERT_TRUE(task.PreProcessingImpl());
  ASSERT_TRUE(task.RunImpl());
  ASSERT_TRUE(task.PostProcessingImpl());

  if (world.rank() == 0) {
    ASSERT_EQ(out, expected_sorted);
  }
}
}  // namespace
}  // namespace sotskov_a_shell_sorting_with_simple_merging_all

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_positive_numbers) {
  std::vector<int> input = {5, 1, 8, 6, 2, 7, 1, 4};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_negative_numbers) {
  std::vector<int> input = {-8, -3, -12, -7, -4, -10};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_ordered_array) {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_with_duplicates) {
  std::vector<int> input = {4, 2, 2, 8, 4, 6, 6, 2};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_single_element) {
  std::vector<int> input = {77};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_empty_array) {
  std::vector<int> input = {};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_random_vector) {
  sotskov_a_shell_sorting_with_simple_merging_all::RandomVectorParams rparams{
      .size = 20, .min_value = -100, .max_value = 100};
  auto input = sotskov_a_shell_sorting_with_simple_merging_all::GenerateRandomVector(rparams);
  auto expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_reverse_sorted_array) {
  std::vector<int> input = {9, 8, 7, 6, 5, 4, 3, 2, 1};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_large_array) {
  std::vector<int> input(1023);
  std::iota(input.rbegin(), input.rend(), 1);
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}

TEST(sotskov_a_shell_sorting_with_simple_merging_all, test_sort_complex_double_reverse_pattern) {
  std::vector<int> input = {9, 7, 5, 3, 1, 9, 7, 5, 3, 1, 8, 6, 4, 2, 0, 8, 6, 4, 2, 0};
  std::vector<int> expected = input;
  sotskov_a_shell_sorting_with_simple_merging_all::ShellSortWithSimpleMerging(expected);

  sotskov_a_shell_sorting_with_simple_merging_all::SortingTestParams params{.expected = expected, .input = input};
  sotskov_a_shell_sorting_with_simple_merging_all::RunSortingTest(params);
}