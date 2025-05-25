#include <gtest/gtest.h>

#include <algorithm>
#include <boost/mpi/communicator.hpp>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>

#include "all/gusev_n_sorting_int_simple_merging/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {
ppc::core::TaskDataPtr CreateTaskData(std::vector<int>& input, std::vector<int>& output) {
  auto task_data = std::make_shared<ppc::core::TaskData>();

  task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(input.data()));
  task_data->inputs_count.emplace_back(input.size());
  task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(output.data()));
  task_data->outputs_count.emplace_back(output.size());

  return task_data;
}

void RunT(ppc::core::TaskDataPtr& task_data) {
  gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL task(task_data);
  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
}

void CheckResults(const std::vector<int>& expected, const std::vector<int>& out) {
  boost::mpi::communicator world;
  if (world.rank() == 0) {
    EXPECT_EQ(expected, out);
  }
}
}  // namespace

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_basic) {
  std::vector<int> in = {170, 45, 75, 90, 802, 24, 2, 66};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_all_zeroes) {
  std::vector<int> in = {0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_empty) {
  std::vector<int> in;
  std::vector<int> out;

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  boost::mpi::communicator world;
  if (world.rank() == 0) {
    EXPECT_TRUE(out.empty());
  }
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_single_element) {
  std::vector<int> in = {42};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  CheckResults(in, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_negative_numbers) {
  std::vector<int> in = {3, -1, 0, -5, 2, -3};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_random) {
  size_t size = 1000;
  std::vector<int> in(size);

  std::ranges::generate(in, []() { return (std::rand() % 2000) - 1000; });

  std::vector<int> out(in.size());
  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_duplicates) {
  std::vector<int> in = {5, 3, 5, -2, 3, -2, -2, 5, 0, 0, 7, 7, -7};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_reverse_order) {
  std::vector<int> in = {9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_validation_empty) {
  std::vector<int> in;
  std::vector<int> out;

  auto task_data = CreateTaskData(in, out);
  gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL task(task_data);
  EXPECT_TRUE(task.Validation());
}

TEST(gusev_n_sorting_int_simple_merging_all, test_validation_size_mismatch) {
  std::vector<int> in = {1, 2, 3};
  std::vector<int> out(2);

  auto task_data = CreateTaskData(in, out);
  gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_int_min_max) {
  std::vector<int> in = {std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_all_negative) {
  std::vector<int> in = {-5, -10, -15, -20, -25};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_all_positive) {
  std::vector<int> in = {5, 10, 15, 20, 25};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_small_large_values) {
  std::vector<int> in = {1, 1000000, 5, 5000000, 10, 10000000};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  std::ranges::sort(expected);
  CheckResults(expected, out);
}

TEST(gusev_n_sorting_int_simple_merging_all, test_validation_nullptr) {
  std::vector<int> in = {1, 2, 3};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  task_data->inputs[0] = nullptr;

  gusev_n_sorting_int_simple_merging_all::SortingIntSimpleMergingALL task(task_data);
  EXPECT_FALSE(task.Validation());
}

TEST(gusev_n_sorting_int_simple_merging_all, test_radix_sort_already_sorted) {
  std::vector<int> in = {-10, -5, 0, 5, 10, 15, 20};
  std::vector<int> out(in.size());

  auto task_data = CreateTaskData(in, out);
  RunT(task_data);

  std::vector<int> expected = in;
  CheckResults(expected, out);
}
